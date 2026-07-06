# -*- coding: utf-8 -*-
"""联邦框架：单塔模型 + 两方 VFL 模型（配置驱动）。

设计目标：一个可复用、可审计的纵向联邦框架。
- 非标签方 B：本地子模型 fB，只向 A 发送隐表示 h_N（切割层唯一出域内容）。
- 标签方 A：本地 bottom fA，收到 h_N 后在 top 网络融合 [h_N; h_L] 出预测。
- 切割层：前向传 h_N、反向传 ∂L/∂h_N；防御钩子在梯度出域前扰动；通信逐笔记账。
- 增强：可选本地分支(top_local)用未对齐样本+合成嵌入交替训练(HeuristicVFL)。
"""
import numpy as np

from layers import MLP, CrossNet, EmbeddingBag, bce, sigmoid


class Tower:
    """单方/集中式模型：Embedding [+CrossNet] + MLP + logistic。"""
    def __init__(self, vocab_sizes, cfg, rng):
        d = cfg["emb_dim"]
        self.emb = EmbeddingBag(vocab_sizes, d, rng, lr=cfg["lr"])
        in_dim = len(vocab_sizes) * d
        self.cross = CrossNet(in_dim, cfg["cross_layers"], rng, lr=cfg["lr"]) if cfg["cross_layers"] > 0 else None
        self.mlp = MLP([in_dim] + cfg["a_hidden"] + [1], rng, lr=cfg["lr"],
                       last_linear=True, dropout=cfg["dropout"])

    def _forward(self, codes, train):
        e = self.emb.forward(codes)
        h = self.cross.forward(e, train=train) if self.cross else e
        return e, self.mlp.forward(h, train=train)[:, 0]

    def train_step(self, codes, y):
        e, z = self._forward(codes, True)
        p = sigmoid(z)
        dz = ((p - y) / len(y)).astype(np.float32)[:, None]
        g = self.mlp.backward(dz)
        if self.cross:
            g = self.cross.backward(g)
        self.emb.backward(codes, g)
        return bce(y, p)

    def predict(self, codes, batch=100000):
        out = np.empty(len(codes), dtype=np.float32)
        for s in range(0, len(codes), batch):
            _, z = self._forward(codes[s:s + batch], False)
            out[s:s + batch] = sigmoid(z)
        return out

    def arrays(self):
        a = self.emb.arrays() + self.mlp.arrays()
        if self.cross:
            a = a + self.cross.arrays()
        return a


class VFLModel:
    """两方拆分学习框架。切割层只交换 h_N 与 ∂L/∂h_N。"""
    def __init__(self, vocab_a, vocab_b, cfg, rng):
        d = cfg["emb_dim"]
        self.cfg = cfg
        self.H_N = cfg["b_hidden"][-1]
        self.H_L = cfg["a_hidden"][-1]
        # 非标签方 B
        self.embB = EmbeddingBag(vocab_b, d, rng, lr=cfg["lr"])
        self.mlpB = MLP([len(vocab_b) * d] + cfg["b_hidden"], rng, lr=cfg["lr"], dropout=cfg["dropout"])
        # 标签方 A
        self.embA = EmbeddingBag(vocab_a, d, rng, lr=cfg["lr"])
        self.bottomA = MLP([len(vocab_a) * d] + cfg["a_hidden"], rng, lr=cfg["lr"], dropout=cfg["dropout"])
        top_in = self.H_N + self.H_L
        self.top_cross = CrossNet(top_in, cfg["cross_layers"], rng, lr=cfg["lr"]) if cfg["cross_layers"] > 0 else None
        self.top = MLP([top_in] + cfg["top_hidden"] + [1], rng, lr=cfg["lr"], last_linear=True, dropout=cfg["dropout"])
        # 增强用本地分支（共享 embA/bottomA，仅 top 独立）
        if cfg.get("enhance"):
            self.top_local = MLP([top_in] + cfg["top_hidden"] + [1], rng, lr=cfg["lr"], last_linear=True, dropout=cfg["dropout"])
        self.comm_bytes = 0
        self.steps = 0

    def _top_forward(self, hN, hL, train, local=False):
        cat = np.concatenate([hN, hL], axis=1)
        h = self.top_cross.forward(cat, train=train) if self.top_cross else cat
        net = self.top_local if local else self.top
        return cat, net.forward(h, train=train)[:, 0]

    def _top_backward(self, dz, local=False):
        net = self.top_local if local else self.top
        g = net.backward(dz)
        if self.top_cross:
            g = self.top_cross.backward(g)
        return g  # 对 [h_N; h_L] 的梯度

    def train_step(self, codes_a, codes_b, y, defense=None, attack_log=None):
        eB = self.embB.forward(codes_b)
        hN = self.mlpB.forward(eB, train=True)
        self.comm_bytes += hN.nbytes                 # B → A
        eA = self.embA.forward(codes_a)
        hL = self.bottomA.forward(eA, train=True)
        _, z = self._top_forward(hN, hL, True)
        p = sigmoid(z)
        dz = ((p - y) / len(y)).astype(np.float32)[:, None]
        g_cat = self._top_backward(dz)
        g_hN, g_hL = g_cat[:, :self.H_N], g_cat[:, self.H_N:]
        # A 本地反向
        geA = self.bottomA.backward(g_hL)
        self.embA.backward(codes_a, geA)
        # 切割层梯度出域前扰动
        g_send = defense(g_hN, y) if defense is not None else g_hN
        if attack_log is not None:
            attack_log["norms"].append(np.linalg.norm(g_send, axis=1))
            attack_log["labels"].append(y.copy())
        self.comm_bytes += g_send.nbytes             # A → B
        geB = self.mlpB.backward(g_send.astype(np.float32))
        self.embB.backward(codes_b, geB)
        self.steps += 1
        return bce(y, p)

    def train_step_unaligned(self, codes_a, h_tilde, y):
        """增强：本地分支用合成 h̃_N + 未对齐样本，监督信号入共享 embA/bottomA。"""
        eA = self.embA.forward(codes_a)
        hL = self.bottomA.forward(eA, train=True)
        _, z = self._top_forward(h_tilde, hL, True, local=True)
        p = sigmoid(z)
        dz = ((p - y) / len(y)).astype(np.float32)[:, None]
        g_cat = self._top_backward(dz, local=True)
        g_hL = g_cat[:, self.H_N:]
        geA = self.bottomA.backward(g_hL)
        self.embA.backward(codes_a, geA)
        return bce(y, p)

    def forward_hN(self, codes_b, batch=100000):
        out = np.empty((len(codes_b), self.H_N), dtype=np.float32)
        for s in range(0, len(codes_b), batch):
            eB = self.embB.forward(codes_b[s:s + batch])
            out[s:s + batch] = self.mlpB.forward(eB, train=False)
            self.comm_bytes += out[s:s + batch].nbytes
        return out

    def predict(self, codes_a, codes_b, batch=100000):
        out = np.empty(len(codes_a), dtype=np.float32)
        for s in range(0, len(codes_a), batch):
            eB = self.embB.forward(codes_b[s:s + batch])
            hN = self.mlpB.forward(eB, train=False)
            eA = self.embA.forward(codes_a[s:s + batch])
            hL = self.bottomA.forward(eA, train=False)
            _, z = self._top_forward(hN, hL, False)
            out[s:s + batch] = sigmoid(z)
        return out

    def arrays(self):
        a = (self.embB.arrays() + self.mlpB.arrays() + self.embA.arrays()
             + self.bottomA.arrays() + self.top.arrays())
        if self.top_cross:
            a = a + self.top_cross.arrays()
        return a


# ---------------- 防御算子 ----------------
class MixPro:
    def __init__(self, alpha=0.6, phi_goal=np.sqrt(3) / 2, seed=42):
        self.alpha, self.phi_goal, self.rng = alpha, phi_goal, np.random.default_rng(seed)

    def __call__(self, g, y=None):
        B = len(g)
        lam = np.maximum(self.rng.beta(self.alpha, self.alpha, (B, 1)), 0.5).astype(np.float32)
        r = self.rng.integers(0, B, B)
        g_mix = lam * g + (1 - lam) * g[r]
        g_bar = g.mean(axis=0, keepdims=True)
        nb = np.linalg.norm(g_bar) + 1e-12
        nm = np.linalg.norm(g_mix, axis=1, keepdims=True) + 1e-12
        phi = np.clip((g_mix @ g_bar.T) / (nm * nb), -1 + 1e-6, 1 - 1e-6)
        pg = self.phi_goal
        coef = nm * (pg * np.sqrt(1 - phi ** 2) - phi * np.sqrt(1 - pg ** 2)) / (nb * np.sqrt(1 - pg ** 2))
        return np.where(phi >= pg, g_mix, g_mix + coef * g_bar).astype(np.float32)
