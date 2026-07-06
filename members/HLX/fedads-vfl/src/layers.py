# -*- coding: utf-8 -*-
"""网络层：Adam / EmbeddingBag / MLP / CrossNet（numpy 手写）。

透明切割层理念（可插桩通信/防御/攻击），新增：
- Dropout（多轮训练防过拟合）
- CrossNet（DCN 风格特征交叉，用于顶层跨方交互）
"""
import numpy as np


class Adam:
    def __init__(self, shape, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.m = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0

    def step(self, w, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        w -= self.lr * mh / (np.sqrt(vh) + self.eps)

    def step_rows(self, w, rows, g_rows):
        self.t += 1
        m, v = self.m[rows], self.v[rows]
        m = self.b1 * m + (1 - self.b1) * g_rows
        v = self.b2 * v + (1 - self.b2) * g_rows * g_rows
        self.m[rows], self.v[rows] = m, v
        mh = m / (1 - self.b1 ** self.t)
        vh = v / (1 - self.b2 ** self.t)
        w[rows] -= self.lr * mh / (np.sqrt(vh) + self.eps)


class EmbeddingBag:
    def __init__(self, vocab_sizes, dim, rng, lr=1e-3):
        self.dim = dim
        self.tables = [rng.normal(0, 0.05, (vs, dim)).astype(np.float32) for vs in vocab_sizes]
        self.opts = [Adam(t.shape, lr=lr) for t in self.tables]

    def forward(self, codes):
        return np.concatenate([t[codes[:, j]] for j, t in enumerate(self.tables)], axis=1)

    def backward(self, codes, grad):
        d = self.dim
        for j, (t, opt) in enumerate(zip(self.tables, self.opts)):
            g = grad[:, j * d:(j + 1) * d]
            rows = codes[:, j]
            uniq, inv = np.unique(rows, return_inverse=True)
            acc = np.zeros((len(uniq), d), dtype=np.float32)
            np.add.at(acc, inv, g)
            opt.step_rows(t, uniq, acc)

    def n_params(self):
        return sum(t.size for t in self.tables)

    def arrays(self):
        return self.tables


class MLP:
    def __init__(self, dims, rng, lr=1e-3, last_linear=False, dropout=0.0):
        self.W, self.b, self.optW, self.optb = [], [], [], []
        for i in range(len(dims) - 1):
            w = (rng.normal(0, 1, (dims[i], dims[i + 1])) * np.sqrt(2.0 / dims[i])).astype(np.float32)
            self.W.append(w); self.b.append(np.zeros(dims[i + 1], dtype=np.float32))
            self.optW.append(Adam(w.shape, lr=lr)); self.optb.append(Adam((dims[i + 1],), lr=lr))
        self.last_linear = last_linear
        self.dropout = dropout
        self.rng = rng

    def forward(self, x, train=True):
        acts, masks = [x], []
        h = x
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            h = h @ w + b
            if not (self.last_linear and i == len(self.W) - 1):
                h = np.maximum(h, 0)
                if train and self.dropout > 0:
                    mask = (self.rng.random(h.shape) >= self.dropout).astype(np.float32) / (1 - self.dropout)
                    h = h * mask
                    masks.append(mask)
                else:
                    masks.append(None)
            else:
                masks.append(None)
            acts.append(h)
        if train:
            self.acts, self.masks = acts, masks
        return h

    def backward(self, grad):
        for i in reversed(range(len(self.W))):
            h_out, h_in = self.acts[i + 1], self.acts[i]
            if not (self.last_linear and i == len(self.W) - 1):
                if self.masks[i] is not None:
                    grad = grad * self.masks[i]
                grad = grad * (h_out > 0)
            gW = h_in.T @ grad
            gb = grad.sum(axis=0)
            grad_in = grad @ self.W[i].T
            self.optW[i].step(self.W[i], gW)
            self.optb[i].step(self.b[i], gb)
            grad = grad_in
        return grad

    def n_params(self):
        return sum(w.size + b.size for w, b in zip(self.W, self.b))

    def arrays(self):
        return self.W + self.b


class CrossNet:
    """DCN 风格特征交叉：x_{l+1} = x0 * (x_l @ w + b) + x_l，学习顶层跨方高阶交互。

    proj = x_l @ W + b 为标量投影([B,1])，故 W 形状[dim,1]、b 形状[1,1]。
    """
    def __init__(self, dim, n_layers, rng, lr=1e-3):
        self.n = n_layers
        self.W = [(rng.normal(0, 1, (dim, 1)) * np.sqrt(1.0 / dim)).astype(np.float32) for _ in range(n_layers)]
        self.b = [np.zeros((1, 1), dtype=np.float32) for _ in range(n_layers)]
        self.optW = [Adam(w.shape, lr=lr) for w in self.W]
        self.optb = [Adam(b.shape, lr=lr) for b in self.b]

    def forward(self, x0, train=True):
        xs = [x0]
        x = x0
        for i in range(self.n):
            proj = x @ self.W[i] + self.b[i]         # [B,1]
            x = x0 * proj + x
            xs.append(x)
        if train:
            self.xs = xs
        return x

    def backward(self, grad):
        x0 = self.xs[0]
        for i in reversed(range(self.n)):
            x_prev = self.xs[i]
            # x = x0*proj + x_prev, proj = x_prev@W + b
            g_proj = (grad * x0).sum(axis=1, keepdims=True)          # [B,1]
            gW = x_prev.T @ g_proj                                   # [dim,1]
            gb = g_proj.sum(axis=0, keepdims=True)                   # [1,1]
            grad_prev = grad + g_proj @ self.W[i].T                  # 直连 + 经 proj 回传
            self.optW[i].step(self.W[i], gW)
            self.optb[i].step(self.b[i], gb)
            grad = grad_prev
        return grad

    def n_params(self):
        return sum(w.size + b.size for w, b in zip(self.W, self.b))

    def arrays(self):
        return self.W + self.b


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def bce(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
