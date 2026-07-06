# -*- coding: utf-8 -*-
"""安全验证：切割层梯度的标签泄露 + MixPro 防护（诚实口径）。

只保留在本数据上真实成立的攻击（梯度范数攻击）。嵌入聚类攻击在本数据无效
(LeakAUC≈0.43，B 方特征过粗)，不再作为证据保留。
输出：security.json + fig5_security.png
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)
from metrics import auc_rank, save_json, setup_mpl  # noqa: E402
from vfl import VFLModel, MixPro  # noqa: E402
from train import fit_vfl, log  # noqa: E402

OUT = os.path.abspath(os.path.join(HERE, "..", "..", "outputs", "fedads_vfl"))
RES = os.path.join(OUT, "results")

CFG = {"emb_dim": 8, "lr": 1e-3, "batch": 512, "seed": 42,
       "b_hidden": [128, 32], "a_hidden": [256, 128], "top_hidden": [],
       "cross_layers": 0, "dropout": 0.0, "epochs": 1, "patience": 99, "enhance": False}


def main():
    p = os.path.join(OUT, "data", "mc5_full")
    A = np.load(os.path.join(p, "A.npz"))
    B = np.load(os.path.join(p, "B.npz"))
    aud = json.load(open(os.path.join(p, "audit.json")))
    XA, XB = A["aligned"], B["aligned"]
    y = A["y_aligned"].astype(np.float32)
    XA_te, XB_te = A["test"], B["test"]
    y_te = A["y_test"].astype(np.float32)
    tr, val = A["tr_idx"], A["val_idx"]

    out = {}
    for name, dfn in [("无防护", None), ("MixPro防护", MixPro(seed=42))]:
        rng = np.random.default_rng(CFG["seed"])
        m = VFLModel(aud["vocab_sizes_A"], aud["vocab_sizes_B"], CFG, rng)
        alog = {"norms": [], "labels": []}
        fit_vfl(m, XA, XB, y, tr, val, CFG, name, defense=dfn, attack_log=alog)
        norms = np.concatenate(alog["norms"])
        labels = np.concatenate(alog["labels"])
        leak = float(auc_rank(labels, norms))
        te_auc = float(auc_rank(y_te, m.predict(XA_te, XB_te)))
        out[name] = {"leak_auc_gradnorm": leak, "test_auc": te_auc}
        log(f"{name}: LeakAUC={leak:.4f} 测试AUC={te_auc:.5f}")

    out["MixPro防护"]["delta_leak"] = (out["MixPro防护"]["leak_auc_gradnorm"]
                                     - out["无防护"]["leak_auc_gradnorm"]) / out["无防护"]["leak_auc_gradnorm"]
    out["MixPro防护"]["auc_cost"] = out["MixPro防护"]["test_auc"] - out["无防护"]["test_auc"]
    out["说明"] = "梯度范数攻击=诚实但好奇的B方用切割层梯度范数推断标签；嵌入聚类攻击在本数据无效不再保留"
    save_json(out, os.path.join(RES, "security.json"))

    plt = setup_mpl()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    names = ["无防护", "MixPro防护"]
    leaks = [out[n]["leak_auc_gradnorm"] for n in names]
    bars = ax.bar(names, leaks, color=["#c44e52", "#55a868"], width=0.5)
    for b, v in zip(bars, leaks):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
    ax.axhline(0.5, ls="--", color="#666", lw=1)
    ax.text(1.3, 0.51, "随机猜测", fontsize=8, color="#666")
    ax.set_ylabel("标签泄露 LeakAUC(越低越安全)")
    ax.set_ylim(0.4, 1.08)
    ax.set_title("切割层梯度的标签泄露与防护")
    ax = axes[1]
    aucs = [out[n]["test_auc"] for n in names]
    bars = ax.bar(names, aucs, color=["#c44e52", "#55a868"], width=0.5)
    for b, v in zip(bars, aucs):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom")
    ax.set_ylabel("测试集 AUC")
    ax.set_ylim(min(aucs) - 0.01, max(aucs) + 0.01)
    ax.set_title(f"防护的效用代价：ΔAUC={out['MixPro防护']['auc_cost']:+.4f}")
    fig.savefig(os.path.join(RES, "figures", "fig5_security.png"))
    log("security4 完成")


if __name__ == "__main__":
    main()
