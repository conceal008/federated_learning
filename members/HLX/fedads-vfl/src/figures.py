# -*- coding: utf-8 -*-
"""图表：消融梯队 / 四模型对比 / ROC+PR / OOV 修复分析。全中文。"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)
from metrics import setup_mpl  # noqa: E402

OUT = os.path.abspath(os.path.join(HERE, "..", "..", "outputs", "fedads_vfl"))
RES = os.path.join(OUT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

CN = {"Label-only": "标签方单方", "Non-label-only": "非标签方单方",
      "Centralized": "集中式(全特征)", "VanillaVFL": "联邦VFL",
      "EnhancedVFL": "联邦VFL+未对齐增强"}
COLOR = {"Label-only": "#8c8c8c", "Non-label-only": "#bfbfbf",
         "Centralized": "#4c72b0", "VanillaVFL": "#dd8452", "EnhancedVFL": "#c44e52"}

plt = setup_mpl()
ab = json.load(open(os.path.join(RES, "ablation.json")))
dfm = pd.read_csv(os.path.join(RES, "metrics.csv"), index_col=0)

# ---- fig1 消融梯队（含未采纳杠杆，诚实展示） ----
rungs = ab["ablation"]
fig, ax = plt.subplots(figsize=(10, 4.6))
labels = [r["rung"] for r in rungs]
vals = [r["val_auc"] for r in rungs]
tests = [r["test_auc"] for r in rungs]
colors = []
for r in rungs:
    if "adopted" not in r:
        colors.append("#4c72b0")           # 基线
    elif r["adopted"]:
        colors.append("#55a868")           # 采纳
    else:
        colors.append("#c44e52")           # 未采纳(无效杠杆，诚实标注)
xs = np.arange(len(labels))
bars = ax.bar(xs, vals, color=colors, width=0.55)
ax.plot(xs, tests, "o--", color="#333", lw=1, ms=5, label="测试集 AUC")
for b, v, t in zip(bars, vals, tests):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.0004, f"{v:.4f}", ha="center", fontsize=8)
ax.set_xticks(xs, labels, fontsize=8.5)
lo = min(vals + tests) - 0.004
ax.set_ylim(lo, max(vals + tests) + 0.004)
ax.set_ylabel("验证集 AUC(柱) / 测试集 AUC(线)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc="#4c72b0", label="基线"), Patch(fc="#55a868", label="采纳的杠杆"),
                   Patch(fc="#c44e52", label="未采纳(无提升,如实记录)"),
                   plt.Line2D([], [], color="#333", ls="--", marker="o", label="测试集 AUC")],
          fontsize=8)
ax.set_title("性能消融梯队：每步只加一个杠杆，验证集提升才采纳")
fig.savefig(os.path.join(FIG, "fig1_ablation.png"))

# ---- fig2 四模型对比 ----
models = [m for m in ["Label-only", "Non-label-only", "Centralized", "VanillaVFL", "EnhancedVFL"] if m in dfm.index]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
ax = axes[0]
xs = np.arange(len(models))
aucs = [dfm.loc[m, "AUC"] for m in models]
b1 = ax.bar(xs, aucs, 0.5, color=[COLOR[m] for m in models])
for b in b1:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(xs, [CN[m] for m in models], fontsize=8, rotation=12)
ax.set_ylim(min(aucs) - 0.01, max(aucs) + 0.012)
ax.set_ylabel("测试集 AUC"); ax.set_title("四模型对比（联邦 vs 单方 vs 集中式）")
ax = axes[1]
bars = ax.bar([CN[m] for m in models], [dfm.loc[m, "PR_AUC"] for m in models], color=[COLOR[m] for m in models])
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.5f}", ha="center", va="bottom", fontsize=7.5)
ax.tick_params(axis="x", rotation=12, labelsize=8); ax.set_ylabel("PR-AUC"); ax.set_title("PR-AUC(0.6% 正样本场景)")
ax = axes[2]
bars = ax.bar([CN[m] for m in models], [dfm.loc[m, "NLL"] for m in models], color=[COLOR[m] for m in models])
ax.tick_params(axis="x", rotation=12, labelsize=8); ax.set_ylabel("NLL(越低越好)"); ax.set_title("校准(NLL)")
ax.set_ylim(min(dfm.loc[models, "NLL"]) * 0.98, max(dfm.loc[models, "NLL"]) * 1.01)
fig.savefig(os.path.join(FIG, "fig2_models.png"))

# ---- fig3 ROC / PR ----
from sklearn.metrics import precision_recall_curve, roc_curve
p = np.load(os.path.join(RES, "predictions.npz"))
y = p["y"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for m in models:
    s = p[m.replace("-", "_")]
    fpr, tpr, _ = roc_curve(y, s)
    st = max(len(fpr) // 2000, 1)
    axes[0].plot(fpr[::st], tpr[::st], label=f"{CN[m]} ({dfm.loc[m,'AUC']:.4f})", color=COLOR[m], lw=1.2)
    pr, rc, _ = precision_recall_curve(y, s)
    st = max(len(rc) // 2000, 1)
    axes[1].plot(rc[::st], pr[::st], label=CN[m], color=COLOR[m], lw=1.2)
axes[0].plot([0, 1], [0, 1], "--", color="#999", lw=0.8)
axes[0].set_xlabel("假正率"); axes[0].set_ylabel("真正率"); axes[0].set_title("ROC 曲线"); axes[0].legend(fontsize=7.5)
axes[1].set_xlabel("召回率"); axes[1].set_ylabel("精确率"); axes[1].set_yscale("log")
axes[1].set_title("PR 曲线"); axes[1].legend(fontsize=7.5)
fig.savefig(os.path.join(FIG, "fig3_roc_pr.png"))

# ---- fig4 OOV 修复分析 ----
aud5 = json.load(open(os.path.join(OUT, "data", "mc5_full", "audit.json")))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.bar(["不截断\n(min_count=1)", "频次截断\n(min_count=5)"], [2759634, aud5["vocab_sizes_A"][10]],
       color=["#d9d9d9", "#55a868"])
ax.set_ylabel("用户ID(l_u_fea_1)词表规模")
ax.set_title("频次截断:罕见用户归入 OOV 桶")
for i, v in enumerate([2759634, aud5["vocab_sizes_A"][10]]):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
ax = axes[1]
ax.bar(["不截断", "频次截断"], [65.1, aud5["test_oov_A"]["l_u_fea_1"] * 100], color=["#d9d9d9", "#55a868"])
ax.set_ylabel("测试集用户ID OOV 率(%)")
ax.set_title("OOV 率上升，但 OOV 嵌入在训练期被充分训练\n(不截断时 OOV 向量从未被训练=噪声)")
for i, v in enumerate([65.1, aud5["test_oov_A"]["l_u_fea_1"] * 100]):
    ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
fig.savefig(os.path.join(FIG, "fig4_oov.png"))

print("figures saved:", sorted(os.listdir(FIG)))
