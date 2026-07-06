# -*- coding: utf-8 -*-
"""指标与工具：秩和 AUC、PR-AUC、NLL、KS、Lift/Capture、Bootstrap 置信区间、绘图样式。"""
import json
import time

import numpy as np


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_json(obj, path):
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=_default)


def auc_rank(y, s):
    """秩和 AUC，O(n log n)，并列值取平均秩（全向量化，适合百万级测试集）。"""
    y = np.asarray(y, dtype=np.int8)
    s = np.asarray(s)
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    new_grp = np.r_[True, s_sorted[1:] != s_sorted[:-1]]
    grp = np.cumsum(new_grp) - 1
    counts = np.bincount(grp)
    csum = np.cumsum(counts)
    avg_rank = (csum - counts + 1 + csum) / 2.0
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = avg_rank[grp]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def nll(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ks_stat(y, s):
    order = np.argsort(-s)
    y_sorted = np.asarray(y)[order]
    cum_pos = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    cum_neg = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
    return float(np.max(np.abs(cum_pos - cum_neg)))


def pr_auc(y, s):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s))


def lift_capture(y, s, fracs=(0.01, 0.05, 0.10, 0.20)):
    """Top-K% 打分样本中的转化捕获率与提升度。"""
    y = np.asarray(y)
    order = np.argsort(-s, kind="mergesort")
    total_pos = y.sum()
    base_rate = total_pos / len(y)
    out = {}
    for f in fracs:
        k = max(int(len(y) * f), 1)
        topk = y[order[:k]]
        out[f"Lift_{int(f*100)}pct"] = float((topk.sum() / k) / base_rate) if base_rate > 0 else float("nan")
        out[f"Capture_{int(f*100)}pct"] = float(topk.sum() / max(total_pos, 1))
    return out


def full_metrics(y, p):
    m = {"AUC": auc_rank(y, p), "PR_AUC": pr_auc(y, p), "NLL": nll(y, p), "KS": ks_stat(y, p)}
    m.update(lift_capture(y, p))
    return m


def bootstrap_gain(y, s_new, s_base, metric=auc_rank, n=200, seed=42):
    """测试集有放回重采样下 metric(s_new)-metric(s_base) 的均值与 95% CI。"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    gains = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        gains[i] = metric(y[idx], np.asarray(s_new)[idx]) - metric(y[idx], np.asarray(s_base)[idx])
    return {"mean": float(gains.mean()), "ci_low": float(np.quantile(gains, 0.025)),
            "ci_high": float(np.quantile(gains, 0.975)),
            "positive_ratio": float((gains > 0).mean()), "n": n}


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Hiragino Sans GB", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.bbox"] = "tight"
    return plt
