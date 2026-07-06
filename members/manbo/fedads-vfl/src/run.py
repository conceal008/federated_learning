# -*- coding: utf-8 -*-
"""编排：性能导向的贪心消融梯队 + 最佳配置四模型对比 + 标签泄露安全验证。

梯队(每步只加一个杠杆，验证集 AUC 有提升才保留，否则诚实记为无效并不携带)：
  A0 基线(mc1,1轮,拼接) → +频次截断(mc5) → +多轮早停 → +特征交叉 → +全量未对齐增强
最佳配置下训练 Local/Non-label/Centralized/VFL 做公平对比，测试集全量评估 + Bootstrap CI。
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)
from metrics import auc_rank, full_metrics, bootstrap_gain, save_json  # noqa: E402
from vfl import Tower, VFLModel, MixPro  # noqa: E402
from train import fit_tower, fit_vfl, enhance_vfl, user_hN_table, log  # noqa: E402

OUT = os.path.abspath(os.path.join(HERE, "..", "..", "outputs", "fedads_vfl"))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "outputs", "fedads_vfl"))
RES = os.path.join(OUT, "results")
os.makedirs(RES, exist_ok=True)
UID_COL = 10  # l_u_fea_1

BASE = {"emb_dim": 8, "lr": 1e-3, "batch": 512, "seed": 42,
        "b_hidden": [128, 32], "a_hidden": [256, 128], "top_hidden": [],
        "cross_layers": 0, "dropout": 0.0, "epochs": 1, "patience": 99, "enhance": False}


def load_mc1():
    """基础配置：mc=1 编码 + 验证切分(对齐行序一致)。"""
    pa = os.path.join(OUT, "party_A", "processed")
    pb = os.path.join(OUT, "party_B", "processed")
    ra = os.path.join(OUT, "party_A", "raw")
    d = {"XA_al": np.load(os.path.join(pa, "aligned_codes.npy")),
         "XB_al": np.load(os.path.join(pb, "aligned_codes.npy")),
         "XA_te": np.load(os.path.join(pa, "test_codes.npy")),
         "XB_te": np.load(os.path.join(pb, "test_codes.npy")),
         "y_al": np.load(os.path.join(ra, "aligned_label.npy")).astype(np.float32),
         "y_te": np.load(os.path.join(ra, "test_label.npy")).astype(np.float32)}
    va = json.load(open(os.path.join(OUT, "party_A", "audit", "vocab_audit.json")))["vocab_sizes"]
    vb = json.load(open(os.path.join(OUT, "party_B", "audit", "vocab_audit.json")))["vocab_sizes"]
    meta = np.load(os.path.join(OUT, "data", "mc5_full", "A.npz"))
    d["tr"], d["val"] = meta["tr_idx"], meta["val_idx"]
    d["vocab_a"], d["vocab_b"] = va, vb
    return d


def load_mc5(with_unaligned=False):
    p = os.path.join(OUT, "data", "mc5_full")
    A = np.load(os.path.join(p, "A.npz"))
    B = np.load(os.path.join(p, "B.npz"))
    aud = json.load(open(os.path.join(p, "audit.json")))
    d = {"XA_al": A["aligned"], "XB_al": B["aligned"], "XA_te": A["test"], "XB_te": B["test"],
         "y_al": A["y_aligned"].astype(np.float32), "y_te": A["y_test"].astype(np.float32),
         "tr": A["tr_idx"], "val": A["val_idx"],
         "vocab_a": aud["vocab_sizes_A"], "vocab_b": aud["vocab_sizes_B"]}
    if with_unaligned:
        d["XA_un"] = A["unaligned"]; d["y_un"] = A["y_unaligned"].astype(np.float32)
    return d


def run_vfl(cfg, data, name, enhance=False):
    rng = np.random.default_rng(cfg["seed"])
    m = VFLModel(data["vocab_a"], data["vocab_b"], cfg, rng)
    if enhance:
        # 先在对齐上预训练，再算用户级 h_N 表，再交替训练
        fit_vfl(m, data["XA_al"], data["XB_al"], data["y_al"], data["tr"], data["val"], cfg, name + "-pre")
        lut, cov = user_hN_table(m, data["XA_al"], data["XB_al"], UID_COL, cfg["n_users"])
        h_tilde = lut[data["XA_un"][:, UID_COL]]
        info = enhance_vfl(m, data["XA_al"], data["XB_al"], data["y_al"], data["tr"], data["val"],
                           data["XA_un"], data["y_un"], h_tilde, cfg, name)
        info["synth_coverage"] = cov
    else:
        info = fit_vfl(m, data["XA_al"], data["XB_al"], data["y_al"], data["tr"], data["val"], cfg, name)
    te = m.predict(data["XA_te"], data["XB_te"])
    info["test_auc"] = auc_rank(data["y_te"], te)
    return m, te, info


def main():
    results = {"ablation": [], "started": time.strftime("%Y-%m-%d %H:%M")}

    # ---------- A0 基线 (mc1) ----------
    log("===== A0 基线：mc=1, 1轮, 拼接 =====")
    d1 = load_mc1()
    cfg = dict(BASE)
    _, _, info = run_vfl(cfg, d1, "A0")
    results["ablation"].append({"rung": "A0 基线(mc1,1轮)", "cfg": "mc1,1ep,concat",
                                "val_auc": info["best_val_auc"], "test_auc": info["test_auc"]})
    save_json(results, os.path.join(RES, "ablation.json"))
    best_val, best_desc = info["best_val_auc"], "A0"
    del d1

    # ---------- 加载 mc5 ----------
    d5 = load_mc5(with_unaligned=False)

    # ---------- 贪心加杠杆 ----------
    cur = dict(BASE)                       # 当前最佳配置(从基线口径出发)
    levers = [
        ("+频次截断(mc5)", {}),                                   # 仅换数据到 mc5
        ("+多轮早停", {"epochs": 8, "patience": 2, "dropout": 0.1}),
        ("+特征交叉", {"cross_layers": 2}),
    ]
    for label, change in levers:
        trial = dict(cur); trial.update(change)
        log(f"===== 试验 {label}：{change or 'mc5数据'} =====")
        _, _, info = run_vfl(trial, d5, label)
        adopt = info["best_val_auc"] > best_val + 1e-4
        results["ablation"].append({"rung": label, "cfg": {k: trial[k] for k in ["epochs", "cross_layers", "dropout"]},
                                    "val_auc": info["best_val_auc"], "test_auc": info["test_auc"],
                                    "delta_val": info["best_val_auc"] - best_val, "adopted": bool(adopt)})
        save_json(results, os.path.join(RES, "ablation.json"))
        log(f"{label}: val {info['best_val_auc']:.5f} (Δ{info['best_val_auc']-best_val:+.5f}) {'采纳' if adopt else '不采纳(诚实记录)'}")
        if adopt:
            cur, best_val, best_desc = trial, info["best_val_auc"], label

    # ---------- A4 全量未对齐增强 ----------
    log("===== +全量未对齐增强(HeuristicVFL) =====")
    d5u = load_mc5(with_unaligned=True)
    cfg_enh = dict(cur); cfg_enh.update({"enhance": True, "n_users": d5["vocab_a"][UID_COL]})
    if cfg_enh["epochs"] < 3:
        cfg_enh["epochs"] = 4  # 增强需足够轮次
    m_best, te_best, info = run_vfl(cfg_enh, d5u, "增强")
    adopt = info["best_val_auc"] > best_val + 1e-4
    results["ablation"].append({"rung": "+全量未对齐增强", "val_auc": info["best_val_auc"],
                                "test_auc": info["test_auc"], "delta_val": info["best_val_auc"] - best_val,
                                "adopted": bool(adopt), "synth_coverage": info.get("synth_coverage")})
    save_json(results, os.path.join(RES, "ablation.json"))
    if adopt:
        best_val, best_desc = info["best_val_auc"], "增强"
        best_cfg = cfg_enh
    else:
        best_cfg = dict(cur)
    results["best_desc"] = best_desc
    results["best_cfg"] = {k: best_cfg[k] for k in ["emb_dim", "batch", "epochs", "cross_layers", "dropout", "enhance"]}

    # ---------- 最佳配置下四模型公平对比 ----------
    log("===== 最佳配置四模型对比 =====")
    fair = dict(cur); fair["enhance"] = False   # 四模型用非增强口径公平比(VFL 用增强单列)
    preds = {}
    # Local (标签方单方)
    rng = np.random.default_rng(fair["seed"])
    m = Tower(d5["vocab_a"], fair, rng)
    fit_tower(m, d5["XA_al"], d5["y_al"], d5["tr"], d5["val"], fair, "Label-only")
    preds["Label-only"] = m.predict(d5["XA_te"])
    # Non-label-only
    rng = np.random.default_rng(fair["seed"])
    m = Tower(d5["vocab_b"], fair, rng)
    fit_tower(m, d5["XB_al"], d5["y_al"], d5["tr"], d5["val"], fair, "Non-label-only")
    preds["Non-label-only"] = m.predict(d5["XB_te"])
    # Centralized
    rng = np.random.default_rng(fair["seed"])
    XA_al, XB_al = d5["XA_al"], d5["XB_al"]
    offset = np.array(d5["vocab_a"], dtype=np.int64)  # 拼接词表
    Xc_al = np.concatenate([XA_al, XB_al], axis=1)
    Xc_te = np.concatenate([d5["XA_te"], d5["XB_te"]], axis=1)
    m = Tower(d5["vocab_a"] + d5["vocab_b"], fair, rng)
    fit_tower(m, Xc_al, d5["y_al"], d5["tr"], d5["val"], fair, "Centralized")
    preds["Centralized"] = m.predict(Xc_te)
    del Xc_al, Xc_te
    # VanillaVFL (非增强最佳)
    _, te_v, _ = run_vfl(fair, d5, "VanillaVFL")
    preds["VanillaVFL"] = te_v
    # 增强 VFL
    preds["EnhancedVFL"] = te_best

    metrics = {k: full_metrics(d5["y_te"], p) for k, p in preds.items()}
    import pandas as pd
    pd.DataFrame(metrics).T.to_csv(os.path.join(RES, "metrics.csv"))
    np.savez_compressed(os.path.join(RES, "predictions.npz"),
                        y=d5["y_te"].astype(np.int8), **{k.replace("-", "_"): v for k, v in preds.items()})
    results["final_metrics"] = metrics
    results["bootstrap"] = {
        "VanillaVFL_vs_Label-only": bootstrap_gain(d5["y_te"], preds["VanillaVFL"], preds["Label-only"]),
        "EnhancedVFL_vs_VanillaVFL": bootstrap_gain(d5["y_te"], preds["EnhancedVFL"], preds["VanillaVFL"]),
        "EnhancedVFL_vs_Label-only": bootstrap_gain(d5["y_te"], preds["EnhancedVFL"], preds["Label-only"]),
    }
    save_json(results, os.path.join(RES, "ablation.json"))
    for k, v in metrics.items():
        log(f"{k}: AUC={v['AUC']:.5f} PR={v['PR_AUC']:.5f} NLL={v['NLL']:.5f}")
    log("run4 完成")


if __name__ == "__main__":
    main()
