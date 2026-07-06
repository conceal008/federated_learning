# -*- coding: utf-8 -*-
"""最终重训：验证集已选定最佳配置(mc5+特征交叉+1轮，不用多轮/增强)，
在全量对齐数据(不留验证集)上重训四模型，得到公平的最终数字。
覆盖 metrics.csv / predictions.npz，并写 final_full 到 ablation.json。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)
from metrics import auc_rank, full_metrics, bootstrap_gain, save_json  # noqa: E402
from vfl import Tower, VFLModel  # noqa: E402
from train import log  # noqa: E402

OUT = os.path.abspath(os.path.join(HERE, "..", "..", "outputs", "fedads_vfl"))
RES = os.path.join(OUT, "results")
BEST = {"emb_dim": 8, "lr": 1e-3, "batch": 512, "seed": 42,
        "b_hidden": [128, 32], "a_hidden": [256, 128], "top_hidden": [],
        "cross_layers": 2, "dropout": 0.0, "epochs": 1, "patience": 99, "enhance": False}


def train_tower_full(vocab, X, y, cfg, name):
    rng = np.random.default_rng(cfg["seed"])
    m = Tower(vocab, cfg, rng)
    order = rng.permutation(len(y))
    for s in range(0, len(order), cfg["batch"]):
        idx = order[s:s + cfg["batch"]]
        m.train_step(X[idx], y[idx])
    log(f"{name} 全量重训完成")
    return m


def main():
    p = os.path.join(OUT, "data", "mc5_full")
    A = np.load(os.path.join(p, "A.npz")); B = np.load(os.path.join(p, "B.npz"))
    aud = json.load(open(os.path.join(p, "audit.json")))
    va, vb = aud["vocab_sizes_A"], aud["vocab_sizes_B"]
    XA, XB = A["aligned"], B["aligned"]
    y = A["y_aligned"].astype(np.float32)
    XA_te, XB_te, y_te = A["test"], B["test"], A["y_test"].astype(np.float32)

    preds = {}
    m = train_tower_full(va, XA, y, BEST, "Label-only"); preds["Label-only"] = m.predict(XA_te)
    m = train_tower_full(vb, XB, y, BEST, "Non-label-only"); preds["Non-label-only"] = m.predict(XB_te)
    Xc, Xc_te = np.concatenate([XA, XB], 1), np.concatenate([XA_te, XB_te], 1)
    m = train_tower_full(va + vb, Xc, y, BEST, "Centralized"); preds["Centralized"] = m.predict(Xc_te)
    del Xc, Xc_te
    rng = np.random.default_rng(BEST["seed"])
    mv = VFLModel(va, vb, BEST, rng)
    order = rng.permutation(len(y))
    for s in range(0, len(order), BEST["batch"]):
        idx = order[s:s + BEST["batch"]]
        mv.train_step(XA[idx], XB[idx], y[idx])
    preds["VanillaVFL"] = mv.predict(XA_te, XB_te)
    log("VanillaVFL 全量重训完成")

    metrics = {k: full_metrics(y_te, p_) for k, p_ in preds.items()}
    pd.DataFrame(metrics).T.to_csv(os.path.join(RES, "metrics.csv"))
    np.savez_compressed(os.path.join(RES, "predictions.npz"),
                        y=y_te.astype(np.int8), **{k.replace("-", "_"): v for k, v in preds.items()})
    boot = bootstrap_gain(y_te, preds["VanillaVFL"], preds["Label-only"])
    ab = json.load(open(os.path.join(RES, "ablation.json")))
    ab["final_full_retrain"] = {"config": "mc5+特征交叉+1轮，全量对齐",
                                "metrics": metrics,
                                "vfl_vs_label_boot": boot,
                                }
    save_json(ab, os.path.join(RES, "ablation.json"))
    for k, v in metrics.items():
        log(f"{k}: AUC={v['AUC']:.5f} PR={v['PR_AUC']:.5f} NLL={v['NLL']:.5f}")
    log(f"联邦增益 Δ{boot['mean']:+.4f} CI[{boot['ci_low']:+.4f},{boot['ci_high']:+.4f}]")
    log("final4 完成")


if __name__ == "__main__":
    main()
