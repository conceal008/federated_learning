# -*- coding: utf-8 -*-
"""批大小是之前固定为 512 的未调超参。此处在验证集上正当扫 batch∈{128,256,512}
(仅用验证集选择，不看测试集)，选最佳后全量重训评测一次。诚实的超参调优。"""
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
from train import fit_vfl, log  # noqa: E402

OUT = os.path.abspath(os.path.join(HERE, "..", "..", "outputs", "fedads_vfl"))
RES = os.path.join(OUT, "results")
ARCH = {"emb_dim": 8, "lr": 1e-3, "seed": 42, "b_hidden": [128, 32], "a_hidden": [256, 128],
        "top_hidden": [], "cross_layers": 2, "dropout": 0.0, "epochs": 1, "patience": 99, "enhance": False}


def main():
    p = os.path.join(OUT, "data", "mc5_full")
    A = np.load(os.path.join(p, "A.npz")); B = np.load(os.path.join(p, "B.npz"))
    aud = json.load(open(os.path.join(p, "audit.json")))
    va, vb = aud["vocab_sizes_A"], aud["vocab_sizes_B"]
    XA, XB, y = A["aligned"], B["aligned"], A["y_aligned"].astype(np.float32)
    XA_te, XB_te, y_te = A["test"], B["test"], A["y_test"].astype(np.float32)
    tr, val = A["tr_idx"], A["val_idx"]

    sweep = {}
    for bs in [128, 256, 512]:
        cfg = dict(ARCH); cfg["batch"] = bs
        rng = np.random.default_rng(cfg["seed"])
        m = VFLModel(va, vb, cfg, rng)
        info = fit_vfl(m, XA, XB, y, tr, val, cfg, f"batch{bs}")
        sweep[bs] = info["best_val_auc"]
        log(f"batch={bs} val_AUC={info['best_val_auc']:.5f}")
    best_bs = max(sweep, key=sweep.get)
    log(f"验证集最佳 batch={best_bs} (val={sweep[best_bs]:.5f})")

    # 全量重训四模型 @ best batch
    cfg = dict(ARCH); cfg["batch"] = best_bs
    preds = {}
    def tower_full(vocab, X):
        rng = np.random.default_rng(cfg["seed"]); m = Tower(vocab, cfg, rng)
        order = rng.permutation(len(y))
        for s in range(0, len(order), best_bs):
            m.train_step(X[order[s:s+best_bs]], y[order[s:s+best_bs]])
        return m
    preds["Label-only"] = tower_full(va, XA).predict(XA_te)
    preds["Non-label-only"] = tower_full(vb, XB).predict(XB_te)
    Xc, Xc_te = np.concatenate([XA, XB], 1), np.concatenate([XA_te, XB_te], 1)
    preds["Centralized"] = tower_full(va + vb, Xc).predict(Xc_te)
    del Xc, Xc_te
    rng = np.random.default_rng(cfg["seed"]); mv = VFLModel(va, vb, cfg, rng)
    order = rng.permutation(len(y))
    for s in range(0, len(order), best_bs):
        mv.train_step(XA[order[s:s+best_bs]], XB[order[s:s+best_bs]], y[order[s:s+best_bs]])
    preds["VanillaVFL"] = mv.predict(XA_te, XB_te)

    metrics = {k: full_metrics(y_te, pr) for k, pr in preds.items()}
    pd.DataFrame(metrics).T.to_csv(os.path.join(RES, "metrics.csv"))
    np.savez_compressed(os.path.join(RES, "predictions.npz"),
                        y=y_te.astype(np.int8), **{k.replace("-", "_"): v for k, v in preds.items()})
    boot = bootstrap_gain(y_te, preds["VanillaVFL"], preds["Label-only"])
    ab = json.load(open(os.path.join(RES, "ablation.json")))
    ab["batch_sweep_val"] = {str(k): v for k, v in sweep.items()}
    ab["final_full_retrain"] = {"config": f"mc5+特征交叉+1轮+batch{best_bs}(验证集选)，全量对齐",
                                "best_batch": best_bs, "metrics": metrics, "vfl_vs_label_boot": boot,
                                }
    save_json(ab, os.path.join(RES, "ablation.json"))
    for k, v in metrics.items():
        log(f"{k}: AUC={v['AUC']:.5f} PR={v['PR_AUC']:.5f} NLL={v['NLL']:.5f}")
    log(f"联邦增益 Δ{boot['mean']:+.4f} CI[{boot['ci_low']:+.4f},{boot['ci_high']:+.4f}]")
    log("batch_sweep 完成")


if __name__ == "__main__":
    main()
