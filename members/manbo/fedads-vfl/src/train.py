# -*- coding: utf-8 -*-
"""训练器：多轮 + 验证集早停 + 最优参数快照。

设计要点：epoch 级早停防止在 0.6% 正样本上过拟合，模型选择用独立验证集。
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from metrics import auc_rank  # 复用已验证的秩和 AUC


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def snapshot(model):
    return [a.copy() for a in model.arrays()]


def restore(model, snap):
    for a, s in zip(model.arrays(), snap):
        a[...] = s


def fit_tower(model, X, y, tr, val, cfg, name):
    """单塔模型(Local/Non-label/Centralized)训练 + 早停。"""
    bs, rng = cfg["batch"], np.random.default_rng(cfg["seed"])
    best_auc, best_snap, bad = -1, None, 0
    curve = []
    for ep in range(cfg["epochs"]):
        order = rng.permutation(tr)
        t0 = time.time()
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            model.train_step(X[idx], y[idx])
        va = auc_rank(y[val], model.predict(X[val]))
        curve.append(va)
        log(f"{name} epoch {ep+1}/{cfg['epochs']} val_AUC={va:.5f} ({time.time()-t0:.0f}s)")
        if va > best_auc + 1e-5:
            best_auc, best_snap, bad = va, snapshot(model), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                log(f"{name} 早停于 epoch {ep+1}，最佳 val_AUC={best_auc:.5f}")
                break
    if best_snap:
        restore(model, best_snap)
    return {"best_val_auc": best_auc, "val_curve": curve, "epochs_ran": len(curve)}


def fit_vfl(model, XA, XB, y, tr, val, cfg, name, defense=None, attack_log=None):
    """两方 VFL(仅对齐样本)训练 + 早停。"""
    bs, rng = cfg["batch"], np.random.default_rng(cfg["seed"])
    best_auc, best_snap, bad = -1, None, 0
    curve = []
    for ep in range(cfg["epochs"]):
        order = rng.permutation(tr)
        t0 = time.time()
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            model.train_step(XA[idx], XB[idx], y[idx], defense=defense, attack_log=attack_log)
        va = auc_rank(y[val], model.predict(XA[val], XB[val]))
        curve.append(va)
        log(f"{name} epoch {ep+1}/{cfg['epochs']} val_AUC={va:.5f} ({time.time()-t0:.0f}s) comm={model.comm_bytes/1e6:.0f}MB")
        if va > best_auc + 1e-5:
            best_auc, best_snap, bad = va, snapshot(model), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                log(f"{name} 早停于 epoch {ep+1}，最佳 val_AUC={best_auc:.5f}")
                break
    if best_snap:
        restore(model, best_snap)
    return {"best_val_auc": best_auc, "val_curve": curve, "epochs_ran": len(curve)}


def enhance_vfl(model, XA_al, XB_al, y_al, tr, val, XA_un, y_un, h_tilde, cfg, name):
    """增强：对齐 federated 批 与 未对齐 local 批 交替训练(HeuristicVFL)。"""
    bs, rng = cfg["batch"], np.random.default_rng(cfg["seed"])
    n_al, n_un = len(tr), len(y_un)
    p_aligned = n_al / (n_al + n_un)
    order_al, order_un = rng.permutation(tr), rng.permutation(n_un)
    ia = iu = 0
    best_auc, best_snap, bad = -1, None, 0
    curve, iters_per_ep = [], (n_al + n_un) // bs
    for ep in range(cfg["epochs"]):
        t0 = time.time()
        for _ in range(iters_per_ep):
            if rng.random() <= p_aligned:
                idx = order_al[ia:ia + bs]; ia += bs
                if ia >= n_al:
                    order_al, ia = rng.permutation(tr), 0
                model.train_step(XA_al[idx], XB_al[idx], y_al[idx])
            else:
                idx = order_un[iu:iu + bs]; iu += bs
                if iu >= n_un:
                    order_un, iu = rng.permutation(n_un), 0
                model.train_step_unaligned(XA_un[idx], h_tilde[idx], y_un[idx])
        va = auc_rank(y_al[val], model.predict(XA_al[val], XB_al[val]))
        curve.append(va)
        log(f"{name} epoch {ep+1}/{cfg['epochs']} val_AUC={va:.5f} ({time.time()-t0:.0f}s)")
        if va > best_auc + 1e-5:
            best_auc, best_snap, bad = va, snapshot(model), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                log(f"{name} 早停于 epoch {ep+1}"); break
    if best_snap:
        restore(model, best_snap)
    return {"best_val_auc": best_auc, "val_curve": curve, "p_aligned": p_aligned}


def user_hN_table(model, XA_al, XB_al, uid_col, n_users, batch=200000):
    """从预训练 VFL 计算用户级 h_N 均值表(增强合成嵌入源)。"""
    H = model.H_N
    sums = np.zeros((n_users, H), dtype=np.float64)
    cnts = np.zeros(n_users, dtype=np.int64)
    for s in range(0, len(XA_al), batch):
        hN = model.forward_hN(XB_al[s:s + batch])
        uid = XA_al[s:s + batch, uid_col]
        np.add.at(sums, uid, hN)
        np.add.at(cnts, uid, 1)
    seen = cnts > 0
    lut = np.tile((sums[seen] / cnts[seen, None]).mean(axis=0).astype(np.float32), (n_users, 1))
    lut[seen] = (sums[seen] / cnts[seen, None]).astype(np.float32)
    return lut, float(seen.mean())
