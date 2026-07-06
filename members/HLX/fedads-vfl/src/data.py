# -*- coding: utf-8 -*-
"""数据层：频次截断词表 + 全量未对齐 + 验证集切分。

核心设计（性能导向）：
1. 频次截断（min_count）：朴素做法用 min_count=1，OOV(0号)嵌入从不被训练；测试集 65% 用户为
   新用户 → 命中未训练的随机 OOV 向量 = 噪声。截断罕见 ID→OOV，使"未知用户"嵌入在训练期
   就被大量罕见 ID 训到，测试期新用户获得有意义的默认表示。这是最有原则的泛化改进。
2. 全量未对齐：从 CSV 抽取全部 1042 万未对齐样本。
3. 验证集：从 aligned 随机切 10% 作早停/模型选择，测试集仍为官方时间切分(最后一周)。

复用已切分的物理隔离 raw 目录（party_A/raw、party_B/raw），不重复解析 aligned/test CSV。
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                  "outputs", "fedads_vfl"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                  "outputs", "fedads_vfl"))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "FedAds"))
A_RAW = os.path.join(OUT, "party_A", "raw")
B_RAW = os.path.join(OUT, "party_B", "raw")
FIELDS_A = ["l_i_fea_1", "l_i_fea_2", "l_i_fea_3", "l_i_fea_4", "l_i_fea_5",
            "l_i_fea_6", "l_i_fea_7", "l_i_fea_8", "l_i_fea_9", "l_i_fea_10",
            "l_u_fea_1", "l_u_fea_2", "l_u_fea_3", "l_u_fea_4", "l_u_fea_5", "l_u_fea_6"]
FIELDS_B = ["f_u_fea_1", "f_u_fea_2", "f_uc_fea_1", "f_uc_fea_2", "f_c"]
SEED = 42


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def extract_full_unaligned():
    """从 CSV 流式抽取全部 1042 万未对齐样本的标签方特征 + 标签，存 raw。"""
    out_feat = os.path.join(OUT, "raw", "unaligned_full_feat.npy")
    if os.path.exists(out_feat):
        log("全量未对齐已存在，跳过抽取")
        return
    path = os.path.join(DATA_DIR, "sample_train_unaligned.csv")
    feats, labels = [], []
    n = 0
    for chunk in pd.read_csv(path, chunksize=1_000_000, dtype=str):
        cols = [chunk[c].str.slice(2, -2).astype(np.int64).to_numpy() for c in FIELDS_A]
        feats.append(np.column_stack(cols))
        labels.append(chunk["label"].str.slice(2, -2).astype(np.int8).to_numpy())
        n += len(chunk)
        log(f"全量未对齐抽取 {n:,}")
    np.save(out_feat, np.vstack(feats))
    np.save(os.path.join(OUT, "raw", "unaligned_full_label.npy"), np.concatenate(labels))
    log(f"全量未对齐抽取完成 {n:,} 行")


def build_vocab(train_cols, min_count):
    """逐字段建词表：出现次数 >= min_count 的值保留(排序)，其余归 OOV。返回排序后的值数组列表。"""
    vocabs = []
    for col in train_cols:
        vals, cnts = np.unique(col, return_counts=True)
        keep = np.sort(vals[cnts >= min_count])
        vocabs.append(keep)
    return vocabs


def encode(feat, vocabs):
    """int64 原始 ID → int32 词表位置+1；OOV=0。"""
    codes = np.empty(feat.shape, dtype=np.int32)
    oov = np.empty(feat.shape[1])
    for j, vocab in enumerate(vocabs):
        col = feat[:, j]
        if len(vocab) == 0:
            codes[:, j] = 0
            oov[j] = 1.0
            continue
        pos = np.clip(np.searchsorted(vocab, col), 0, len(vocab) - 1)
        hit = vocab[pos] == col
        codes[:, j] = np.where(hit, pos + 1, 0)
        oov[j] = float((~hit).mean())
    return codes, oov


def build_dataset(min_count=5, tag="mc5_full"):
    """构建一个完整数据集(编码/验证切分/OOV审计)到 OUT/data/<tag>/。"""
    out = os.path.join(OUT, "data", tag)
    os.makedirs(out, exist_ok=True)

    log("加载 A 方原始特征")
    xa_al = np.load(os.path.join(A_RAW, "aligned_feat.npy"))
    ya_al = np.load(os.path.join(A_RAW, "aligned_label.npy")).astype(np.int8)
    xa_te = np.load(os.path.join(A_RAW, "test_feat.npy"))
    ya_te = np.load(os.path.join(A_RAW, "test_label.npy")).astype(np.int8)
    xa_un = np.load(os.path.join(OUT, "raw", "unaligned_full_feat.npy"), mmap_mode="r")
    ya_un = np.load(os.path.join(OUT, "raw", "unaligned_full_label.npy")).astype(np.int8)
    xb_al = np.load(os.path.join(B_RAW, "aligned_feat.npy"))
    xb_te = np.load(os.path.join(B_RAW, "test_feat.npy"))

    log(f"A 方建词表(min_count={min_count}，训练期=aligned+全量unaligned)")
    vocab_a = []
    for j in range(len(FIELDS_A)):
        col = np.concatenate([xa_al[:, j], np.asarray(xa_un[:, j])])
        vals, cnts = np.unique(col, return_counts=True)
        vocab_a.append(np.sort(vals[cnts >= min_count]))
        del col
    log(f"B 方建词表(min_count={min_count}，训练期=aligned)")
    vocab_b = build_vocab([xb_al[:, j] for j in range(len(FIELDS_B))], min_count)

    log("编码各切片")
    ca_al, oov_al = encode(xa_al, vocab_a)
    ca_te, oov_te = encode(xa_te, vocab_a)
    ca_un, _ = encode(np.asarray(xa_un), vocab_a)
    cb_al, _ = encode(xb_al, vocab_b)
    cb_te, oov_b = encode(xb_te, vocab_b)

    # 验证集：aligned 随机 10%
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(ya_al))
    n_val = len(ya_al) // 10
    val_idx = np.sort(perm[:n_val])
    tr_idx = np.sort(perm[n_val:])

    np.savez_compressed(os.path.join(out, "A.npz"),
                        aligned=ca_al, test=ca_te, unaligned=ca_un,
                        y_aligned=ya_al, y_test=ya_te, y_unaligned=ya_un,
                        tr_idx=tr_idx, val_idx=val_idx)
    np.savez_compressed(os.path.join(out, "B.npz"), aligned=cb_al, test=cb_te)
    sizes_a = [len(v) + 1 for v in vocab_a]
    sizes_b = [len(v) + 1 for v in vocab_b]
    audit = {
        "min_count": min_count,
        "rows": {"aligned": int(len(ya_al)), "unaligned_full": int(len(ya_un)),
                 "test": int(len(ya_te)), "val": int(n_val), "train": int(len(tr_idx))},
        "vocab_sizes_A": sizes_a, "vocab_sizes_B": sizes_b,
        "test_oov_A": {FIELDS_A[j]: round(float(oov_te[j]), 4) for j in range(len(FIELDS_A))},
        "note_user_id": f"l_u_fea_1 词表 {sizes_a[10]}(截断前 2,759,634)；测试OOV {round(float(oov_te[10]),4)}",
    }
    with open(os.path.join(out, "audit.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    log(f"数据集 {tag} 构建完成：A 词表和 {sum(sizes_a):,}，B 词表和 {sum(sizes_b):,}")
    log(f"用户ID(l_u_fea_1)词表：{sizes_a[10]:,}(截断前 2,759,634)，测试OOV {oov_te[10]*100:.1f}%")
    return audit


if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "results", "figures"), exist_ok=True)
    mc = int(os.environ.get("MIN_COUNT", "5"))
    extract_full_unaligned()
    build_dataset(min_count=mc, tag=f"mc{mc}_full")
