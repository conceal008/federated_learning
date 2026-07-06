# FedAds 纵向联邦学习实验

阿里妈妈 FedAds 数据集（SIGIR 2023 原生 VFL 广告转化 benchmark）上的纵向联邦学习实验记录。

## 目录结构

```
.
├── experiments/
│   ├── 01-linear-lr-leakage-audit/     ← 线性逻辑回归 + 哈希特征 + 泄漏审计（早期可行性验证）
│   └── 02-neural-vfl-psi-heuristic/    ← 神经网络拆分学习 + PSI 对齐 + HeuristicVFL 未对齐样本增强 + 隐私攻防
├── final-dcn-crossnet-vfl/              ← 最终定稿实验：DCN 特征交叉 + 频次截断修复 OOV（结论以此为准）
│   ├── report/                          ← 正式实验报告
│   └── results/                         ← 指标、图表、预测结果
└── src/                                  ← 最终版可复用联邦学习框架代码
```

各目录对应的核心技术：

| 目录 | 核心技术 |
|---|---|
| `01-linear-lr-leakage-audit` | 哈希稀疏特征 + mini-batch 线性逻辑回归拆分学习；数据泄漏审计 |
| `02-neural-vfl-psi-heuristic` | Embedding+MLP 神经网络拆分学习；HMAC-SHA256 PSI 样本对齐；HeuristicVFL 未对齐样本合成增强；MixPro/DP 标签泄露防御 |
| `final-dcn-crossnet-vfl` | 词表频次截断修复 OOV；DCN CrossNet 特征交叉；验证集早停 + 严格消融；梯度范数标签推断攻击 + MixPro 防御 |

## 结论速览

以 `final/report/` 里的报告为准：联邦 VFL 测试 AUC 0.6742，相对标签方单方（0.6532）增益 +0.0211（95% CI [0.0187, 0.0231]）。详见报告全文。

## 数据

本仓库不包含 FedAds 原始数据（CC BY-NC-SA 4.0，非商用许可）及训练用中间二进制产物（数十 GB，不适合放 Git）。如需复现，数据集见 [阿里天池](https://tianchi.aliyun.com/dataset/148347)，处理流程见 `src/` 下代码与各实验的 report。
