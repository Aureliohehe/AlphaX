# XGBoost 基线实验记录

- 实验时间（UTC）：2026-11-01T13:24:06.753805+00:00
- 输入文件：`data/processed/sp500_features.csv.gz`
- 样本数：1,158,676
- 特征数：36
- 标签：`target_1d`，训练时使用 Qlib `CSRankNorm`
- 特征：使用 Qlib `CSRankNorm` 逐日横截面归一化
- 评价：验证集只使用每日 Rank IC 选择树轮数、损失函数和特征方案；回测集仅用于最终报告
- 基础超参：`max_depth=4, eta=0.03, min_child_weight=20, subsample=0.7, colsample_bytree=0.75, gamma=0.1, reg_lambda=10`
- 树轮数候选：`[25, 50, 75, 100, 150, 200, 300, 400]`

## 1. 损失函数比较

候选损失函数为 `reg:squarederror`、`reg:pseudohubererror` 和 `reg:absoluteerror`。每个损失函数均在验证集 Rank IC 最高的树轮数处评估。

验证集选择出的损失函数：`reg:pseudohubererror`。

| 损失函数 | 最佳轮数 | 验证 Rank IC | 回测 Rank IC | 回测 Rank ICIR |
| --- | ---: | ---: | ---: | ---: |
| `reg:squarederror` | 25 | 0.013379 | 0.019641 | 0.130036 |
| `reg:pseudohubererror` | 25 | **0.013848** | 0.018597 | 0.124559 |
| `reg:absoluteerror` | 200 | 0.013783 | **0.019711** | **0.134445** |

> 按验证集选择 `reg:pseudohubererror`；虽然 `reg:absoluteerror` 的回测结果略高，但不能使用回测集反向选择模型。

## 2. `return_1d` 与 `one_day_reversal` 消融

- `full`：同时保留两个特征
- `without_return_1d`：移除 `return_1d`
- `without_one_day_reversal`：移除 `one_day_reversal`

| 特征方案 | 最佳轮数 | 验证 Rank IC | 回测 Rank IC |
| --- | ---: | ---: | ---: |
| `full` | 25 | **0.013848** | 0.018597 |
| `without_return_1d` | 75 | 0.012719 | 0.019326 |
| `without_one_day_reversal` | 50 | 0.012570 | 0.019372 |

两个特征虽然信息完全相关，但保留两者的方案在验证集表现最好；最终模型保留完整特征池。

## 3. 滚动验证

滚动验证的训练集只使用验证年份之前的数据，每个窗口独立按 Rank IC 选择树轮数。使用在标准 2023 验证集上选出的损失函数 `reg:pseudohubererror`。

| 验证年份 | 最佳轮数 | Rank IC | Rank ICIR | Top/Bottom 十分位收益差 |
| --- | ---: | ---: | ---: | ---: |
| 2021 | 100 | 0.020085 | 0.100139 | 0.000727 |
| 2022 | 400 | 0.017832 | 0.098175 | 0.000996 |
| 2023 | 25 | 0.013848 | 0.092830 | -0.000190 |

## 4. 当前结论

1. 验证集 Rank IC 选择出的默认配置为 `reg:pseudohubererror + 25` 棵树。
2. 损失函数差异不大，`reg:pseudohubererror` 在验证集略优；`reg:absoluteerror` 的回测结果不能用于模型选择。
3. `return_1d` 和 `one_day_reversal` 的完整特征方案验证集表现最好。
4. 2021–2023 滚动验证的 Rank IC 均为正，但 2023 年 Top/Bottom 组合收益差为负，说明信号稳定性仍然有限。
5. 后续应继续进行滚动窗口、不同市场状态和组合构建层面的稳健性检验。

## 5. Pairwise 排序目标实验

进一步使用 XGBoost 原生 `rank:pairwise`，并按交易日设置 group。标签转换为每日 10 档整数相关性等级，特征仍使用逐日 `CSRankNorm`。

| 指标 | 验证集 | 回测集 |
| --- | ---: | ---: |
| Rank IC | **0.016456** | 0.011228 |
| Rank ICIR | 0.112973 | 0.073710 |
| Top/Bottom 十分位收益差 | 0.000495 | 0.000749 |

验证集最佳树轮数为 400。排序目标在验证集超过了回归模型和一日反转因子，但回测集明显回落，说明仍然存在较强的时间段过拟合，不能直接替换当前默认模型。

详细记录见 `XGB_RANK_EXPERIMENT.md`，机器结果位于 `data/qlib/xgb_rank_experiment/`。

## 6. XGBoost 与一日反转因子融合

对模型分数和一日反转因子分别做逐日横截面 RankNorm，再用验证集选择线性融合权重。验证集选择：

```text
score = 0.25 * model + 0.75 * one_day_reversal
```

| 方法 | 验证 Rank IC | 回测 Rank IC |
| --- | ---: | ---: |
| 一日反转因子 | 0.016245 | 0.015739 |
| 融合模型 | **0.016557** | 0.017199 |
| 单独 XGBoost | 0.013848 | 0.018597 |

融合模型在验证集和简单反转因子相比有小幅提升，回测也超过简单反转因子，但低于单独 XGBoost；该权重严格按验证集选择，不能据此宣称稳定提升。

详细记录见 `XGB_BLEND_EXPERIMENT.md`，机器结果位于 `data/qlib/xgb_blend_experiment/`。

## 7. 五交易日预测目标实验

将标签改为 `target_5d = close[t+5] / close[t] - 1`，使用相同的技术特征和 `reg:pseudohubererror`。

| 方法 | 验证 Rank IC | 回测 Rank IC |
| --- | ---: | ---: |
| XGBoost 5 日目标 | **0.015432** | 0.012483 |
| `-return_5d` 基线 | -0.003527 | 0.007646 |

五日目标在验证集优于当前一日模型，但回测集明显回落，说明延长预测窗口降低了部分日内噪声，却没有解决跨市场状态的泛化问题。该实验暂不替换一日目标模型。

详细记录见 `XGB_5D_EXPERIMENT.md`，机器结果位于 `data/qlib/xgb_5d_experiment/`。

## 8. 市场状态特征实验

基于当前价格量数据增加逐日市场状态特征：平均收益、中位数收益、上涨比例、横截面离散度、平均波动率和平均成交量比率。状态特征保持原始逐日统计值，没有做 `CSRankNorm`，否则同一交易日的常数特征会被归一化为无信息值。

| 方案 | 验证 Rank IC | 回测 Rank IC |
| --- | ---: | ---: |
| 仅技术特征 | 0.013848 | 0.018597 |
| 加入市场状态 | 0.011795 | **0.020641** |

市场状态特征提高了回测 Rank IC，但验证集下降，因此暂不纳入默认模型，避免根据回测集选择特征方案。详细记录见 `MARKET_STATE_EXPERIMENT.md`。

## 9. 复现实验

```bash
conda run -n tradingagents python run_xgb_experiments.py
conda run -n tradingagents python run_xgb_rank_experiment.py
```

详细机器可读结果位于：

- `data/qlib/xgb_experiments/summary.csv`
- `data/qlib/xgb_experiments/round_search.csv`
- `data/qlib/xgb_experiments/results.json`
