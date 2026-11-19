# 12 个技术因子候选：探索性准入诊断

**不是新的样本外实验。** 在已经查看过先前实验的 2023 验证结果与 2024–2026 留出结果后，才固定本次候选集合。这里不重新执行旧留出区间的 Screener/Trader，也不使用留出标签挑选或放宽门槛；2022/2023 均只作研究诊断。候选于两次运行前写入 [`candidates_technical_exploratory_v1.json`](../../quant-method/alphacrafter/candidates_technical_exploratory_v1.json)，并保持相同内容。候选属于价格量技术特征，不包含新的 point-in-time 基本面/新闻数据，也未证明其为“优质”因子。

## 固定范围与窗口

- 趋势/动量：10、20、60 日收益、20 日均线距离；趋势加速：5 日与 20 日收益差。
- 超买超卖：RSI、布林带标准化偏离、随机指标。
- 风险：5 日波动率、14 日 ATR 相对值。
- 成交量/日内压力：1 日成交量变化、信号日开收盘收益。
- 原始输入：`machine-learn/data/processed/sp500_features_label_dates.csv.gz`，启用显式标签日期和本地报价日期并集日历。方向固定为各轮训练期的 1 日 Rank IC 符号；严格使用原有 Miner 1/5 日门槛。
- 2022 轮：截止 2022-12-30，训练 2016-01-04–2021-12-30，准入验证 2021-12-31–2022-12-30；产物在忽略 Git 的 `machine-learn/data/qlib/alphacrafter_explore12_2022/`。
- 2023 轮：截止 2023-12-29，训练 2016-01-04–2022-12-30，准入验证 2023-01-03–2023-12-29；产物在 `machine-learn/data/qlib/alphacrafter_explore12_2023/`。

两个目录各自独立运行，未继承另一轮 registry，也未运行 2024–2026 留出收益。每轮 `candidate_metrics.csv`、`temporal_audit.json` 和 `accepted_factors.json` 均已保存。若重跑须选择**新输出目录**；示例（2022 轮）：

```bash
python3 quant-method/alphacrafter/miner_lite.py \
  --input quant-method/machine-learn/data/processed/sp500_features_label_dates.csv.gz \
  --calendar quant-method/machine-learn/data/processed/sp500_features_label_dates.csv.gz.calendar.csv \
  --require-explicit-label-dates \
  --candidate-file quant-method/alphacrafter/candidates_technical_exploratory_v1.json \
  --as-of 2022-12-30 --train-start 2016-01-04 --train-end 2021-12-30 \
  --validation-start 2021-12-31 --validation-end 2022-12-30 \
  --backtest-start 2022-12-31 --backtest-end 2022-12-31 \
  --output-dir quant-method/machine-learn/data/qlib/alphacrafter_explore12_2022_rerun
```

## 结果

| 截止日期 | 候选数 | 通过数 | 训练 1 日最大方向 ICIR | 训练 5 日最大方向 ICIR | 准入 1 日最大方向 ICIR | 准入 5 日最大方向 ICIR |
|---|---:|---:|---:|---:|---:|---:|
| 2022-12-30 | 12 | **0** | 0.097 | 0.087 | 0.061 | 0.083 |
| 2023-12-29 | 12 | **0** | 0.064 | 0.067 | 0.111 | 0.085 |

以上每格均是该区间内所有候选的最大值，**不是某一个因子的连贯成绩**。1 日 ICIR 门槛为 > 0.20，5 日为 > 0.25；仅这一项已经让全部候选不能通过，IC、Rank IC、命中率仍需同时满足。例如 2023 准入期 `open_close_return` 的方向 ICIR 为 0.111、命中率约 57.0%，依然不合格。覆盖率和换手率不是本轮主要瓶颈。

**没有从本轮产生可交易策略或收益数字。** 表内 ICIR 是审计前 `ddof=1` 代码生成的旧运行值；指标修正与同候选收盘标签诊断见 [`METRIC_AUDIT.md`](METRIC_AUDIT.md)。不应为了得到非零收益而强制使用失败因子，或者事后调低门槛并声称原留出期仍是未用过的样本外数据。后续应优先检查阈值口径与数据时点/成分股真实性，提出更有机制依据的数据源和候选，另设独立实验；2024–2026 窗口若重用，只能报告为迭代研究。
