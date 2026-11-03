# 组合策略统一对比

统一设置：Topk=50、`n_drop=10`、风险敞口 95%、买卖费率各 5bp、冲击成本 5bp；信号 T 日收盘后生成，T+1 日开盘交易。基准为同一 489 只有效 S&P 500 股票池的等权组合，首日按开盘到收盘计算。

## 2023 诊断集（非独立验证）

该区间用于原始模型配置和 Miner-lite 因子准入，因此不能作为所有策略的无偏验证集；这里只用于展示策略行为，最终结论以 2024–2026 留出集和前推实验为准。

| 策略 | 累计收益 | 年化收益 | Sharpe | 最大回撤 | 相对基准 |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost close-to-close 标签 | 20.60% | 20.78% | 1.23 | -13.77% | +0.55 个百分点 |
| XGBoost open-to-close 标签 | 15.33% | 15.46% | 1.11 | -12.98% | -4.72 个百分点 |
| Miner-lite 7 因子 composite | 24.82% | 25.05% | 1.28 | -14.79% | +4.77 个百分点 |
| 一日反转 | 22.23% | 22.43% | 1.19 | -14.33% | +2.18 个百分点 |
| 等权基准 | 20.05% | 20.23% | — | — | — |

## 2024-01-02–2026-02-26 最终留出集

| 策略 | 累计收益 | 年化收益 | Sharpe | 最大回撤 | 相对基准 |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost close-to-close 标签 | 28.55% | 12.44% | 0.72 | -18.62% | -11.69 个百分点 |
| XGBoost open-to-close 标签 | 35.12% | 15.08% | 1.04 | -15.92% | -5.13 个百分点 |
| Miner-lite 7 因子 composite | 26.53% | 11.61% | 0.61 | -24.91% | -13.72 个百分点 |
| 一日反转 | 40.36% | 17.14% | 0.90 | -22.09% | +0.11 个百分点 |
| 等权基准 | 40.25% | 17.10% | — | — | — |

## 判断

- Miner-lite 和 close-to-close XGBoost 在 2023 诊断集表现较好，但在最终留出集明显回落，存在市场状态依赖。
- open-to-close XGBoost 的风险调整收益最好，但仍未跑赢等权基准。
- 一日反转在最终留出集接近基准，是目前最稳妥的主基线。
- 当前不能据此宣称 Miner-lite 或 XGBoost 已产生稳定 alpha；下一步应使用时间滚动的因子准入和融合，而不是继续针对留出集优化参数。

数据目录：

- `data/qlib/portfolio_backtest_validation_xgb_close_drop10/`
- `data/qlib/portfolio_backtest_validation_xgb_open_close_drop10/`
- `data/qlib/portfolio_backtest_validation_miner_lite_drop10/`
- `data/qlib/portfolio_backtest_validation_reversal_drop10/`
- `data/qlib/portfolio_backtest_model_drop10/`
- `data/qlib/portfolio_backtest_open_close_drop10/`
- `data/qlib/portfolio_backtest_miner_lite_drop10/`
- `data/qlib/portfolio_backtest_reversal_drop10/`
