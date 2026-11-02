# AlphaCrafter 对比实验：数据

论文没有随文发布可直接下载的数据集。根据论文 Appendix B.1，数据来源是：

- S&P 500 日线 OHLCV：Yahoo Finance
- CSI 300 日线 OHLCV：BaoStock
- PE/PS/PB/DYR、财务报表、新闻公告：Lixinger（需要另行获得权限/API）

当前先获取 **S&P 500**，因为它最容易复现 XGBoost baseline。下载命令：

```bash
python3 download_market_data.py --workers 8
```

数据位置：`data/raw/sp500/`，每个股票一个 CSV，字段为：

`date,ticker,open,high,low,close,adj_close,volume`

时间范围：2016-01-04 至 2026-02-27，严格覆盖论文的训练、验证和回测区间：

- 训练：2016-01-04–2022-12-30
- 验证：2023-01-03–2023-12-29
- 回测：2024-01-02–2026-02-27

## 重要可复现性约定

1. 成分股文件固定为 datasets/s-and-p-500-companies 在 2025-08-12 的 commit，而不是使用会变化的 master，避免在 2026-02-27 之后运行时引入成分股未来信息。
2. Yahoo Finance 当前无法返回其中 14 个股票代码的完整历史序列，因此当前有效股票数为 489/503；具体失败项见 `data/raw/sp500/manifest.json`。后续训练和 Agent 实验必须使用同一个有效股票集合。
3. `adj_close` 用于收益/标签计算；OHLC 用于技术特征。不要把未来日期数据用于训练。
4. 这一步只有价格量数据。论文提到的 Lixinger 基本面、财务报表和新闻还没有公开可复现的下载入口；若要严格复现“technical + fundamental”的 XGBoost，需要补齐该部分，并在数据可用日期上做 point-in-time 对齐，避免财报发布日期之后才可见的数据泄漏。

论文原始划分还报告了 CSI 300；本目录先固定 S&P 500 路线，待模型和回测流程稳定后再增加 BaoStock 数据。

## 技术指标特征

运行：

```bash
python3 build_features.py
```

输出：`data/processed/sp500_features.csv.gz`。每行对应股票在交易日 `t` 的信号，`target_1d` 是 `t` 到下一个交易日的复权收盘价收益。当前生成 35 个特征，包括：

- 动量：1/5/10/20/60 日收益率、价格相对 SMA/EMA
- 趋势：MACD、MACD signal、MACD histogram
- 超买超卖：RSI、布林带 z-score/宽度、随机指标
- 波动和风险：ATR、5/20/60 日波动率、滚动高低点距离
- 日内价格行为：振幅、开收盘收益、收盘价在日内区间的位置
- 成交量：成交量变化、相对 20 日均量、成交量波动率

指标只使用当日及之前的数据；标签使用下一交易日数据。前 120 个交易日作为指标预热期，不进入训练/验证/回测样本。

## Qlib + XGBoost 基线

当前基线暂时只使用价格量技术特征，不接入基本面和新闻数据。训练脚本使用 Qlib 的 `DatasetH` 和 `XGBModel`，以 36 个特征预测 `target_1d`（下一交易日复权收益），其中显式加入 `one_day_reversal = -return_1d`。训练时默认使用 Qlib 的 `CSRankNorm` 对特征和标签分别做逐日横截面 RankNorm，评估时恢复使用原始收益；同时将一日反转因子作为简单基线。可通过 `--feature-transform none` 关闭特征归一化：

已有 `tradingagents` conda 环境时直接运行：

```bash
conda run -n tradingagents python train_xgboost_qlib.py
```

当前冻结的默认配置来自验证集 Rank IC 实验：`reg:pseudohubererror`、25 棵树。完整的损失函数、树轮数、特征消融和滚动验证实验可运行：

```bash
conda run -n tradingagents python run_xgb_experiments.py
```

实验记录位于 `XGB_EXPERIMENTS.md`、`XGB_RANK_EXPERIMENT.md`、`XGB_BLEND_EXPERIMENT.md`、`XGB_5D_EXPERIMENT.md` 和 `MARKET_STATE_EXPERIMENT.md`；详细机器结果位于对应的 `data/qlib/` 实验目录。

如果环境尚未安装依赖，再执行：

```bash
conda run -n tradingagents python -m pip install -r requirements.txt
```

默认时间切分为：训练集 2016–2022，验证集 2023，回测集 2024-01-02–2026-02-27。当前默认使用较浅树、较高 `min_child_weight`、子采样、列采样和 L2 正则来抑制过拟合。预测结果、`feature_importance.csv`（weight/gain/cover/total_gain/total_cover）、每日 IC、Rank IC、Rank ICIR、Top/Bottom 十分位组合收益，以及简单因子基线对比写入 `data/qlib/xgboost_baseline/`。该目录及生成文件不纳入 Git。
