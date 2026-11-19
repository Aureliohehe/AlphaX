# Qlib 行情 + 三组固定模型／组合回测（回溯实验）

借鉴 [QuantaAlpha](https://github.com/QuantaAlpha/QuantaAlpha/blob/main/README.md) 的 **官方因子／自定义因子／合并因子** 对照，而不是复制论文或该仓库的模型/结果。入口 [`run_qlib_three_way.py`](../../quant-method/machine-learn/run_qlib_three_way.py) **直接从已有本地 Qlib `sp500_provider` 读取行情与 Alpha158 表达式**，固定 LightGBM 参数训练预测，再调用本项目 Qlib `TopkDropoutStrategy` 执行组合回测。不再使用原特征 CSV 充当行情；不下载外部数据，不连接券商。

**结果只能当迭代诊断，不是新的样本外 Alpha。** Yahoo 构建的 Qlib provider 用当前 S&P 成员快照回填历史（缺退出成分）；两个自定义表达式对应的是事后提出的 `roc30_ts_slope_30` / `corr20_ts_mean_20`，在历史回测当时不可用；2022–2023 收益此前也已查看。自定义是这两个提案的 Qlib 表达式实现，不代表 QuantaAlpha 原项目的因子，也没有事后按回测收益再选择特征。

## 实验口径

| 内容 | 固定参数 |
|---|---|
| 数据 | 本地 Qlib `data/qlib/sp500_provider`，US 日线；20 个选自 Qlib **官方 Alpha158** 配置的特征（**不是** QuantaAlpha 特定的 Alpha158(20) 名单） |
| 额外 2 个特征 | `Slope(Ref($close,30)/$close,30)` 和 `Mean(Corr($close,Log($volume+1),20),20)` |
| 分组 | 官方 20；额外 2；官方 20 + 额外 2；三组独立训练相同 LightGBM 回归模型 |
| 训练／验证／历史比较 | 2018–2020／2021／2022–2023；测试区间**已被历史研究使用**，只称历史比较 |
| 标签 | 信号日之后**市场下一交易日**开盘至收盘收益；无下一市场日有效报价的股票标签缺失，训练/验证期末标签单独隔离 |
| 组合 | 每天 T 收盘信号 → T+1 开盘成交，Topk 50、日替换最多 10、95% 风险仓位；开平各 5bp + 冲击 5bp；统一等权本地 provider 基准 |

```bash
/home/jack/miniconda3/envs/tradingagents/bin/python \
  quant-method/machine-learn/run_qlib_three_way.py \
  --output-dir quant-method/machine-learn/data/qlib/qlib_three_way_new
```

输出目录必须全新。若换 Qlib provider，请传 `--provider-dir`，须包含 `calendars/day.txt` 和 `instruments/all.txt`；资产池与口径会随 provider 改变，必须作为**新实验**，不能与此表视为同一数据。失败记 `status.json=failed`，不覆盖结果。

## 已运行：本地美股历史比较

产物（Git 忽略）`data/qlib/qlib_three_way_retrospective_v3/`，`status.json=completed_research`，含 `manifest.json`（provider 日历/成员表和代码哈希）、`factor_expressions.json`、`report.md`、`results.json`，每组单独保存 `predictions.csv`、`backtest/summary.json`、逐日 `portfolio_report.csv`、`positions.pkl` 和日志。共同交易日窗口及基准经过核对。

| 特征集合 | 2021 验证 Rank IC | 2022–2023 **累计**扣费收益 | 等权参考 | 相对参考 | 最大回撤 |
|---|---:|---:|---:|---:|---:|
| Qlib 官方 Alpha158 中固定 20 个 | 0.0086 | +4.74% | +10.05% | -5.31 pct | -21.83% |
| 两个事后 LLM 提案 | -0.0066 | +7.45% | +10.05% | -2.61 pct | -20.52% |
| 20+2 | 0.0056 | +10.92% | +10.05% | +0.87 pct | -25.22% |

注意：训练标签是下一交易日**开盘到收盘**，但 Qlib TopkDropout 对未替换股票会**持续持仓并计入隔夜收益**；三组组合执行口径相同，模型标签却不覆盖全部持仓收益。这与此前“每天收盘全平”的本地演示不同，需在后续单独对齐和审计。

合并组历史累计收益略高于同口径参考，但 **2021 验证 Rank IC 反而低于基础组**，风险回撤更大，Sharpe 仅 **0.334**；**不能说两个 LLM 因子已证实有增益**。所有数字是两年累计值，不能写成“年化 10.92%”或与之前 10% 仓位日内演示直接比较。此实验也不是 QuantaAlpha 声称结果的复现。

若要下一步研究更可信的增益，先冻结候选生成和模型选择规则、使用以前没有参与选择的市场/未来区间、取得 PIT 股票成员；在此之前不要用这个已看过的历史窗口调参追赶收益。
