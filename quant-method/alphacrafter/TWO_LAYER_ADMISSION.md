# 两层准入：研究候选池与正式策略验证

另有独立的 [5/20 日中低频路线](MEDIUM_FREQUENCY.md)，使用可执行周期标签及交易预算；下文原 1/5 日研究池作为历史实验保留。

现在将两个决定分开：因子是否值得继续研究，以及冻结后的组合是否通过独立的策略验证。原论文单因子严格门槛保留作对照，不再是进入研究池或参加组合验证的必要条件；研究池也不是正式交易因子库。

## 第一层：按周期筛选研究候选

`admission.py:ResearchPolicy` 默认阈值：

| 指标 | 训练期与开发期都需满足 |
|---|---:|
| 方向调整后 Rank IC | >0.015 |
| 方向调整后 Rank ICIR | >0.10 |
| 方向命中率 | >55% |
| Coverage | >90% |
| Turnover | <40% |

1 日、5 日分别评价，不要求同时通过。每个周期分别用该周期训练 Rank IC 确定方向，开发期不能重新翻转。Pearson IC/ICIR 不作研究池硬门槛。缺失、非有限值、零有效观测、尚未完成的开发窗口不产生合格候选；留出指标不参与研究池选择。

Miner 自动新增 `research_pool.json` 与 `research_assessments.json`。前者按 `candidates_by_horizon` 存储合格候选及完整定义，后者保存每个候选/周期的指标和失败原因。所有研究候选为 `status=research_candidate`、`strategy_admitted=false`。

`--research-market cn/us` 标明市场；未指定时为 unknown，不能直接进入现有 Trader 的研究池路径。原 `accepted_factors.json`、`factor_library.json` 与快照仍使用原严格规则，没有把弱因子混入旧论文对照流程。

研究池内可保留高度相关的表达式，后续仍需做训练内年度稳定性、信号相关性和组合增量检查；候选数量不是独立 alpha 的数量。

## 第二层：冻结策略，再做独立验证

正式策略层沿用现有 Trader 的组合指标门槛：扣费后年化收益 >8%、Sharpe >0.6、最大回撤 >-8%，最多 3 个配置；只在通过者中选最高 Sharpe，没有可行配置则现金等待。

进入该层前需冻结因子、周期、训练方向、权重、选股/持仓规则、成交时点与费用。策略验证期必须晚于研究截止，最终留出期必须晚于策略验证期；不能用开发期表现再次冒充策略独立验证，也不要求研究池中每个单因子重新通过原 1/5 日联合强门槛。

现有 US Trader 可通过 `--research-pool ... --research-horizon 1` 引用研究池，随后对独立提供的预测执行原扣费组合验证。该参数校验研究截止、非空周期、市场和标签兼容性，输出 `admission_stage=formal_strategy_validation` 与研究池输入哈希。预测构建的时间正确性仍由上游策略负责；仅引用研究池不能证明预测确实来自这些因子。

通用 Trader 仅支持 US execution 标签的 1 日日内代理。CN、paper_close 或 5 日研究池会明确拒绝。CSI300 现有独立入口 `run_cn_portfolio_research.py`，使用跨日持仓研究代理测试组合，包含 T+1、缺失估值标记及费用；其可分割复权单位和保守开盘过滤仍不等于真实成交模型。详见 [CSI300 收益测试](CSI300_PORTFOLIO_EXPERIMENT.md)。

Screener 明确拒绝把研究池当作 `factor_library` 读取，包括旧库回溯模式。这防止研究资格自动变成交易资格。原滚动管线继续运行严格论文对照；新研究池要经过单独的策略定义和验证后才有策略层结果。

## 本次 CSI300 研究池

复用已冻结的 80 候选训练/开发指标，不重跑行情或模型、不读取 2022–2025 因子表现：

- 2010–2018 训练，2019–2021 作为开发期。
- **1 日 13 个候选；5 日 0 个。** 主要是短期反转、收盘位置、均线偏离等，包括 `KSFT2`、`close_location` 与 `return_5d`；其中有高度相关或排序等价的表达式，不是 13 个独立因子。
- 研究池最初输出时状态为 pending_strategy_definition。后续已冻结三个组合，在跨日研究代理中评估 2022–2023，三个均失败（见收益测试）；不能宣称已获正式可交易性认证。
- 2024–2025 为最终留出；由于本轮没有可行配置，未评估留出期策略收益。原初始计划文件保留，后续结果单独记录。
- 放宽的研究阈值是在查看旧开发结果后决定的，因此这 13 个是探索性研究候选，不是新的独立验证结论。历史成员与复权的时点认证问题依然存在。

本地产物（Git 忽略）：`machine-learn/data/qlib/alphacrafter_csi300_research_pool_v1/`，包含研究池、160 条候选/周期评估、独立验证计划、源哈希与报告。原严格 0/80 实验保持不变。

可对已冻结的 Miner 指标生成独立研究池，使用新的输出目录：

```bash
python3 quant-method/alphacrafter/run_research_pool.py \
  --metrics quant-method/machine-learn/data/qlib/alphacrafter_csi300_frozen80_v1/admission/candidate_metrics.json \
  --candidates quant-method/machine-learn/data/qlib/alphacrafter_csi300_frozen80_v1/candidates_frozen.json \
  --temporal-audit quant-method/machine-learn/data/qlib/alphacrafter_csi300_frozen80_v1/admission/temporal_audit.json \
  --market cn \
  --output-dir quant-method/machine-learn/data/qlib/alphacrafter_csi300_research_pool_new
```

命令默认的独立策略验证/留出日期针对本次实验；其他窗口使用 `--strategy-validation-start/end`、`--holdout-start/end` 显式指定。输出只包含研究池与计划，不输出交易预测、库或历史快照。
