# AlphaCrafter 复现工程

已完成[冻结配置的正收益留出验证](POSITIVE_HOLDOUT.md)：四因子、30 只股票、周频调仓策略在 2024–2025 留出期扣费累计 **+28.87%**，两倍佣金及滑点下 **+28.09%**，两个年度均为正；141 项测试通过。最大回撤 -8.27%，仍未通过原 -8% 正式策略门槛；研究收益转正不代表论文复现或实盘认证。该留出期现已被观察，后续调参须另设独立检验。

新增[固定组合与绝对趋势对照](SIGNAL_STABILITY.md)：18 条开发期路径中，固定四因子标准成本累计 +5.22%；绝对趋势过滤的 15 只版本为 +6.67%，两倍成本 +7.28%。年度亏损和策略门槛问题仍未解决，未评估留出。另修复了近并列分数的浮点排序不稳定问题，137 项测试通过。

新增[正收益配置成本复核](POSITIVE_ROBUSTNESS.md)：修复排序后，周频剔除反转组在标准/两倍/三倍佣金与滑点下，开发期扣费累计为 +3.42%/+1.48%/-0.94%；2022 年均亏损，仍未通过策略门槛，未评估留出。

当前表达式和评价口径已统一，见 [表达式语义](EXPRESSION_SEMANTICS.md)：普通 JSON 与 AST 均按原始字段求值，只有显式 `rank` 才在表达式内部排名；IC 评价表达式输出，组合前再统一排名。旧版库和快照需在新实验目录重建，历史结果没有自动重算。

后续[仓位与因子组诊断](MEDIUM_FREQUENCY_ATTRIBUTION.md)完成 10 组开发期对照：补仓至日均约 77%–79% 后，5/20 日全因子扣费累计为 -10.39%/-7.50%；周频剔除反转组为 +3.42%，但年化仅 1.76%，所有配置仍未通过策略门槛。未评估留出。

已重跑[中低频收益测试](MEDIUM_FREQUENCY_RETURNS.md)：严格准入仍无候选；额外绕过准入的 12 因子探索性诊断，在已研究的 2022–2023 窗口，5/20 日调仓扣费累计收益分别为 -6.75%/-6.58%，均未通过策略门槛，未评估 2024–2025 留出。

新增 [中低频因子与策略 Harness](MEDIUM_FREQUENCY.md)：5/20 日开盘到开盘标签、12 个中长期候选、同周期成熟标签筛选，以及调仓间隔、最短持仓、排名缓冲、每日换手和费用预算。首轮真实研究池均无通过者，未强行生成策略收益，2024–2025 留出未评估。

已新增 [两层准入](TWO_LAYER_ADMISSION.md)：按周期筛选研究候选，再对冻结的组合做独立策略验证。CSI300 当前进入 1 日研究池 13 个、5 日 0 个；[跨日组合收益测试](CSI300_PORTFOLIO_EXPERIMENT.md)的三个配置在 2022–2023 验证期扣费亏损，均未通过，未评估 2024–2025 留出收益。原严格论文对照结果保持不变。

**当前状态：部分流程实现，尚未复现论文收益。** 论文对齐审计、严格 Trader 诊断结果和后续优先级见 [`REPRODUCTION_STATUS.md`](REPRODUCTION_STATUS.md)。固定三因子的准入失败明细见 [`GATE_DIAGNOSIS.md`](GATE_DIAGNOSIS.md)，后续固定十二个技术候选的探索性结果见 [`CANDIDATE_EXPLORATION.md`](CANDIDATE_EXPLORATION.md)，论文指标/标签口径对照见 [`METRIC_AUDIT.md`](METRIC_AUDIT.md)，raw 与 RankNorm 的配对量化见 [`RANKNORM_AUDIT.md`](RANKNORM_AUDIT.md)，Qlib Alpha158 候选与 GitHub 调研见 [`ALPHA158_PRESET_REVIEW.md`](ALPHA158_PRESET_REVIEW.md)，严格单因子与固定 XGBoost 模型两条探索路线见 [`TWO_TRACK_EXPERIMENT.md`](TWO_TRACK_EXPERIMENT.md)，数据时点/独立日历障碍见 [`DATA_PROVENANCE_AUDIT.md`](DATA_PROVENANCE_AUDIT.md)；严格实验仍未产生可行策略；另有明确降低准入门槛、已跑出模拟持仓但**不能当作论文或实盘收益**的 LLM Agent 执行演示，见 [`AGENT_EXECUTION_DEMO.md`](AGENT_EXECUTION_DEMO.md)。下文旧实验数字保留作历史记录，不代表修正后的严格口径结果。新的时间隔离、不可覆盖快照与历史回放用法见 [`TEMPORAL_REPLAY.md`](TEMPORAL_REPLAY.md)。

Miner Harness 不依赖 LLM 即可运行：

1. 从现有技术特征 schema 生成安全的候选因子表达式；
2. 按原始字段计算表达式（内部排名须显式声明），组合前做逐日横截面 RankNorm；
3. 计算训练、验证和留出回测的 IC、ICIR、Rank IC、Rank ICIR、Hit Ratio、Coverage、Turnover 和 Top/Bottom spread；
4. 按论文 Miner 的 1 日/5 日质量门槛，通过训练集和验证集后写入因子注册表。

`return_1d` 与 `one_day_reversal` 数学上完全等价，因此不进入候选池，只作为简单反转基线。

## 固定预算批量候选研究

`run_candidate_batch.py` 将已冻结的 12 个 LLM 提案与确定性技术候选按结构去重、受预算约束，逐项报告严格准入失败原因，并在**同一个历史准入窗口**做扣费模拟诊断；不会把事后生成的候选输出为历史可交易快照。已运行 32 候选研究实验，严格通过 0 个；命令和限制见 [`CANDIDATE_BATCH.md`](CANDIDATE_BATCH.md)。对其中 12 个冻结 LLM 因子的放宽准入敏感性实验见 [`LLM_GATE_SWEEP.md`](LLM_GATE_SWEEP.md)：仅 1 日弱正值准入能进 1 个，同时保留 1/5 日须把命中率降至 >49% 才进 1 个。进一步的 101 级连续放宽轨迹已让 12/12 全部进入**回溯研究名单**，但终点必须允许负 IC/ICIR 和仅约 43% 的方向命中，故不能当作有效准入规则。以上都不是新样本外收益，也不改变下述严格流程。阶梯各因子集合在已看过的 2023 历史窗口逐个运行 Screener 和本地 Trader 的扣费对比见 [`GATE_LADDER_COMPARISON.md`](GATE_LADDER_COMPARISON.md)；历史最高收益不能据此选择有效门槛。

## 自动滚动实验

`run_pipeline.py` 已串起按日期维护 Miner、Screener 历史回放、独立 Trader 验证和冻结后的留出评估，详见 [`PIPELINE.md`](PIPELINE.md)。先检查计划：

```bash
python3 quant-method/alphacrafter/run_pipeline.py \
  --output-dir quant-method/machine-learn/data/qlib/alphacrafter_pipeline_pit \
  --dry-run
```

默认每 90 个日历日维护，准入窗口为最近 252 个交易日；Trader 选择文件先冻结，才开始生成留出期快照。输出目录不能复用，失败保留日志，不自动恢复。旧缓存未覆盖默认完整窗口时会明确失败，需要重建数据或显式缩短实验范围。

## 单轮 Miner

在项目根目录执行：

```bash
python3 quant-method/alphacrafter/miner_lite.py \
  --output-dir quant-method/machine-learn/data/qlib/alphacrafter_miner_pit
```

首次运行请使用新目录：旧版本 registry 没有时间隔离认证，默认拒绝继承。

默认输入：

```text
quant-method/machine-learn/data/processed/sp500_features.csv.gz
```

默认输出：

```text
quant-method/machine-learn/data/qlib/alphacrafter_miner_lite/
```

主要文件：

- `candidates.json`：本轮候选因子
- `candidate_metrics.csv`：所有候选的分段指标
- `accepted_factors.json`：通过准入门槛的因子
- `predictions.csv`：仅快照可用日的等权 composite 信号（不向历史回填）
- `diagnostic_predictions.csv`：最终库在历史数据上的回算，不能用于样本外评估
- `snapshots/YYYY-MM-DD.json`：不可覆盖的当日因子库快照
- `temporal_audit.json`：信息截止、标签结束日、分段剔除行数
- `registry.json`：因子注册表和维护状态快照
- `factor_library.json`：持久化的活跃因子库
- `report.md`：本轮报告
- `novelty.json`：候选因子结构多样性和相对历史因子的新颖性

默认切分为训练集 2016–2022、验证集 2023、留出回测 2024–2026，可通过 `--train-start/end`、`--validation-start/end`、`--backtest-start/end` 修改。`--as-of` 现在是真实信息截止：未来特征不可见，1/5 日标签分别在各段期末做 purge。留出窗口只报告截止日以前已经兑现的标签。

若需要独立的 2023 年 Trader 验证，应先在 2022 年末完成因子准入，不能用默认 2023 年准入库回算 2023 年信号。

## Qlib 组合回测

以下为旧版实验命令与结果，不是当前版本重新验证的成绩。历史连续信号现在应通过 Screener 快照回放生成，不能使用 `diagnostic_predictions.csv` 代替。旧实验的 7 个通过因子等权平均，使用 Topk=50、每天最多替换 10 只股票：

```bash
conda run -n tradingagents python quant-method/machine-learn/qlib_portfolio_backtest.py \
  --predictions quant-method/machine-learn/data/qlib/alphacrafter_miner_lite/predictions.csv \
  --n-drop 10 \
  --output-dir quant-method/machine-learn/data/qlib/portfolio_backtest_miner_lite_drop10
```

结果写入 `portfolio_backtest_miner_lite_drop10/`。当前留出回测累计收益 **26.53%**、年化收益 **11.61%**、Sharpe **0.61**、最大回撤 **-24.91%**；相对同口径等权基准低 **13.72 个百分点**。Miner 和 Screener 默认使用 `target_open_close`（T+1 开盘到收盘）进行因子评估，和 Trader 的执行时点一致；旧版特征文件会明确回退到 close-to-close 标签。

已进一步完成嵌套滚动因子准入：训练集确定方向，前一年度验证集选择因子数量和净收益，后续年度只做前向评估，结果见 [`MINER_WALK_FORWARD.md`](MINER_WALK_FORWARD.md)。滚动版本最终留出收益为 **25.55%**，相对基准低 **14.69 个百分点**，暂不建议启用 Miner-lite 作为默认策略。

## 接入未来 LLM Miner

### 训练内反馈搜索（可选）

首轮真实 DeepSeek 实验已完成：30 提案、28 唯一候选、严格准入 0 个，见 [反馈搜索实验报告](FEEDBACK_SEARCH_EXPERIMENT.md)。

新增 CSI300 数据测试：52 个单字段因子 + 28 个冻结 LLM 组合，2010–2018 训练、2019–2021 准入，严格通过 0/80；有局部短期排序信号，2022–2025 留出指标未计算。见 [CSI300 因子测试](CSI300_FACTOR_EXPERIMENT.md) 和 [数据审计](CSI300_DATA_AUDIT.md)。

默认仍为单轮生成。增加 `--search-rounds` 可启用有限预算的研究闭环：

```bash
python3 quant-method/alphacrafter/miner_lite.py \
  --llm-model deepseek-chat \
  --search-rounds 3 --search-budget 30 --search-batch-size 10 \
  --search-folds 3 \
  --output-dir quant-method/machine-learn/data/qlib/miner_feedback_new
```

需要配置模型环境变量；该命令会调用外部模型。反馈搜索仅支持 direct 后端，不能同时指定候选文件。此模式用 `search-budget` 和 `search-batch-size` 控制候选槽位，忽略 `llm-count`；重复或不足的提案也消耗槽位，不自动补抽。预算限制请求的候选数量，不限制 token 或费用；模型超额返回时截断。

搜索器仅接收训练窗口的行，先截断跨训练期末的标签，再将交易日均分成 `search-folds + 1` 个连续块。第一块作初始方向估计；后续每折用此前所有块的已兑现 1 日标签定方向，评价下一块的 1/5 日表现，并在每块末尾分别 purge。每块至少需要 5 个交易日（实际研究应使用远长于此的窗口）。反馈包含各折 IC、Rank IC、ICIR、命中率、覆盖率、换手及失败原因；后续提案接收此前全部反馈，跨轮按表达式与 ID 去重。

`search_feedback.json` 保存每轮预算和逐候选诊断。全部唯一候选冻结后才进入原训练/验证准入逻辑，**不按折内最好成绩自动准入，也不更改正式门槛**。搜索折已被用于自适应研究，不是独立样本外；验证/留出指标不反馈给生成器。该功能没有修复历史成分股数据问题，也不保证找到有效因子。相似度目前只处理表达式重复，尚不处理信号相关性或极端值归因。

LLM 不需要修改验证器，只需输出候选 JSON：

```json
[
  {
    "candidate_id": "volume_reversal_20",
    "op": "difference",
    "features": ["volume_ratio_20d", "volume_volatility_20d"],
    "hypothesis": "成交量冲击和成交量稳定性的差异具有短期预测力"
  }
]
```

然后运行：

```bash
python3 quant-method/alphacrafter/miner_lite.py \
  --candidate-file candidates_from_llm.json
```

候选可以使用安全 AST 表达式；所有表达式都会经过 schema、深度和窗口校验，避免执行任意 LLM 代码。启用 LLM 时默认运行 3 个专业 Miner agent，分别关注动量/反转、波动/成交量和复合信号，最后按表达式去重。

当前验证器同时评估 1 日和 5 日前瞻收益，并记录 IC、ICIR、Rank IC、Rank ICIR、Hit Ratio、Coverage 和横截面排名 Turnover。默认准入门槛遵循论文：1 日 IC/Rank IC > 0.015、ICIR/Rank ICIR > 0.20；5 日对应阈值为 0.025/0.25；方向调整后的 Hit Ratio > 0.60、Coverage > 0.90、Turnover < 0.40。可通过 `--min-train-ic`、`--min-valid-ic` 等参数显式覆盖 IC 门槛。

```bash
python3 quant-method/alphacrafter/miner_lite.py \
  --min-coverage 0.90 \
  --max-turnover 0.40
```

旧版特征文件如果没有 `target_5d`，Miner 会从 `return_1d` 安全推导；重新运行 `build_features.py` 会直接生成该标签。详细多周期指标写入 `candidate_metrics.csv` 和 `report.md`。

因子库持久化到 `factor_library.json` 和按日期归档的 `snapshots/`；已有因子通过 registry 继承，默认每 90 天到期重验。`--as-of YYYY-MM-DD` 同时约束所有可见数据与维护时点。快照不可覆盖，同一截止日重跑不同实验请使用新目录；滚动训练/验证窗口目前需调用者显式传入。

当前还支持安全的因子 DSL/AST，不执行 LLM 返回的任意代码。例如：

```json
{
  "candidate_id": "return_5d_ts_zscore",
  "features": ["return_5d"],
  "expression": {"op": "ts_zscore", "feature": "return_5d", "window": 20}
}
```

允许的时间序列算子包括 `ts_mean`、`ts_std`、`ts_sum`、`ts_min`、`ts_max`、`ts_quantile`、`ts_delta`、`ts_slope`、`ts_rsquare`、`ts_resi`、`ts_zscore`、`ts_rank` 和 `ts_corr`，另有 `abs`、`log` 及安全算术组合，窗口限制为 2–252 个交易日。模板模式可用 `--max-temporal` 控制时间序列候选数量。

也可以显式启用 MinerLLM 作为候选生成器；LLM 只负责提出 JSON，后续验证和准入逻辑完全相同。DeepSeek 使用内置的 OpenAI-compatible 客户端，不需要安装 LiteLLM：

```bash
export DEEPSEEK_API_KEY=...
python3 quant-method/alphacrafter/miner_lite.py \
  --llm-model deepseek-chat \
  --llm-count 40
```

也支持 `deepseek/deepseek-chat`、`deepseek-reasoner`。可用 `--llm-agents` 调整专业 agent 数量。默认的 `direct` 后端顺序调用 Agent；安装 LangGraph 后可以用 `Send` 做并行 fan-out/fan-in：

```bash
python3 -m pip install -r quant-method/alphacrafter/requirements.txt
python3 quant-method/alphacrafter/miner_lite.py \
  --llm-model deepseek-chat \
  --llm-orchestrator langgraph \
  --llm-agents 3 \
  --llm-count 40
```

LangGraph 只负责候选生成编排；候选返回后仍经过相同的 AST 校验、去重、IC/ICIR 准入和因子库维护。其他模型名仍通过 LiteLLM 调用，例如：

```bash
python3 -m pip install litellm
export ANTHROPIC_API_KEY=...
python3 quant-method/alphacrafter/miner_lite.py \
  --llm-model anthropic/claude-sonnet-4-5 \
  --llm-count 40
```

默认不调用外部模型，使用可复现的模板生成器。API key 只从环境变量读取，不会写入输出文件。因子库现在保存每次再验证的 `validation_history`，注册表保存 `run_history` 和淘汰因子的最后失败指标。

## 扩展基本面和新闻数据

`build_features.py` 支持可选的 point-in-time CSV 面板，必须包含 `ticker` 和 `available_date`（没有时才使用 `date`）。数据只通过 as-of backward join 合并，不会使用发布日期之后的信息：

```bash
python3 quant-method/machine-learn/build_features.py \
  --fundamentals fundamentals.csv \
  --news news.csv
```

基本面字段会生成 `fundamental_*` 特征；新闻面板支持 `sentiment`/`sentiment_score`/`polarity`，生成每日 `news_count` 和 `news_sentiment`。缺失的外部数据不会被静默填充，最终由 Miner 的 Coverage 门槛处理。

## Screener-lite

Screener-lite 按过去 10 个信号日 Rank IC 动态筛选因子，要求这些标签已在当前日收盘前兑现；延迟/未知结束日时保守不筛选。默认只接受新版时间隔离快照，快照在 `as_of` 当日收盘可用，之前不产生信号。支持 `--library-history snapshots/` 逐日恢复准入/淘汰状态（见时间回放文档）。旧库仅可显式加 `--allow-retrospective-library` 做回溯诊断，不能作为无泄漏结果。以下静态库示例须先替换为新生成的库路径：

```bash
python3 quant-method/alphacrafter/screener_lite.py \
  --factor-library quant-method/machine-learn/data/qlib/alphacrafter_miner_persistent/factor_library.json \
  --output-dir quant-method/machine-learn/data/qlib/alphacrafter_screener_lite
```

输出的 `predictions.csv` 可直接用于 Qlib 组合回测；`selection.csv` 记录每天选中的因子、方向和权重，`regime.csv` 记录市场状态摘要。可以单独启用最新信号日的 DeepSeek regime judge：

```bash
python3 quant-method/alphacrafter/screener_lite.py \
  --factor-library quant-method/machine-learn/data/qlib/alphacrafter_miner_persistent/factor_library.json \
  --llm-regime-model deepseek-chat
```

当前 LLM regime 结果只写入 `regime.csv` 和 `registry.json`。语义 relevance Tool Calling `judge_factor_relevance` 已接入最新信号日的筛选循环：输入 regime 和因子 AST，返回 `relevant`、`confidence`、`reason`。历史日期仍使用确定性筛选，避免回测期间产生大量 API 调用。

```bash
python3 quant-method/alphacrafter/screener_lite.py \
  --factor-library quant-method/machine-learn/data/qlib/alphacrafter_miner_persistent/factor_library.json \
  --llm-regime-model deepseek-chat \
  --llm-relevance-model deepseek-chat \
  --llm-similarity-model deepseek-chat
```

## Trader-lite

Trader-lite 使用 Screener 的预测信号，搜索显式定义的长短组合配置（`beta`、`gamma`、`nlong`、`nshort`），在验证集选择配置，再对后续留出集做一次前向评估：

```bash
python3 quant-method/alphacrafter/trader_lite.py \
  --predictions quant-method/machine-learn/data/qlib/alphacrafter_screener_lite/predictions.csv \
  --output-dir quant-method/machine-learn/data/qlib/alphacrafter_trader_lite
```

输出 `positions.csv`、`portfolio_returns.csv`、`summary.json` 和运行参数。v2 按论文 Algorithm 3 最多验证 3 个配置；仅从 AR > 8%、Sharpe > 0.6、MDD > -8% 的配置中选取最高 Sharpe，**没有可行策略就保持现金**，不回退选取失败配置。输入日期必须严格分离，并剔除执行日跨过验证期末的标签。

默认 β=0.8，首个候选 γ=0.5；美国市场年无风险收益默认 3.81%，可用 `--annual-risk-free-rate` 修改。最大回撤包含初始净值，零波动 Sharpe 输出 `null`。`summary.json` 记录输入 SHA-256 和 pandas/numpy 版本。

**执行口径是日内代理，不是论文的持续持仓模拟器**：T 收盘产生信号，T+1 开盘建仓、收盘平仓，每日开平仓都按实际成交额收取 1bp；无信号时现金等待，不沿用旧仓位。选股不依赖未来标签是否存在，持仓收益缺失会报错，不能默认零收益。尚未模拟保证金、借券、流动性、滑点，也不执行 LLM 生成代码或连接券商。

注意：Screener 新增快照回放，但不能凭空恢复旧实验缺失的历史快照。用 2023 年参与筛选的因子产生 2023 年信号，再用于 Trader 验证，仍属于重复使用验证信息。需重新按时间顺序生成因子库；旧回算结果只能作为诊断。

滚动因子准入可运行：

```bash
python3 quant-method/alphacrafter/run_miner_walk_forward.py
```

完整回归测试（Miner、Screener、Trader 及特征生成/命令行集成）：

```bash
cd quant-method/alphacrafter
python3 -m unittest discover -v
```

特征生成和 Screener 现在保留没有未来标签的最新交易日；Screener 合成信号时任何成分缺失都保留为缺失，不再因因子顺序不同得到不同结果。旧特征/信号缓存不会自动更新，需要另行重建。

## 四因子与多 Agent 加权回测

四因子策略为核心仓，多 Agent 策略为探索仓。默认初始资金分配为 80% / 20%，用 `--agent-weight` 调整。两路分别按净值滚动，期间不划拨资金，实际权重随表现漂移。每日合并净值等于两个子账本的净值之和，等价于按前一日资金权重加权当日净收益。

1. 多 Agent Miner 生成候选，沿用上游准入流程，Screener 对通过准入的沪深 300 候选生成收盘分数。
2. 将获准的沪深 300 Screener `predictions.csv` 传给 `--approved-agent-signals`。入口兼容 `datetime,instrument,prediction` 和 `date,ticker,prediction`，检查历史成员、信号日和唯一性。调用方负责确认准入和信号来源。首个执行日需要上一个交易日的收盘信号，信号文件的末日最迟为最后执行日的前一交易日。
3. 若本窗口没有获准策略，使用 `--agent-cash`。Agent 子账本按冻结四因子实验的 A 股执行参数模拟，标准和双倍佣金及滑点分别输出。产物包含三路指标、合并逐日净值、Agent 交易和持仓，以及输入指纹。

```bash
python3 quant-method/alphacrafter/run_core_agent_blend.py \
  --core-dir quant-method/machine-learn/data/qlib/alphacrafter_positive_holdout_20261117 \
  --agent-cash --agent-weight 0.20 \
  --output-dir quant-method/machine-learn/data/qlib/alphacrafter_core_agent_blend
```

该入口从已完成的四因子实验读取窗口与成本设置。2024–2025 的数据已被观察，可用于链路回放。组合仍可能亏损，评估 Agent 增量需要预先冻结权重、准入与执行规则，并使用新的评估窗口。

## 四因子模板与替换对照

`candidates_four_factor_core_v1.json` 保存四项原始因子，使用 Miner 候选格式。配套周频研究入口按原公式生成信号，每个模板只改一个窗口参数，逐一替换组合中对应的一项。2022–2023 年的两档成本结果用于冻结一个变体，按最差年度扣费收益排序。随后在已观察过的 2024–2025 年回放四因子对照、替换后的四因子组合和只用所选模板信号的组合。三路采用同一执行配置，回放基准净值与历史四因子账本逐日复核。

```bash
python3 quant-method/alphacrafter/run_four_factor_template_research.py \
  --output-dir quant-method/machine-learn/data/qlib/alphacrafter_four_factor_templates
```

该入口使用 JSON 预设和确定性模板，研究的是单项替换的扣费表现。原 Miner 的 1 日和 5 日准入门槛不参与模板选择。

### 按频率初筛和周频替换准入

日频研究继续使用既有初筛。周频和月频研究各自使用从次日开盘起持有 5 或 20 个交易日的收益标签，并只在对应调仓日抽样。阈值写入 `research_pool.json` 的 `policies_by_horizon`。以下是初始研究门槛，正式短周期 Miner 入库仍按自身规则执行。

| 频率 | Rank IC | Rank ICIR | 方向命中率 | 覆盖率 | 调仓观察间的排名换手 |
|---|---:|---:|---:|---:|---:|
| 日频 | >0.015 | >0.10 | >55% | >90% | <40% |
| 周频 | >0.010 | >0.10 | >53% | >90% | <50% |
| 月频 | >0.010 | >0.08 | >52% | >90% | <65% |

周频四因子替换还需要在独立窗口内，同时通过标准与双倍佣金、滑点的对照。每档都要求净收益提高、最大回撤和最差年度收益不恶化。

### A 股 TraderHarness 选择调仓频率

运行 `run_medium_frequency.py --miner-dir <Miner 输出目录> --evaluate-strategies --output-dir <新目录>`，同一验证窗口内评估日频、周频、月频配置。日频 A 股使用次日开盘建仓和至少一个交易日的持仓锁，按开盘到开盘的标签研究；它与原日内 Trader 的开盘到收盘口径不同。没有候选因子或没有筛出信号的频率记为跳过。TraderHarness 在收益、Sharpe、回撤、执行约束合格者中选扣费后 Sharpe 最高的频率，同分优先调仓更慢的配置。完整选择写入 `strategy_selection.json`，无合格配置时返回空仓决策。

频率入口只从 Miner 的 candidates.json 和 registry.json 接收候选，支持 JSON、模板和 LLM 三种来源。候选用各频率的标签独立重评，Miner 旧的短周期入库结论不作为周频或月频准入。Python 量价字段可作为候选的输入。若表达式使用额外字段，通过 --feature-panel 提供按交易日和股票代码对齐的特征。历史实跑文件保留原命令和来源。

### 组合指标初筛

`run_combination_metrics.py` 从 Miner 输出读取四因子核心及带 `control_id` 的替换候选，并使用频率研究产出的特征和开盘到开盘标签。它用原组合和每个单项替换做配对比较，分别输出组合 Rank IC、Rank ICIR、滚动窗口胜数、前 30 名最差一成持有收益、名单更换和覆盖率。前 30 名只按信号排，若有任一股票缺少已成熟收益，该日不进入配对持有收益统计。日、周、月以 1、5、20 个交易日分别采样。该入口不模拟成交，也不写入交易准入结果。

```bash
python3 quant-method/alphacrafter/run_combination_metrics.py \
  --miner-dir <Miner输出目录> --panel <频率研究的features_labels.csv.gz> \
  --calendar <qlib_day.txt> --start 2022-01-01 --end 2023-12-31 \
  --output <新报告.json>
```
