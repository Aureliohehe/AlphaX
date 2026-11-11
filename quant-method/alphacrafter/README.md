# AlphaCrafter 复现工程

**当前状态：部分流程实现，尚未复现论文收益。** 论文对齐审计、严格 Trader 诊断结果和后续优先级见 [`REPRODUCTION_STATUS.md`](REPRODUCTION_STATUS.md)。下文旧实验数字保留作历史记录，不代表修正后的严格口径结果。新的时间隔离、不可覆盖快照与历史回放用法见 [`TEMPORAL_REPLAY.md`](TEMPORAL_REPLAY.md)。

Miner Harness 不依赖 LLM 即可运行：

1. 从现有技术特征 schema 生成安全的候选因子表达式；
2. 逐日横截面 RankNorm；
3. 计算训练、验证和留出回测的 IC、ICIR、Rank IC、Rank ICIR、Hit Ratio、Coverage、Turnover 和 Top/Bottom spread；
4. 按论文 Miner 的 1 日/5 日质量门槛，通过训练集和验证集后写入因子注册表。

`return_1d` 与 `one_day_reversal` 数学上完全等价，因此不进入候选池，只作为简单反转基线。

## 运行

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
