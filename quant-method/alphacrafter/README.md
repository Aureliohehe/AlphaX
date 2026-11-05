# AlphaCrafter Miner-lite

这是 AlphaCrafter 的第一版 Miner Harness，不依赖 LLM 即可运行：

1. 从现有技术特征 schema 生成安全的候选因子表达式；
2. 逐日横截面 RankNorm；
3. 计算训练、验证和留出回测的 Rank IC、Rank ICIR、Hit Ratio、Top/Bottom spread；
4. 通过训练集方向和验证集准入门槛后写入因子注册表。

`return_1d` 与 `one_day_reversal` 数学上完全等价，因此不进入候选池，只作为简单反转基线。

## 运行

在项目根目录执行：

```bash
python3 quant-method/alphacrafter/miner_lite.py
```

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
- `predictions.csv`：通过因子方向调整后的等权 composite 信号
- `registry.json`：因子注册表和维护状态快照
- `factor_library.json`：持久化的活跃因子库
- `report.md`：本轮报告

当前时间切分为：训练集 2016–2022、验证集 2023、留出回测 2024–2026。验证集用于准入，留出回测只用于最终报告。

## Qlib 组合回测

`predictions.csv` 可直接接入 Qlib 组合回测。当前 7 个通过因子按验证集确定的方向做等权平均，使用 Topk=50、每天最多替换 10 只股票：

```bash
conda run -n tradingagents python quant-method/machine-learn/qlib_portfolio_backtest.py \
  --predictions quant-method/machine-learn/data/qlib/alphacrafter_miner_lite/predictions.csv \
  --n-drop 10 \
  --output-dir quant-method/machine-learn/data/qlib/portfolio_backtest_miner_lite_drop10
```

结果写入 `portfolio_backtest_miner_lite_drop10/`。当前留出回测累计收益 **26.53%**、年化收益 **11.61%**、Sharpe **0.61**、最大回撤 **-24.91%**；相对同口径等权基准低 **13.72 个百分点**。该组合仍沿用 Miner 的 `target_1d` 选因子逻辑，而实际在次日开盘成交，因此结果是无前视但标签与执行时点不完全匹配的保守测试。

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

当前允许的操作是 `feature`、`difference` 和 `sum`，避免执行任意 LLM 代码。

当前验证器同时评估 1 日和 5 日前瞻收益，并记录 Rank IC、Rank ICIR、Hit Ratio、Coverage 和横截面排名 Turnover。默认准入质量约束为 Coverage 不低于 0.90、Turnover 不高于 0.40，可通过以下参数调整：

```bash
python3 quant-method/alphacrafter/miner_lite.py \
  --min-coverage 0.90 \
  --max-turnover 0.40
```

旧版特征文件如果没有 `target_5d`，Miner 会从 `return_1d` 安全推导；重新运行 `build_features.py` 会直接生成该标签。详细多周期指标写入 `candidate_metrics.csv` 和 `report.md`。

因子库会持久化到 `factor_library.json`，每次运行自动加载已有因子；默认每 90 天重新验证并淘汰衰减因子。可用 `--maintenance-interval-days` 调整周期，或用 `--as-of YYYY-MM-DD` 指定维护日期。

当前还支持安全的因子 DSL/AST，不执行 LLM 返回的任意代码。例如：

```json
{
  "candidate_id": "return_5d_ts_zscore",
  "features": ["return_5d"],
  "expression": {"op": "ts_zscore", "feature": "return_5d", "window": 20}
}
```

允许的时间序列算子包括 `ts_mean`、`ts_std`、`ts_delta`、`ts_zscore`、`ts_rank` 和 `ts_corr`，窗口限制为 2–252 个交易日。模板模式可用 `--max-temporal` 控制时间序列候选数量。

也可以显式启用 MinerLLM 作为候选生成器；LLM 只负责提出 JSON，后续验证和准入逻辑完全相同。DeepSeek 使用内置的 OpenAI-compatible 客户端，不需要安装 LiteLLM：

```bash
export DEEPSEEK_API_KEY=...
python3 quant-method/alphacrafter/miner_lite.py \
  --llm-model deepseek-chat \
  --llm-count 40
```

也支持 `deepseek/deepseek-chat`、`deepseek-reasoner`。其他模型名仍通过 LiteLLM 调用，例如：

```bash
python3 -m pip install litellm
export ANTHROPIC_API_KEY=...
python3 quant-method/alphacrafter/miner_lite.py \
  --llm-model anthropic/claude-sonnet-4-5 \
  --llm-count 40
```

默认不调用外部模型，使用可复现的模板生成器。API key 只从环境变量读取，不会写入输出文件。

## Screener-lite

Screener-lite 从持久化因子库中按过去 10 个已产生标签的交易日 Rank IC 动态筛选因子，执行表达式去重，并按绝对适用性分配权重和方向。每个信号日只使用此前的收益标签：

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

滚动因子准入可运行：

```bash
python3 quant-method/alphacrafter/run_miner_walk_forward.py
```

时间边界和成本选择的回归测试：

```bash
cd quant-method/alphacrafter
python3 -m unittest -v test_harness.py
```
