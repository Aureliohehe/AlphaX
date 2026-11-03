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
- `registry.json`：因子注册表快照
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

也可以显式启用 LiteLLM 作为候选生成器；LLM 只负责提出 JSON，后续验证和准入逻辑完全相同：

```bash
python3 -m pip install litellm
export ANTHROPIC_API_KEY=...
python3 quant-method/alphacrafter/miner_lite.py \
  --llm-model anthropic/claude-sonnet-4-5 \
  --llm-count 40
```

也支持 `openai/gpt-4.1` 等 LiteLLM 模型名。默认不调用外部模型，使用可复现的模板生成器。

滚动因子准入可运行：

```bash
python3 quant-method/alphacrafter/run_miner_walk_forward.py
```

时间边界和成本选择的回归测试：

```bash
cd quant-method/alphacrafter
python3 -m unittest -v test_harness.py
```
