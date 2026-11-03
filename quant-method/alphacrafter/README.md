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
- `registry.json`：因子注册表快照
- `report.md`：本轮报告

当前时间切分为：训练集 2016–2022、验证集 2023、留出回测 2024–2026。验证集用于准入，留出回测只用于最终报告。

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
