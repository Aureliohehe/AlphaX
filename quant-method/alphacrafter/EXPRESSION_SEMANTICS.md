# 因子表达式与评价语义

当前约定：候选的 JSON 写法不影响求值。普通 `feature/sum/difference` 是原始字段 AST 的简写，展示、求值、新颖性分析与候选导出使用同一表达式定义。

| 简写 | 等价 AST / 含义 |
|---|---|
| `op=feature, features=[x]` | `{"op":"feature","feature":"x"}`，返回原始 x |
| `op=sum, features=[x,y]` | 原始 x + 原始 y |
| `op=difference, features=[x,y]` | 原始 x − 原始 y |
| 显式 `rank(x)` | 在每个信号日对 x 做横截面百分位排名；并列取平均秩 |
| `ts_rank(x,w)` | 股票自身滚动窗口内的时间序列排名，与横截面 rank 不同 |

所有候选通过 `evaluate_candidate_score(candidate, frame, dates=None)` 进入同一个安全 AST 求值器，frame 必须包含原始字段及 date，时间序列算子还需要 ticker。没有隐式标准化、截尾或输入排名；使用原始差值时，需要由候选设计者判断输入尺度是否合理。

如需保留过去“排名差”的经济含义，应显式写成：

```json
{
  "candidate_id": "rank_spread_5d_20d",
  "expression": {
    "op": "difference",
    "inputs": [
      {"op": "rank", "input": {"op": "feature", "feature": "return_5d"}},
      {"op": "rank", "input": {"op": "feature", "feature": "return_20d"}}
    ]
  }
}
```

评价阶段：Pearson IC 使用每日表达式输出与已兑现收益的相关性；Rank IC 对同一输出和收益分别排名再计算相关性。ICIR / Rank ICIR 使用日相关系数均值除以总体标准差（ddof=0），不年化。命中率为日 Rank IC 为正的比例，准入时按训练方向调整。覆盖率与排名换手沿用既有定义。一日反转评价基线也使用原始因子值。

组合阶段：Miner 诊断组合、Screener 与滚动 Miner 均先对完整表达式输出做逐日百分位排名并减去 0.5，再施加方向和权重。因此，不同量纲不会直接支配组合；任何必需成分缺失，组合信号保持缺失。组合标准化不回写单因子的 IC 评价。

## 版本与迁移

库/快照兼容策略版本升级为 `label-end-v2-raw-expression`。原 `label-end-v1` 的库与注册表不能默认继承或回放，需使用新输出目录重新准入；显式启用旧库回溯诊断也不构成与旧计算结果一致的复现。历史指标、方向、准入与收益不会自动更新。

固定技术候选 JSON 仍可读取，其中 `difference` 现在表示原始差值。若研究目标是历史排名差，须采用上面的显式 rank AST 并作为新候选记录。原有实验目录和历史报告不覆盖。

本次统一不调整标签口径、收益周期、准入阈值、股票池或交易执行规则；通过数量是否变化需另行运行实验验证。
