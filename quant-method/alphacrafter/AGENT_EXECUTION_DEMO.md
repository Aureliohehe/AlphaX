# LLM Miner 因子进入执行链路：真实数据探索性演示

**目标是验证代码链路可以产生模拟成交，不是论文严格复现、样本外 Alpha 或真实券商订单。** 用户要求优先看到 Agent 发现因子并纳入执行，因此本演示在完全隔离的输出目录采用**显式降低的实验准入规则**。原 `miner_lite.py` 的论文 1/5 日严格规则、`run_pipeline.py` 的冻结/拒绝逻辑均未替换；本次不声称严格策略已通过。

## 实际执行顺序

1. `miner_lite.py` 用已有的 `DEEPSEEK_API_KEY` 调用一次 `deepseek-chat` Miner（1 agent、最多 12 个候选），输入截至 **2022-12-30** 可见的 Alpha158 风格特征，训练 2016–2021、准入验证 2022，后续收益不入 prompt。LLM 返回安全 AST，逐项验证，候选和模型响应哈希记录在本地 `alphacrafter_agent_2022_strict_v2/`；没有记录密钥。第一次请求因冗余 `features` 列表和 AST 叶子不一致被安全验证器拒绝，未形成因子库；随后代码仅对已验证安全 AST 的冗余叶子列表作归一化，并允许逐项拒绝非法表达式，重新调用一次成功（两次调用均涉及外部 API）。
2. **严格 Miner：12 个候选，0 个通过**。这属于正常拒绝，不被改写。用它保存的 `candidates.json`、`candidate_metrics.json` 在**另一个独立实验目录**应用明确命名的演示规则：仅要求 1 日的训练/准入验证两段，方向调整后 IC、Rank IC、ICIR、Rank ICIR 均 >0，Rank IC 为正的天数比例 >50%，Coverage >90%、排名换手 <40%；**不要求 5 日指标或论文 ICIR >0.2、命中率 >60%**。2022 截止仅 `roc30_ts_slope_30` 一项通过该演示规则；其 1 日训练 Rank IC 约 **0.00115**、2022 准入 Rank IC 约 **0.00891**，预测强度很弱。
3. `run_agent_execution_demo.py` 将这一项以 `strict_paper_accepted=false`、`admission_policy=...` 写入**独立的回溯诊断因子库** `factor_library_demo.json`。LLM 候选是当前才生成的，**不能伪造 2022 年历史准入快照**；库明确记录实际 `generated_at_utc` 和仅用于指标计算的 `historical_metric_cutoff=2022-12-30`。Screener 必须显式使用 `--allow-retrospective-library`，输出 `library_scope=retrospective_diagnostic`，随后在 2023 历史信号日回算：沿用过去 10 个已成熟交易日 Rank IC 的 `|suitability| >= 0.02`，无信号时保持现金；没有用未来收益筛选当日股票。**该库不能冒充按历史时点生成的 PIT 因子库。**
4. 为检验交易而非冒充论文 Trader，**事先固定**模拟配置：纯多头 10 只、总权重 10%、T 收盘信号→T+1 开盘建仓/收盘清仓，每腿成交额 1bp；不使用 Trader 的 AR>8%/Sharpe>0.6/MDD>-8% 选择门槛。`load_inputs()` 检查标签真实兑现日；持仓收益缺失则失败。没有接入券商、保证金或持续持仓模型。

2023 年模拟执行结果（本地 `machine-learn/data/qlib/alphacrafter_agent_execution_demo_v2/`，`status.json=completed_exploratory_demo`）：

| 指标 | 结果 |
|---|---:|
| LLM 候选 / 演示准入 / 论文严格准入 | 12 / **1** / **0** |
| Screener 信号股票行 | 83,654 |
| 验证期有仓位日期 / 逐只持仓行 | **172 天 / 1,720 行** |
| 日内扣费累计 / 年化收益 | **+3.72% / +3.76%** |
| Sharpe（扣年无风险利率 3.81%） / 最大回撤 | **-0.015 / -1.35%** |
| 交易口径的无费用等权参考值 / 相对差额 | +14.61% / -10.89 个百分点 |

**能成交，不等于因子合格或策略有效**：该结果低于参考值，Sharpe 甚至略负；对真实资金没有上线依据。2023 已在之前实验中查看过，2025 年后的股票当前名单也被用于历史面板，LLM 本身可能记忆历史结果，所以该收益**仅说明代码链路真实地产生了仓位和成本，不构成独立样本外验证**。

产物：`factor_library_demo.json`（单项因子、真实生成时点、非论文准入与回溯诊断标识）、`admission_decisions.json`（所有候选决策）、`screener/{predictions.csv,selection.csv,regime.csv}`、`trades_daily.csv`、`positions.csv`、`summary.json`、`manifest.json`（输入/代码 SHA-256 和固定配置）、`status.json`。原始 Miner 的 `accepted_factors.json` 仍是空列表。早先的本地 `...execution_demo_v1/` 曾把事后生成的 LLM 因子写成 2022 年快照，**那是时间来源错误的旧诊断，不能使用**；v2 移除这种历史快照并标记回溯范围。使用新目录重跑：

```bash
python3 quant-method/alphacrafter/run_agent_execution_demo.py \
  --miner-dir quant-method/machine-learn/data/qlib/alphacrafter_agent_2022_strict_v2 \
  --input quant-method/machine-learn/data/qlib/alphacrafter_alpha158_curated_v1/features.csv.gz \
  --calendar quant-method/machine-learn/data/processed/sp500_xnys_xcal_4_13_3.csv \
  --output-dir quant-method/machine-learn/data/qlib/alphacrafter_agent_execution_demo_new
```

下一步若目标是**正常可上线的 Agent 交易**，必须先定义最低可接受风险和真实新窗口，纳入完整数据时点、交易滑点/成交约束和券商 paper-trading 接口；不能把本演示的宽松因子库转存为严格库，或看完 2023 收益再修改规则宣称未用验证集。
