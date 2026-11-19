# 固定预算的候选因子批量诊断（回溯研究）

入口：`run_candidate_batch.py`。**不是新的样本外实验，也不是自动填充严格因子库的开关。** 2022 年收益已被以前实验查看；冻结的 12 个 DeepSeek 提案是在历史截止日之后生成的，不能倒签为当年可用。额外 20 个技术候选由可解释的安全 AST 模板生成；全部候选在运行前固定、验证输入特征 schema、按表达式结构去重并受预算约束。候选假设、来源和去重结果写在 `generation_audit.json`。目前没有调用新 LLM API，也没有跨期历史 LLM 调用。

```bash
python3 quant-method/alphacrafter/run_candidate_batch.py \
  --budget 32 \
  --output-dir quant-method/machine-learn/data/qlib/alphacrafter_candidate_batch_new
```

默认使用本地 Alpha158 风格特征和显式标签结束日/XNYS 日历；训练 2016–2021、2022 准入，信息截止 2022-12-30；不读取 2023+ 的收益进行准入或交易诊断。通过未修改的 `miner_lite.py` 严格 1/5 日门槛逐项评估，输出 `miner/candidate_metrics.json`、`findings.json`（假设、方向、四段 IC/ICIR/命中率等、逐项失败原因及交易诊断）、`report.md`、`feedback.md` 和 `status.json`。额外使用**固定训练方向**对每个因子在**同一个 2022 准入窗口**单独模拟：纯多头 10 只、最多 10% 总资金、T 收盘信号 → T+1 日内收益、每天双边各 1bp；缺失持仓收益则 `failed_closed`，不能视为零或换股。这不是另一套无泄漏留出期，更不能用这里的收益挑因子后声称 2022 是样本外。`--skip-trades` 可只诊断因子质量。目录必须是新目录，失败写 `status.json=failed`，不自动覆盖。

**关键安全边界**：Miner 为计算历史指标会暂时生成带历史日期的快照，但脚本立即删除其 `factor_library.json`、`registry.json`、快照、准入库与预测信号，不向 Screener 输出假冒 PIT 的交易因子库；`miner/RETROSPECTIVE_ONLY.txt` 说明用途。`findings.json` 中 `strict_accepted` 只表示历史数值门槛是否通过，**不表示拥有历史可用性认证**。

## 已运行：32 候选研究诊断

本地产物（未纳入 Git）：`machine-learn/data/qlib/alphacrafter_candidate_batch_v1/`，`status.json=completed_research`。12 个冻结 LLM 候选 + 20 个模板候选（趋势、蜡烛/区间、风险、量价及部分时间序列变化）；本批没有结构重复。**0/32 通过原严格门槛**；在训练、准入 × 1/5 日共 128 个细分检查中，ICIR 未通过 **128/128**、Rank ICIR **128/128**、命中率 **128/128**，Coverage/Turnover 不是主要瓶颈。32 个 2022 准入期模拟都产生过仓位，其中 8 个净收益为正，但这些收益与训练/准入指标来自同一段历史，不足以证明因子有效；不得按模拟收益事后筛选并把它们放入严格库。相比单纯增大 LLM 生成数量，这批结果更提示要核查**标签/数据时点/候选机制与门槛口径**，而不是降低门槛直到出现交易。

数据仍存在当前成员快照用于历史截面的幸存者偏差；日内模拟不含成交价/股数/滑点。任何真正可交易收益结论需修复历史成分 PIT、使用未被反复查看的新窗口，并冻结候选生成与选择规则。
