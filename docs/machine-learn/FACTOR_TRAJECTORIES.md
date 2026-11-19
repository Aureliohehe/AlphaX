# 固定预算的 Qlib 因子轨迹实验

借鉴 QuantaAlpha 的假设→表达式→Qlib 回测→反馈→变异／交叉轨迹，不复制它的“RankIC>0 即成功”或对测试期反复进化。入口 [`run_factor_trajectories.py`](../../quant-method/machine-learn/run_factor_trajectories.py)。**这次不是新 LLM API 挖掘**：当前进程无 `DEEPSEEK_API_KEY`，先以此前 LLM 提出的两个历史假设为父节点，固定四个参数化窗口变异与一个横截面秩交叉，做可重复的最小实验。可选 `--mutations proposals.json` 接收恰好四条事先固定的 `parent_id`/`window`/`hypothesis` JSON；只允许两个已有父节点和窗口 5–60，**不执行外来代码或任意 Qlib 字符串**。每次使用新输出目录，事先锁定 8 次评估预算。

```bash
/home/jack/miniconda3/envs/tradingagents/bin/python \
  quant-method/machine-learn/run_factor_trajectories.py \
  --output-dir quant-method/machine-learn/data/qlib/qlib_factor_trajectories_new
```

- 使用已有 Qlib 美股 provider；固定 20 个 Qlib 官方 Alpha158 表达式 + 单个候选分别训练相同 LightGBM。训练 2018–2020，**仅在 2021 验证**；不读取 2022+ 的行情、标签或收益。T+1 市场下一交易日开收盘标签，验证期末剔除未兑现标签。Qlib Topk50 Drop10、95% 风险仓位、开平各 5bp 和冲击 5bp 固定，逐个保存 `predictions.csv` 与 Qlib `portfolio_report.csv`。
- 输出 `trajectories_planned.json`、`trajectories.json`（假设、父 ID、表达式、预测 Rank IC、对基线增量、因子间秩相关、扣费净收益与反馈）、`next_round_prompt.md`、输入/代码哈希、失败/成功状态；没有研究库注入、历史快照或任何新的测试期信号。**这里的 Rank IC 是基础 20 特征加上候选后的模型 Rank IC，不是候选单因子 IC**。Qlib TopkDropout 持续持仓可产生与开收盘训练标签不一致的隔夜收益；各组交易配置一致，但未解决模型标签与持仓收益不匹配问题。
- 2021 已在旧实验中查看过；两个原 LLM 假设是事后提出的，美股 provider 使用当前成分股回溯历史；本轮是**回溯训练反馈演示**，不能算新样本外验证，不能从这里选历史最优就声称 Alpha。

本地产物 `data/qlib/qlib_factor_trajectories_v4/`（Git 忽略，`status.json=completed_research`），看 `report.md` 与每个 `<id>/backtest/summary.json`。**8/8 次预算执行完成；没有一个变异/交叉让 2021 模型验证 Rank IC 超过基线 0.00861。** 最接近的是交叉 `cross_short`（0.00834）；`roc30_slope30_w45` 在同一验证窗口的组合累计收益 28.89%，对比基线 26.12%，但其 Rank IC 0.00819 仍较差。不可只按这年收益挑它再称为无偏选择。

要真正运行下一轮 LLM 研究，可将 `next_round_prompt.md` 交给模型输出四个**受约束的结构化提案**，先冻结文件和预算再执行；若沿用相同 2021 窗口多轮迭代，它也会逐渐被过拟合，最终必须用新的、真正未用过的时段/市场检验，并修复成员时点数据。
