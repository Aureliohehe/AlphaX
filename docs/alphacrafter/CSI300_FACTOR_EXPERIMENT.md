# CSI300 固定 80 候选：首次因子质量测试

**结果：严格联合准入 0/80；存在局部排序信号，但本轮没有找到可以正式入库的因子。** 测试的是预测质量，没有运行 A 股交易策略或计算可交易收益。

## 冻结设置

- 输入：QuantaAlpha 固定版本的复权 OHLCV，来源与成员覆盖见 [CSI300 数据审计](CSI300_DATA_AUDIT.md)。价格按供应值使用，不重复乘 `$factor`，该字段不参与预测。
- 候选：52 个基础/Alpha158 安全单字段 AST + 上轮美股研究中已冻结的 28 个 LLM 组合，共 80 个；在评估 CSI300 指标前冻结，没有新增模型 API 调用或根据本市场结果修改表达式。
- 训练：2010–2018；准入：2019–2021。方向仅由训练 1 日 Rank IC 决定，跨窗口、跨 horizon 不重新翻转。
- 2022–2025 保留，预定只对准入通过的候选评估。本轮没有通过者，因此未计算留出期因子指标。
- 标签：复权收盘至未来第 1/5 个市场交易日收盘的收益；按每段期末与信息截止分别剔除未兑现标签。这是研究口径，不等于能在收盘后以当天收盘价成交。
- 原严格门槛不变：1 日 IC/Rank IC >0.015、ICIR/Rank ICIR >0.20；5 日对应 >0.025、>0.25；方向命中率 >60%、Coverage >90%、Turnover <40%，训练和准入均须通过。

技术指标先在股票完整历史上预热，再按供应历史成员区间筛选。缺失行情不前向填充，不按未来标签是否存在删除股票；滚动收益显式禁止 pandas 隐式填充。2010–2025 特征表 **1,165,800 行**，与该窗口供应成员区间的预期行数相符。

## 主要结果

80 候选 × 训练/准入 × 1/5 日，共 320 个检查单元：

| 检查 | 未通过数 |
|---|---:|
| Pearson IC | 299/320 |
| Pearson ICIR | 320/320 |
| Rank IC | 184/320 |
| Rank ICIR | 299/320 |
| 方向命中率 | 305/320 |
| Coverage | 104/320 |
| Turnover | 0/320 |

即使单独检查某一个 horizon，也没有候选能同时满足该 horizon 的训练与准入全部条件。不能仅把“1/5 日同时达标”视为本批失败的唯一原因。

### 有研究价值但不合格的例子

以下均使用固定训练方向（负），数值已做方向调整；不是根据留出收益挑选的策略。

| 候选 | 窗口/周期 | Rank IC | Rank ICIR | 方向命中率 | Pearson ICIR |
|---|---|---:|---:|---:|---:|
| `alpha158_ksft2` | 训练 1 日 | 0.02430 | 0.1506 | 58.1% | 0.0483 |
| `alpha158_ksft2` | 准入 1 日 | 0.04507 | 0.3203 | 65.6% | 0.1840 |
| `alpha158_ksft2` | 准入 5 日 | 0.01756 | 0.1299 | 52.8% | 0.0873 |
| `alpha158_corr20` | 准入 5 日 | 0.03597 | 0.2390 | 59.3% | 0.2025 |

`KSFT2 = (2×close-high-low)/(high-low+epsilon)`，训练方向为负，可解释为对当日收盘在区间中位置的短期反转研究。准入 1 日排序指标较强，但 Pearson ICIR 仍不足，训练期强度/命中率及准入 5 日均未达标，不能准入。

`CORR20` 是价格与 log(1+volume) 的 20 日相关，5 日准入 Rank IC 较高，但 ICIR 与命中率不足；训练覆盖率仅约 87.0%，还需核查缺失/停牌对滚动窗口的影响。

预先存在的一日反转基线也不合格：训练/准入 1 日 Rank ICIR 约 0.089/0.191，命中率约 55.9%/58.4%。基线单独诊断，未塞入候选库或用于留出筛选。

## 数据检查与结论边界

另对截至 2021-12-31 已兑现的准入输入标签检查：1 日标签范围约 -20.0% 至 +20.0%，没有绝对值 >50% 的记录；5 日标签范围约 -41.0% 至 +61.5%，没有绝对值 >100% 的记录。该检查没有发现明显超大跳点来解释所有 Pearson/Rank 指标分歧，**不等于价格或公司行动已完成全面认证**，也没有事后截尾、删除极端标签或调整门槛。

本轮支持的结论是：这批价量因子中存在短期排序信息，但还没有满足当前稳定性和质量要求的因子。下一轮可围绕短期位置反转与量价机制做训练内定向研究，并核查滚动缺失；不能直接把局部较好的例子放入正式库。供应成员、日历和复权数据的历史时点可靠性仍未独立认证；表达式是事后提出的，因此结果属于回溯研究。

## 复核与运行

入口：`run_csi300_research.py`；适配：`csi300_data.py`；需要 pandas/numpy/h5py。默认依赖已下载的 CSI300 文件及上一轮本地冻结候选/schema；选择全新输出目录：

```bash
/tmp/alpha-csi-inspect/bin/python quant-method/alphacrafter/run_csi300_research.py \
  --output-dir quant-method/machine-learn/data/qlib/alphacrafter_csi300_new_run
```

本地产物（Git 忽略）：`machine-learn/data/qlib/alphacrafter_csi300_frozen80_v1/`，`status.json=completed_research`。

- `manifest.json`、`environment.json`、`source_snapshot/`：窗口、输入/代码 SHA-256、运行环境与代码副本。
- `candidates_frozen.json`、`holdout_selection_frozen.json`：评估前候选、留出前准入选择；后者为空。
- `findings.json`、`analysis.json`、`report.md`：80 个候选完整指标、逐项失败原因和对照表。
- `baseline_diagnostic.json`、`admission_label_distribution_audit.json`：基线与已兑现标签分布。
- `feature_audit.json`、`universe_row_audit.json`：股票池行数与适配约束。
- `admission/temporal_audit.json`：标签 purge 与信息截止；`holdout_metrics.json` 为空列表。

Miner 暂时生成的因子库、注册表、历史快照与预测文件已在评估后移除，仅保留研究指标和 `RETROSPECTIVE_ONLY.txt`。
