# 两条独立研究路线：严格单因子准入 vs Alpha158 风格模型

本轮**没有更改 Miner、Screener 或 Trader 的准入/交易门槛**，没有重跑已查看的 2024–2026 留出收益。2021–2023 的历史结果只作探索性比较；2023 验证已在此前实验中使用，模型参数和特征也不是对未来的盲选。不能把以下指标称为新样本外业绩。

## A. AlphaCrafter 单因子准入（未改）

固定的 18 个 Alpha158 风格候选、数据来源及 GitHub 对照见 [`ALPHA158_PRESET_REVIEW.md`](ALPHA158_PRESET_REVIEW.md)。独立的 2022、2023 准入轮次均为 **0/18**。因子库仍为空，不能为了产生持仓而把模型输出或失败单因子塞进既有 Screener/Trader。继续记录 IC、ICIR、方向命中率的逐项原因；候选扩充和数据/官方表达式校验应作为新的实验记录。

## B. Qlib 思路的特征融合诊断（不是 Qlib 官方工作流）

入口：`machine-learn/run_alpha158_model_diagnostic.py`。从同一固定候选文件提取 18 个 `alpha158_` 风格技术特征；加载后先截断至 2023-12-29，2024+ 的特征/标签不参与训练、标准化、验证或因子选择。逐日横截面百分位转换特征，在**本折训练期**将 `target_open_close` 逐日转成百分位标签，用 XGBoost 固定参数训练：`reg:squarederror`、25 棵树、max_depth=3、eta=0.05、seed=42；**不调参、不 early stopping、不择优轮次**。这种数据处理借鉴 Qlib 多特征建模，但本脚本直接调用 XGBoost，不使用 Qlib 的 `Alpha158` handler、`DatasetH` 或 `XGBModel`，不得称为官方 Qlib 基线。下一交易日 T+1 开盘至收盘的原始收益只用于验证指标，训练/验证末日不成熟的标签分别剔除；预测仍针对所有当日可见特征，不按未来标签筛掉股票。

| 折 | 扩展训练截止 | 验证信号日期 | 已兑现验证日 | 模型平均日 Rank IC | 模型 Rank ICIR | 模型命中率 | 一日反转 Rank IC / Rank ICIR |
|---|---|---|---:|---:|---:|---:|---:|
| 2021 | 2020-12-31 | 2021-01-04–2021-12-31 | 251 | 0.01497 | 0.117 | 53.4% | 0.02757 / 0.128 |
| 2022 | 2021-12-31 | 2022-01-03–2022-12-30 | 250 | -0.01456 | -0.093 | 43.2% | -0.01059 / -0.048 |
| 2023 | 2022-12-30 | 2023-01-03–2023-12-29 | 249 | 0.00478 | 0.042 | 53.8% | 0.01572 / 0.100 |

**结果：并没有稳定胜过简单一日反转，更不能由此推算交易收益。** 2022 折模型预测方向甚至为负。它演示了 Qlib 式“让模型学习多个特征”的可运行路径，**不证明组合因子有效**；没有选择、冻结或交易策略，没有扣费的净值曲线，也不和严格 Miner 的 0% 现金收益作横向收益比较。

成功产物（本地忽略 Git）：`machine-learn/data/qlib/alphacrafter_alpha158_model_v1_audited/`；检查 `status.json` 为 `completed`。目录含输入/代码 SHA-256 的 `manifest.json`、每折模型、预测、逐日 IC、`summary.json`。运行器校验市场日历、显式标签结束日、训练/验证期成熟标签、源文件在运行中未变；如果失败则没有 `completed` 标记，旧目录不能覆盖。已用独立第三方 XNYS 日历确认整日日期覆盖，但历史股票成分/生存者偏差**仍未解决**：现有面板中有 130,418 行早于当前快照 `Date added`；详见 [`DATA_PROVENANCE_AUDIT.md`](DATA_PROVENANCE_AUDIT.md)。真实 VWAP 与历史复权价格版本仍不可用。

```bash
conda run --no-capture-output -n tradingagents python \
  quant-method/machine-learn/run_alpha158_model_diagnostic.py \
  --input quant-method/machine-learn/data/qlib/alphacrafter_alpha158_curated_v1/features.csv.gz \
  --calendar quant-method/machine-learn/data/processed/sp500_features_label_dates.csv.gz.calendar.csv \
  --candidate-file quant-method/alphacrafter/candidates_alpha158_curated_v1.json \
  --output-dir quant-method/machine-learn/data/qlib/alphacrafter_alpha158_model_new_run
```

## 必须先解决的条件

1. 从源数据重建**点时成分股/退市覆盖**与可信交易所日历；核对官方 Alpha158 表达式、本地近似字段、OHLC 调整与真实 VWAP；新闻/基本面需真实发布日期，不能让未来信息进入历史。
2. 在训练窗口内预设候选搜索预算、相似度约束和搜索日志；严格单因子路线与模型路线分别审计，不能拿 Qlib 模型 IC 冒充 Miner `accepted_factors`。
3. 若要报告可交易收益，先定义冻结模型/信号、选股数量、成交时点、费用与借券/保证金模型；不能由 IC 直接推导收益。2024–2026 已被查看，任何重测都须标明为迭代研究；真正新样本外应换用之前未参与选择的、有来源验证的时间或市场数据。
