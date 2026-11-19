# Alpha158 默认因子可行性与 GitHub 同类项目调研

**结论：可以把有来源记录的 Alpha158 风格字段作为候选生成器的可选 preset；不能不经准入就设为生产活跃因子。** Qlib 的 Alpha158 是供模型学习的特征集合，不表示每个字段都能单独通过 AlphaCrafter Miner 的严格 IC/ICIR/命中率门槛。当前 18 个固定候选的诊断是 **0/18**，因此没有替换 Miner 的默认候选或启动 Trader。2023 验证集、2024–2026 留出结果此前均已查看，以下仅是探索性结果，不是新样本外成绩。

## GitHub 来源与措施（固定源码版本）

- Microsoft Qlib [v0.8.6 `Alpha158` handler](https://github.com/microsoft/qlib/blob/36950b905dfcae7e2fca321102a30b7bc4f9857e/qlib/contrib/data/handler.py#L140-L194)（该提交日期 2022-06-15，早于本实验最早的 2022-12-30 准入截止）：`get_feature_config()` 包含 K 线、价格和滚动家族；[默认标签](https://github.com/microsoft/qlib/blob/36950b905dfcae7e2fca321102a30b7bc4f9857e/qlib/contrib/data/handler.py#L192-L193) 使用 `Ref($close,-2)/Ref($close,-1)-1`，**与我们 T+1 开盘到收盘标签不一样**。Qlib 的 [Alpha158 LightGBM 工作流](https://github.com/microsoft/qlib/blob/36950b905dfcae7e2fca321102a30b7bc4f9857e/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml)将特征交由模型训练，区分 train/valid/test、限制归一化拟合窗口，单独指定交易成本和 TopkDropoutStrategy；不是把 158 个字段逐一通过 Miner 门槛后直接交易。
- [AlphaGen 的计算器接口](https://github.com/RL-MLDM/alphagen/blob/259687e8f316994426416c530a94842a2fe6405e/alphagen/data/calculator.py)将单因子 IC、因子两两相似度（mutual IC）和组合 IC 分开；[因子池](https://github.com/RL-MLDM/alphagen/blob/259687e8f316994426416c530a94842a2fe6405e/alphagen/models/linear_alpha_pool.py#L55-L104)拒绝高度相似的候选（默认 `mutual_ic > 0.99`），容量满时通过权重优化淘汰冗余因子；[README](https://github.com/RL-MLDM/alphagen/blob/259687e8f316994426416c530a94842a2fe6405e/README.md)明确组合前需处理因子尺度、警告数据来源时效性。这个 GitHub 版本晚于 2022，仅作为**方法论参考**，没有用其后验特征决定本次 2022 候选。

可借鉴的措施：固定来源和预设候选池、训练期限定的归一化、去重复/相关性审计、分离单因子与模型/组合绩效、预设交易执行成本和验证/测试边界。**不要**直接照搬 AlphaGen 的优化或 Qlib 示例回测收益来替代本仓库准入，尤其不能在旧留出集上选择最佳因子。

## 本地特征并非 Qlib 官方的 158 个完全等价字段

`machine-learn/build_features.py` 生成 **149 个 `alpha158_` 前缀特征**（另外还有原有技术字段），是本项目 pandas 的 Alpha158-*style* 实现，并非加载了 Qlib 官方 provider：`vwap` 用 OHLC 典型价近似；`beta/rsqr/resi` 用简化回归代理；`imax/imin/imxd` 以及 `rank` 的实现语义与 Qlib 官方表达式不同；部分 Qlib 量价操作没有复刻。其中所选 18 项已在同源 Qlib provider 的 2022 年六只股票样本上做数值对照（最大绝对误差 <8e-6；见 [`DATA_PROVENANCE_AUDIT.md`](DATA_PROVENANCE_AUDIT.md)），但这不认证完整因子库、官方数据来源或历史成分。若要精确宣称使用官方 Alpha158，仍须接入一致的 OHLCV/VWAP、表达式和因子生成器，并做全字段一致性测试。

本次只从先验的 K 线、动量、均线、波动、价格区间、量价相关、涨跌频率与方向幅度家族选了 **18 个**字段，名单和假设固定在 [`candidates_alpha158_curated_v1.json`](../../quant-method/alphacrafter/candidates_alpha158_curated_v1.json)（SHA-256 `3447cf37ad5c190a01fd7b8f11c17d0ede407383036890d3c4ada119010df715`）。跳过上述明确的代理字段以及近似 VWAP。为避免普通 Miner 特征在 Pearson IC 前自动 RankNorm，本候选用安全 `expression: {"op":"feature", "feature":"..."}` 评价**原始字段值**；不执行任意外部代码。字段只是有公开出处的候选，不代表预先证明了预测能力。原始股票集合和并集交易日历仍未通过 point-in-time 成分/权威日历审计。

## 已执行的严格准入诊断

从已有显式标签日期面板提取这 18 列、`return_1d`（Miner 基线必需）、目标与时间元数据，保留全部 **1,159,161 行**。提取数据位于忽略 Git 的 `machine-learn/data/qlib/alphacrafter_alpha158_curated_v1/features.csv.gz`（SHA-256 `ab0a1ab4db306a32ff4b43753c05d1adafeb937649d4f82c9cc8efabbb89c74f`）；原始面板不修改。使用 `--require-explicit-label-dates`、原始报价日期并集日历、默认执行标签、原严格门槛和**全新空 registry**，独立运行两轮：

| 准入截止 | 训练/验证 | 候选/通过 | 训练 1 日最大方向 ICIR | 验证 1 日最大方向 ICIR | 训练 5 日最大方向 ICIR | 验证 5 日最大方向 ICIR | 验证 1 日最大方向命中率 |
|---|---|---:|---:|---:|---:|---:|---:|
| 2022-12-30 | 2016-01-04–2021-12-30 / 2021-12-31–2022-12-30 | 18 / **0** | 0.066 | 0.044 | 0.121 | 0.102 | 0.542 |
| 2023-12-29 | 2016-01-04–2022-12-30 / 2023-01-03–2023-12-29 | 18 / **0** | 0.042 | 0.175 | 0.092 | 0.185 | 0.570 |

各列最大值可以来自**不同因子**。ICIR 严格阈值：1 日 >0.20、5 日 >0.25；方向命中率 >0.60。两轮均未达到门槛，覆盖率均为 1.0，最大换手约 0.34（阈值 <0.40）。`alpha158_kup2_raw` 在 2023 验证 1 日方向 ICIR 0.175，但训练 1 日 ICIR 仅约 0.042，不能仅因验证较高就使用。

本地产物：`machine-learn/data/qlib/alphacrafter_alpha158_curated_v1/selection_2022/`、`selection_2023/`，含 `candidate_metrics.csv`、`accepted_factors.json`、`temporal_audit.json`。未进行 Screener/Trader 或 2024–2026 收益测试。

## 决策

- **可行**：将本候选文件作为可选、可追溯的 Miner 候选输入，`--candidate-file quant-method/alphacrafter/candidates_alpha158_curated_v1.json`；它适合技术特征诊断和未来不同市场/数据的实验。
- **暂不可行**：把这些字段设为默认**已通过**因子或强行开始交易。Alpha158 能改善模型表达能力不等于每个字段单独符合 AlphaCrafter 严格阈值。
- 已另行建立固定 XGBoost 特征融合的历史探索诊断；2021–2023 验证结果不稳定，**不是 Qlib 官方 Alpha158 基线或交易收益**，见 [`TWO_TRACK_EXPERIMENT.md`](TWO_TRACK_EXPERIMENT.md)。后续仍须先修正/验证官方表达式差异、历史成分与 VWAP 数据，再预注册模型和交易实验，在可信的未使用窗口前推验证。若在已看过的 2024–2026 窗口上调参，必须标为迭代研究，而非全新样本外结果。
