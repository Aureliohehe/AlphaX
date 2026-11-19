# QuantaAlpha CSI300 数据接入前检查

来源：[Hugging Face 数据集](https://huggingface.co/datasets/QuantaAlpha/qlib_csi300)，固定 revision `d63bf5ba30d1d169023110377cbbe93a90a74e07`。已下载完整 `daily_pv.h5` 和调试样本，并从 `cn_data.zip` 用 HTTP Range 读取目录、提取日历和 CSI300 成员区间；未解压全部行情包。

## 实测结构

| 项目 | 完整 HDF5 实测 |
|---|---|
| 实际有行日期 | 2008-12-29–2026-01-09，4,138 个交易日 |
| 行数 | 14,215,449 |
| 实际出现标的 | 5,982 个，包含指数和非 CSI300 股票 |
| 索引 | `datetime`, `instrument` |
| 字段 | `$open`, `$close`, `$high`, `$low`, `$volume`, `$factor`，float32 |
| OHLCV 各列缺失比例 | 约 3.10% |

注意：HDF MultiIndex 的日期 levels 含未使用日期，追溯到 2005 年；不能用 levels 的最小值报告实际行情起点。调试样本仅为 2018–2019 年的 100 个标的，且包含指数，不能当作完整 CSI300。

数据卡称 OHLCV 为复权价量、factor 为复权因子；`$factor` 不是待挖掘的 alpha 标签，也不能直接当作预测特征。复权方向、归一化和历史时点可得性仍需核实，不能再次盲目乘复权因子。文件没有直接提供真实 VWAP、行业、财务或新闻信息。

## 历史股票池与覆盖

Qlib 包中有 `cn_data/instruments/csi300.txt`：15,598 条区间、934 个历史成员代码（整张区间表口径）。筛选应按股票代码与有效起止日，不应把全部 HDF 标的用于 CSI300 横截面。

在 HDF 实际有行的 4,138 个交易日上，按供应区间的首尾日期均包含规则检查：

- 4,135 天有 300 个成员，3 天有 298 个成员。
- 区间预期成员行 1,241,394，HDF 出现成员行 1,240,932，缺少 462 个成员日期对。
- 已出现的成员行中，26,467 行至少一个 OHLCV 字段缺失；不能据此断言停牌，也不应自动填补为可成交价格。
- 该覆盖期内没有整个成员代码完全缺于 HDF；供应日历与实际 HDF 日期集合一致。

这比仅有当前成员快照多了可用的历史区间，但尚未独立验证区间来源、公告和生效时间、退市与代码映射，也未独立认证供应日历。结构检查不等于完整 PIT 认证。

## 对现有流程的影响

1. 需新增中国市场数据适配：按历史区间筛选成员、排除指数、把 `$` 字段映射到基础特征构建接口，保存缺失状态与显式标签结束日。技术特征应在每只股票完整可见历史上预热，之后才筛选准入股票池。
2. Miner 可以在独立实验中评估 close-to-close 的 1/5 日研究标签；它不是可交易收益。不能沿用原美股执行标签与因子库后直接比较。
3. 原 Trader 是次日开盘买入、当天收盘卖出的日内代理。普通 A 股新买入股票受 T+1 约束，不能这样执行；需要另定义至少跨交易日的持仓、标签、费用、停牌与涨跌停处理。规则参考：[上交所交易规则（2026 年修订）3.1.4–3.1.5](https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/universal/c/c_20260424_10816492.shtml)。
4. 本轮只做数据下载和结构审计，未在 CSI300 数据上搜索候选或查看收益。应在搜索前另行冻结训练、准入和留出窗口，保持原美股实验结果独立。

## 本地产物

Git 忽略目录：`machine-learn/data/raw/qlib_csi300_quantaalpha/`。

- `download_manifest.json`：固定版本、原始文件 URL、大小和 SHA-256。
- `daily_pv.h5`：398,130,836 字节，SHA-256 `a357dcc5f641161af9661d9b1605f2f66844de37f8e02b0bc881e87ed16f3e9b`。
- `full_inspection.json`、`debug_inspection.json`：全量与样本结构检查。
- `qlib_csi300.txt`、`qlib_day.txt`、`zip_metadata_inspection.json`：从固定版本 zip 提取的区间、日历及哈希。
- `membership_inspection.json`：成员覆盖和缺失统计；`inspection_scripts/`：本轮检查脚本。

读取 HDF 所需的 tables/h5py 安装在隔离环境 `/tmp/alpha-csi-inspect`，未修改项目原有 Python 环境。
