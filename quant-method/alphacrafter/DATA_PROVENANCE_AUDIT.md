# 数据时点障碍：独立交易日历与成分股来源核验

这份报告是**数据可用性审计，不是新策略收益或完整的 point-in-time 认证**。源文件/附件保持只读；独立日历和审计产物另存于本地忽略 Git 的 `machine-learn/data/`。

## 1. 交易日历：独立复核已通过（不等于交易所认证）

从 PyPI 固定 `exchange-calendars==4.13.3`（纯 Python wheel SHA-256 `02b95a738b3848ab9c52529330f84a15f76c765edeb7790cfbbf0d45d92cc2b5`）计算 XNYS 2016-01-04 至 2026-02-27 共 **2,553** 个交易日；与现有原始股票报价日期的并集逐日相同：缺失 0，额外 0。输出 `machine-learn/data/processed/sp500_xnys_xcal_4_13_3.csv`（SHA-256 `065f6e933b9479d03ac390121e415e9e8efd42f015f1497581a1e12314db3acd`），以及同名 `.manifest.json`（版本、包哈希、日期、输入/输出/代码哈希和审计结果）。

`export_xnys_calendar.py` 在依赖版本不符、参考日历缺失或已有输出时失败；完整窗口 `run_pipeline.py --dry-run` 用此日历确认：技术特征预热缺失前 119 日，之后无整日缺失，默认 90 日维护计划 13 轮。**这是第三方交易所日程实现，不是 NYSE 出具的权威时点数据**。早期 `exchange-calendars==4.2.8` 在本地错误地把 **2025-01-09** 当作开市日（该日为美国前总统卡特葬礼临时休市），所以没有沿用旧软件日历或用原始报价并集假充独立来源。真实成交时段、逐股票停牌仍需逐一检查。

复查方法（在提供固定版本依赖的环境中）：

```bash
python3 quant-method/machine-learn/export_xnys_calendar.py \
  --observed-calendar quant-method/machine-learn/data/processed/sp500_features_label_dates.csv.gz.calendar.csv \
  --output quant-method/machine-learn/data/processed/sp500_xnys_new_audit.csv
```

## 2. 历史成分股：发现明确的未来名单污染，尚无法认证

原始股票集合来自 `datasets/s-and-p-500-companies` 的 [2025-08-12 快照](https://github.com/datasets/s-and-p-500-companies/commit/b61fb6f07286691ee86971cab9956182094aca1f)，在 2026-10-31 下载雅虎日线，共 503 个当前快照代码、489 只取得报价。**这不是 2016–2026 逐日的 S&P 500 成分史。** 快照仅列现存于 2025-08 的股票及其 `Date added`，无法恢复之前已被剔除的公司、历史更名或完整退市数据。

使用 `audit_sp500_universe.py` 流式逐行核对：新特征面板 **1,159,161** 行中，**130,418 行（11.25%）早于该股票在当前快照中的 `Date added`**；其中 2022 年 10,422 行，2023 年 7,092 行。原始文件中有 39 个当前快照股票的加入日期晚于 2022-12-30，24 个晚于 2023-12-29。即使排除这些已知提前纳入的行，**仍缺失已退出的历史成分股**，不能据此宣称完成 PIT 修复。独立日历从首次特征日（2016-06-23）到最后特征日（2026-02-27）的整日覆盖也是 0 缺失/0 额外，但这不能证明逐股票 universe 完整。

本地机器可读产物：`machine-learn/data/qlib/alphacrafter_source_audit_2026_11.json`；其 `status` 为 **`not_point_in_time_certified`**、`historical_membership_complete=false`，包含输入/代码哈希与按年统计。审计器刻意**不输出“通过 PIT”**，也不会将快照的加入日期误用作完整历史成员表。

```bash
python3 quant-method/machine-learn/audit_sp500_universe.py \
  --features quant-method/machine-learn/data/processed/sp500_features_label_dates.csv.gz \
  --snapshot quant-method/machine-learn/data/raw/sp500_constituents.csv \
  --calendar quant-method/machine-learn/data/processed/sp500_xnys_xcal_4_13_3.csv \
  --source-manifest quant-method/machine-learn/data/raw/sp500/manifest.json \
  --output quant-method/machine-learn/data/qlib/alphacrafter_source_audit_new.json
```

## 3. Alpha158 与外部数据：仍有无法用现有输入弥补的部分

本地 `alpha158_` 字段 **149 个**，并非官方 Qlib 158 个表达式的逐字段等价实现。已排除显式的回归代理（BETA/RSQR/RESI）、时序极值位置代理（IMAX/IMIN/IMXD）、不等价 RANK 及 `vwap` 近似，不会把 `(high+low+close)/3` 称为真实 VWAP。针对选出的 18 个字段，额外用 `compare_alpha158_qlib.py` 在同一批 Yahoo 调整后数据生成的 Qlib provider 上做**同源表达式抽样对照**：Qlib 0.9.8、2022 年 AAPL/MSFT/JPM/XOM/NVDA/TSLA 共 1,506 个股票日，18 个字段均无缺失状态错位、绝对误差 >1e-4 的行数为 0，最大绝对误差约 7.88e-6。产物 `machine-learn/data/qlib/alphacrafter_alpha158_same_source_comparison.json` 保存字段表达式、输入/代码哈希与逐字段误差。这证明**所选 18 项在该样本内数值对齐**，不能外推至全部 149/158 项，更不能因同源比较就声称价格/成分 PIT。当前没有真实 VWAP、点时基本面/财报/新闻及可验证的历史雅虎复权因子快照；2026 年回取的 `adj_close` 可包含后来才确定的公司行动信息。仅凭日线文件无法反推出每个历史时点的复权数据版本。

**阻断项**：取得可审计的逐日成分股/退出/更名历史（含当时数据源）、退市标的报价及公司行动时点、必要时真实 VWAP 和点时财报/新闻；之后用独立市场日历重建新特征文件，并记录输入清单、各日股票覆盖和标签结束日。若无法取得，应继续把现有 2021–2023 模型/单因子结果标为**受当前名单与历史数据缺失影响的探索性研究**，不得把 2024–2026 旧留出重新称作未使用的新样本外收益。
