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

## 4. 后续数据源可行性排查（尚未修复）

本地 `data/raw/` 只有 2025-08 成分快照、按该快照下载的 Yahoo 股票报价和 manifest；没有历史加入/退出事件表或退出股票报价。现有 `audit_sp500_universe.py` 只能发现矛盾，不能从 `Date added` 还原历史股票池。排查了 [Qlib 美股指数采集器](https://github.com/microsoft/qlib/blob/main/scripts/data_collector/us_index/collector.py)：其 `SP500Index.get_changes()` 从维基百科网页的变动表解析事件，再由采集器重建名单；网页是可变的第三方记录，不能单凭这个脚本认定完整、权威的 PIT 成分史。此次检查当前维基百科页面时，`pandas.read_html` 只读到当前成分表和导航表，没有读到脚本预期的变动表；直接照搬该采集脚本会失败。即使取得变动事件，也还需要与**同一时点**的锚点名单、代码变更映射以及退市股票报价交叉校验，不能将当前名单或缺失报价当作无风险剔除。

### 已下载的探索性历史成分表（不含报价）

从 [fja05680/sp500](https://github.com/fja05680/sp500) 固定提交 `a2430f2af0c79ddf0748e91de11bdeb1616ab5a7` 下载历史成分 CSV、变动表、README 和 LICENSE 至 Git 忽略的 `machine-learn/data/raw/sp500_historical_research/`，保留 URL、下载时间、大小及 SHA-256 于 `download_manifest.json`；未修改现有原始数据。主 CSV SHA-256 为 `36326709d46d6cd25834de5df457b16f5f96fad3a06b9beac28f7b88aa0b0d54`，记录 1996-01-02 至 2026-08-18 的 **2,720 个变动/快照日期**，不能误读为每日均有一条。2016–2026-02-27 区间出现约 730 个独立代码（按源代码字符串去重）；用变动日之前最新一条名单查询，2022-12-30 有 503 只、2023-12-29 有 503 只。在 2025-08-12 与本地 2025-08 成分快照相比，两组 503 个代码完全相同；**同源/当前名单相符不是历史准确性证明**。项目 README 明确说明 1996–2019 原始数据来自书籍配套下载、2019 年后靠人工核对维基百科变更，维基百科变更表不完整，且退市报价通常需要购买。这是事后重建名单，来源及当时可见性尚未独立认证，不能称作已认证 PIT。

另外取得该项目 2022-12-30 之前的固定提交 `22aeb7a542c7c69568f81f0bac637ff217b3d466` 的历史成分表，单独存入 `sp500_historical_research/asof_2022_12_30/` 并记录 manifest（CSV SHA-256 `c62b7f2344626fa0a45ab79c7c5a40aca9d16b029eee2e44b198ab9683d8738a`）。用 2016-01-04–2022-12-29 **工作日**做向前取最近快照的两版本比对，1,824 天中有 8 天的名单不同：2022-01-10 至 2022-01-19（含一个非交易工作日），旧版本是 `WLTW`、新版本是 `WTW`；到 2022-01-20 一致。说明新版确有事后代码修正；工作日并非交易日，该比对仅用于识别数据版本变化，不能代替交易所日历或历史 PIT 认证。

仅按代码映射本地 `data/raw/sp500/*.csv` 的**文件是否存在**（不是逐日价格覆盖）：这 730 个代码中有 241 个缺本地报价文件；在 2016-06-23 / 2022-12-30 / 2023-12-29 / 2026-02-27 的历史名单中，分别有 166 / 58 / 40 / 21 个代码缺文件。部分是仍可从 Yahoo 补取的旧指数成员，部分已退市或更名；抽查 Yahoo Chart：`AAL` 有历史数据，而 `ATVI`、`AET`、`ABC` 返回 404，不能把缺文件当作无价格、零收益或自动剔除后继续宣称完整回测。必须进一步核实代码映射、交易日逐日价格覆盖、退市收益和公司行动。

### 免费报价补取：不能直接用于回测

运行 `machine-learn/download_historical_member_prices.py`，对上述 730 个历史代码里本地缺报价的 **241 个**逐个查询 Yahoo Chart（2016-01-01–2026-02-27）。单独存放在 `sp500_historical_research/yahoo_supplement/`，两个 `run_*.json` 记录源、输入哈希、状态及成功文件哈希；原始 `data/raw/sp500/` 未被覆盖。结果：**102** 个代码拿到 CSV、**123** 个无数据/404、**16** 个 HTTP 400 等失败。Yahoo 代码会复用，拿到 CSV 不代表拿到了历史公司的价格：在独立 XNYS 日历上运行 `machine-learn/audit_yahoo_supplement.py`，`supplement_audit.json` 显示 **14/102 个报价文件与该代码的历史指数成员交易日零重叠**，包括 `FB`、`EMC`、`POM`、`STI`、`SE` 等，应视为不适用于对应历史成员；其余 88 个有交叠仍**未验证公司身份及逐日完整性**（如 `SNDK` 仅 62/152 个成员交易日有报价，旧/新公司代码复用风险不能由交叠排除）。不能将这 102 个文件直接并入特征构建。

重复补取（默认跳过已下载文件和相同版本/日期的确定性无数据响应，限流或暂时失败可重试）：

```bash
python3 quant-method/machine-learn/download_historical_member_prices.py \
  --membership 'quant-method/machine-learn/data/raw/sp500_historical_research/S&P 500 Historical Components & Changes (Updated).csv' \
  --existing-quotes quant-method/machine-learn/data/raw/sp500 \
  --output-dir quant-method/machine-learn/data/raw/sp500_historical_research/yahoo_supplement
```

下一步数据接入的最低要求：供应源注明历史成分的有效开始/结束日、退出及更名/代码变动记录和可追溯版本；提供相应的退市股票日线与公司行动时点；按交易日验证成员数、成员报价覆盖率、未知代码和缺口，再独立输出新特征文件及输入哈希。未满足前不覆盖旧文件、不把加入日期过滤的子集标作 PIT，也不在旧留出集上宣称新样本外因子收益。
