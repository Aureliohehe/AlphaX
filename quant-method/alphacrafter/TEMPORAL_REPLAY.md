# Miner 时间隔离与 Screener 快照回放

这一步解决“某日到底能看到哪些数据与因子”的工程边界，不代表论文收益复现成功。所有日期均为无时区的交易日日期，信息截止在当日收盘；当日收盘完成准入的库可用于当日收盘生成、次日执行的信号。

## 1. 准入与 Trader 验证使用不同窗口

示例：2016–2021 训练确定方向，2022 年做因子准入，之后才产生 2023 年的 Trader 验证信号。

```bash
python3 quant-method/alphacrafter/miner_lite.py \
  --output-dir quant-method/machine-learn/data/qlib/miner_pit_2022 \
  --as-of 2022-12-30 \
  --train-end 2021-12-31 \
  --validation-start 2022-01-03 \
  --validation-end 2022-12-30 \
  --backtest-start 2023-01-03 \
  --backtest-end 2023-12-29
```

- 先截掉 `date > as_of` 的特征行，再派生缺失的五日标签及每只股票的标签结束日。
- 即使文件的过去行已保存未来收益，也会屏蔽截止日未兑现的值。
- train/validation/backtest 各自要求 `label_end <= min(segment_end, as_of)`，按 1 日和 5 日分别处理。
- `temporal_audit.json` 保存各段 eligible/purged 行数及最后标签结束日。
- 不按照未来标签缺失与否删除特征行。历史指标用于评价；尚未有标签的最新行仍能形成信号。
- 此时 2023 年留出指标应为空，而不是偷偷读取未来文件中的 2023 年标签。

输出：

| 文件 | 用途 |
|---|---|
| `snapshots/2022-12-30.json` | 不可覆盖的因子库状态，包括空库 |
| `factor_library.json` | 最新快照的便捷副本 |
| `registry.json` | 维护历史、准入阈值、分段审计 |
| `predictions.csv` | 仅信息截止日可用的信号；不会回填此前日期 |
| `diagnostic_predictions.csv` | 最终库回算历史信号，仅供诊断 |
| `temporal_audit.json` | 分周期时间边界审计 |

`--snapshot-dir` 可以把多个运行的快照写入同一归档目录。通过 `--registry` 继承上一次**较早或同日**的新版本 registry；未来 registry 和旧版无时间隔离标记的 registry 都会被拒绝。

归档以日期唯一；相同日期不同内容会失败，且不会覆盖原快照。想在同一截止日重新研究不同方案，请使用新的实验目录，不要删除或改写历史文件。不要手工给旧库补版本标记来绕过校验。

## 2. 按历史状态生成信号

```bash
python3 quant-method/alphacrafter/screener_lite.py \
  --library-history quant-method/machine-learn/data/qlib/miner_pit_2022/snapshots \
  --signal-start 2023-01-03 \
  --as-of 2023-12-29 \
  --output-dir quant-method/machine-learn/data/qlib/screener_pit_2023
```

也可使用 `--factor-library factor_library.json`，但单快照在自身 `as_of` 以前不可用，不能还原之前的库。

- 每日选择最近一个不晚于信号日的快照；没有快照就没有候选。
- 空快照表示因子已全部淘汰，不能偷偷回退到上一个非空库。
- 同一个 candidate ID 不允许跨快照改变 AST/定义；新定义必须使用新 ID。
- 新因子可在准入日使用过去特征计算 trailing IC，但不会出现在准入日以前的组合。
- `--signal-start` 只限制输出，保留更早的特征与 IC 历史。
- IC 仍取最近十个信号日；若任一截面存在未知/尚未兑现的标签结束日，则保守跳过该日选择。对于停牌/缺行，这可能减少信号日。
- `regime.csv` 记录每天 `active_factor_count` 与 `selected_factor_count`，可区分“没有可用因子”与“因子未达 suitability 门槛”。

旧快照只允许显式诊断：

```bash
python3 quant-method/alphacrafter/screener_lite.py \
  --factor-library path/to/legacy_factor_library.json \
  --allow-retrospective-library \
  --output-dir path/to/diagnostic_output
```

该模式会明确标记 `library_scope=retrospective_diagnostic`，不允许与 `--library-history` 混用，也不能作为无泄漏证据。

## 3. 已执行的本地真实数据冒烟

未调用外部 LLM，未修改源数据、旧信号缓存或旧库，未调整论文准入门槛。

从本地 `sp500_features.csv.gz` 只抽取必要列并保留截至 2023-01-31 的行，写入：

```text
quant-method/machine-learn/data/qlib/alphacrafter_temporal_smoke/
  features.csv.gz
  candidates.json
  miner/
  screener/
```

抽取列：date、ticker、return_1d、return_5d、volatility_20d、volume_ratio_20d、target_1d、target_5d、target_open_close、target_open_close_5d。保留所有可用股票，不是重新下载或更改行情。

- 面板共 783,854 行、484 只股票、1,663 个交易日。
- 实际特征日期为 2016-06-23 至 2023-01-31，早期技术指标预热导致缺失；不是论文完整市场日历。
- 三个固定候选：return_5d、volatility_20d、volume_ratio_20d 的横截面 rank。
- Miner 截止日 2022-12-30；训练到 2021-12-31，准入验证为 2022 年。
- Miner 可见 774,191 行，2023 年留出指标为 0 行。

运行命令：

```bash
ROOT=quant-method/machine-learn/data/qlib/alphacrafter_temporal_smoke
python3 quant-method/alphacrafter/miner_lite.py \
  --input "$ROOT/features.csv.gz" --candidate-file "$ROOT/candidates.json" \
  --output-dir "$ROOT/miner" --as-of 2022-12-30 \
  --train-end 2021-12-31 \
  --validation-start 2022-01-03 --validation-end 2022-12-30 \
  --backtest-start 2023-01-03 --backtest-end 2023-12-29

python3 quant-method/alphacrafter/screener_lite.py \
  --input "$ROOT/features.csv.gz" --library-history "$ROOT/miner/snapshots" \
  --signal-start 2023-01-03 --as-of 2023-01-31 \
  --output-dir "$ROOT/screener"
```

同日期的冒烟快照已存在；重跑 Miner 请更换输出目录，不能覆盖归档。

| 标签周期 | 训练期末剔除行 | 验证期末剔除行 |
|---|---:|---:|
| 1 日 | 483 | 484 |
| 5 日 | 2,415 | 2,420 |

**结果：3 个候选全部未通过，保存空快照。2023 年 1 月 20 个交易日全部无信号。** 这验证了真实数据下的截止、空库和不回退行为，不是收益评估。尚未扩大搜索，也未声称技术模板能达到论文结果。

## 4. 回归验证与剩余边界

```bash
cd quant-method/alphacrafter
python3 -m unittest discover -v
# 46 tests, OK
```

新增测试验证：

- 修改截止日后的特征、跨训练/验证期末的 1/5 日标签，不改变历史准入和指标（合成例确实有通过因子，非空库恒等测试）。
- 全数据历史回放与提前截断回放产生相同历史信号，包括股票日期不规则的情形。
- 无未来库回填、空快照淘汰、重新准入、不可覆盖、重复日期拒绝、未来 registry 拒绝。
- 旧库必须显式选择诊断模式；输出开始日不会截掉 trailing IC 历史。

后续进展：自动移动窗口、默认 90 天维护、Trader 验证与留出分阶段执行已接入 `run_pipeline.py`，见 [`PIPELINE.md`](PIPELINE.md)。已完成短窗口端到端冒烟，尚未完成完整论文窗口、历史逐日 LLM 编排和跨模型重复实验。新协议不会自动改造旧 `run_miner_walk_forward.py` 或证明旧结果无泄漏。

标签结束日目前从每只股票的特征记录推导，匹配原特征构建的 shift(-h) 语义。旧文件若删除过中间交易日，会保守多剔除；后续需在原始行情处理时直接保存精确结束日。快照协议也不能证明外部候选、原始成分股、新闻发布时间或模型记忆本身没有未来信息。
