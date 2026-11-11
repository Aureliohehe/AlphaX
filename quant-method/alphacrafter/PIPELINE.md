# 自动滚动实验：Miner → Screener → Trader

入口：`run_pipeline.py`。这是确定性、按历史时点执行的实验运行器，不是论文完整多智能体/交易所复现。默认不调用外部模型，也不连接券商。

## 执行顺序

1. **固定计划与参数**：验证日期、输入覆盖和候选配置，保存 `plan.json`、`manifest.json`；记录输入、候选文件及代码 SHA-256。
2. **逐轮构建 Trader 验证期的因子库**：首次准入必须早于 Trader 验证开始。此后每隔至少 90 个日历日，在第一个可用交易日维护一次。
3. **移动评估窗口**：训练从固定起点扩展；因子准入使用截止日之前最近 252 个可见交易日。两段不重叠，1/5 日标签仍各自做期末隔离。历史不足直接失败。
4. **Screener 回放 Trader 验证期**：只使用信号日已激活的快照和已兑现的 IC 标签。
5. **冻结 Trader 配置**：最多三次验证，按论文可行域选最高 Sharpe；没有可行配置则冻结为现金。写一次 `trader/selection.json`。
6. **冻结以后才处理留出期**：按预先固定的规则继续维护因子库，生成留出信号；只读取冻结的 Trader 配置评估一次，不再搜索配置。

这里的“独立”是时间上的可用性隔离，不是禁止在线学习：后续因子维护可以使用验证期/留出期中**已经兑现**的历史数据，但不能利用当时尚未发生的收益。初始因子准入严格早于 Trader 验证；Trader 参数选择结束后不再使用留出表现调参。修改运行规则、候选或阈值后重跑，应视为新实验，而不是继续称同一个留出集为完全未使用。

## 运行方式

先检查计划，不生成任何输出文件：

```bash
python3 quant-method/alphacrafter/run_pipeline.py \
  --input quant-method/machine-learn/data/processed/sp500_features.csv.gz \
  --output-dir quant-method/machine-learn/data/qlib/alphacrafter_pipeline_pit \
  --dry-run
```

确认窗口、数据覆盖和计算预算后，去掉 `--dry-run` 执行。

默认：

| 参数 | 默认值 |
|---|---|
| `--initial-as-of` | 2022-12-30 |
| `--train-start` | 2016-01-04 |
| `--admission-sessions` | 252 个交易日 |
| `--min-train-sessions` | 252 个交易日 |
| `--maintenance-days` | 90 个日历日 |
| `--selection-start/end` | 2023-01-03 / 2023-12-29 |
| `--holdout-start/end` | 2024-01-02 / 2026-02-27 |
| `--configs` | β=0.8；γ=0.5、0、1，分别 50/25、50/50、50/0 个多/空头 |
| `--cost-rate` | 每腿成交额 1bp |
| `--annual-risk-free-rate` | 3.81% |

注意：当前旧特征缓存未覆盖到完整的 2026-02-27，默认完整窗口会被拒绝。应先重建到新文件并核查日期，或显式缩短 `--holdout-end` 并说明窗口差异，运行器不会默默缩短日期冒充论文实验。

候选可以是确定性模板（通过 `--max-pairs`、`--max-temporal` 限制数量），也可以使用 `--candidate-file` 固定 JSON/AST。外部候选需要调用者确保其假设在实验开始前已确定；文件哈希只能证明内容不变，不能证明来源无未来信息。当前运行器不接入历史 LLM 调用，也不提供放宽 Miner/Trader 门槛的开关。

每个子进程可用 `--step-timeout SECONDS` 限制时间。大文件会在每轮重新读取和计算，先用明确标记的低成本子实验验证流程，再安排完整运行。

## 产物与失败行为

```text
output/
  plan.json                  # 日期与移动窗口计划
  manifest.json              # 参数、输入/候选/代码哈希、环境版本
  events.jsonl               # 子进程命令、完成事件、冻结顺序、错误
  logs/                      # 每个 Miner/Screener 子进程的 stdout/stderr
  miner/YYYY-MM-DD/          # 每轮独立结果与 registry
  snapshots/YYYY-MM-DD.json  # 不可覆盖的快照
  screener_selection/        # Trader 验证期信号
  screener_holdout/          # 留出期信号
  trader/
    selection_trial_N.csv    # 最多三个验证组合的逐日账本
    selection.json           # 留出生成前冻结的选择，含信号/快照哈希
    holdout_returns.csv
    holdout_positions.csv
  rounds.csv                 # 各轮训练/准入窗口、通过/活跃/淘汰数
  summary.json               # 留出结果与限制
  status.json                # completed 或 failed
```

- 每轮复制上一轮 registry 到**新目录**再让 Miner 读写，不修改历史轮次的 registry。
- 计划、manifest、旧快照、旧 registry、已冻结的选择和信号会进行哈希核验；输入和代码也会在关键阶段重新核验。快照目录还必须与已完成计划严格一致，不能插入额外快照或在验证阶段使用未来快照。
- 输出目录必须新建或为空；已完成或失败的运行都不允许直接覆盖。
- 不自动恢复失败运行。失败保留日志、计划、已有快照，`status.json` 记下失败阶段；后续阶段不执行。修复后使用新的实验目录。
- 成功状态在结果与事件日志写完后才落盘。消费结果时必须确认 `status.json` 为 `completed`；磁盘/日志故障时可能无法写状态，但不能仅凭部分产物认定成功。
- `--dry-run` 只检查计划及基本配置，不替代真实的候选 AST 校验、数据质量检查或收益评估。

## Trader 执行边界

沿用当前 Trader 的**信号日区间**约定：T 收盘生成信号，T+1 开盘建仓、收盘平仓；期末只评估执行日不晚于窗口结束的信号。因此窗口最后一天通常没有已兑现的下一日收益，账本会少于窗口内信号日期数，不能直接与论文交易日数量比较。

本轮额外修正：

- 空因子库的 header-only 信号 CSV 会保持数值类型，得到现金组合，而不是因 `np.isfinite` 遇到 object 列崩溃。
- `load_inputs(..., as_of=...)` 在建立执行日映射之前截断数据。
- 某只股票“下一条记录”晚于市场下一交易日时，不能把更晚的收益算到次日。该股票的标签会置为不可用；若它被当前信号选中，交易模拟报错而不换股或假定零收益。

这仍是日内代理，没有持续股数持仓、隔夜收益、借券、保证金和完整成交模型。

## 已执行：真实数据三轮短窗口冒烟

使用上一阶段的本地抽取面板：484 只股票、783,854 行；三个预先固定候选为 return_5d、volatility_20d、volume_ratio_20d。没有调用 LLM，没有修改源缓存，也没有放宽 IC、ICIR、Hit Ratio、Coverage、Turnover 或 Trader 可行域门槛。

```bash
ROOT=quant-method/machine-learn/data/qlib
python3 quant-method/alphacrafter/run_pipeline.py \
  --input "$ROOT/alphacrafter_temporal_smoke/features.csv.gz" \
  --candidate-file "$ROOT/alphacrafter_temporal_smoke/candidates.json" \
  --output-dir "$ROOT/alphacrafter_pipeline_smoke" \
  --initial-as-of 2022-12-30 \
  --selection-start 2023-01-03 --selection-end 2023-01-17 \
  --holdout-start 2023-01-18 --holdout-end 2023-01-31 \
  --maintenance-days 14
```

14 日仅用于短窗口测试维护链路，不是论文默认 90 日实验。该输出目录已存在，重跑请换目录。

| 轮次 | 阶段 | 训练截止 | 因子准入区间 | 活跃因子 |
|---|---|---|---|---:|
| 2022-12-30 | Trader 验证之前 | 2021-12-30 | 2021-12-31–2022-12-30 | 0 |
| 2023-01-13 | Trader 验证中 | 2022-01-12 | 2022-01-13–2023-01-13 | 0 |
| 2023-01-27 | 留出期间 | 2022-01-26 | 2022-01-27–2023-01-27 | 0 |

`events.jsonl` 确认：前两轮 → 验证期 Screener → `trader_frozen` → 第三轮 → 留出期 Screener。

**结果：无可行策略，冻结为空仓。** 验证及留出各有 9 个已兑现信号日；留出收益 0、回撤 0、Sharpe 未定义。这只证明流程在真实数据上能完成滚动维护和拒绝失败策略，不代表收益复现成功。

## 回归测试与下一步

```bash
cd quant-method/alphacrafter
python3 -m unittest discover -v
# 58 tests, OK
```

新增测试覆盖维护日历、移动窗口、窗口冲突、数据不足、dry-run 无副作用、失败停止、旧产物不被覆盖、空库现金、延迟标签；合成数据中确实有因子和策略通过门槛，并验证**修改留出结果不会改变冻结选择文件的任何字节**，同时允许留出收益随之变化。

下一优先级：保存精确的原始标签结束日及交易日历，补持续持仓/保证金执行模型；然后接入有预算和可重放缓存的历史 LLM 候选生成与筛选，再开展完整 90 日维护、跨模型和重复试验。当前确定性模板冒烟结果不能代替这些实验。
