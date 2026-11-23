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

注意：旧特征缓存未覆盖到完整的 2026-02-27。已另建带原始标签结束日的本地文件 `machine-learn/data/processed/sp500_features_label_dates.csv.gz`（不覆盖旧缓存），并输出同名 `.calendar.csv` 和 `.manifest.json`。该文件覆盖到 2026-02-27；预热期前 119 个原始交易日没有完整技术特征。完整窗口严格预检可使用：

```bash
python3 quant-method/alphacrafter/run_pipeline.py \
  --input quant-method/machine-learn/data/processed/sp500_features_label_dates.csv.gz \
  --calendar quant-method/machine-learn/data/processed/sp500_features_label_dates.csv.gz.calendar.csv \
  --require-explicit-label-dates --allow-feature-warmup-sessions 120 \
  --output-dir quant-method/machine-learn/data/qlib/alphacrafter_pipeline_label_dates --dry-run
```

实测输出 13 轮计划、实际预热缺失 119 天，预热之后无缺失的全市场交易日；对比报价并集日历检查 1,159,161 行特征的 1/5 日标签结束日，均无错位。预热例外必须显式设置，只允许连续的前缀且数量不超过限制；中间缺日仍失败。此日历是原始报价日期的并集，**未经交易所权威验证**；预检仅证明日期覆盖，尚未运行完整流程，也未认证历史成分股或数据发布时点。

候选可以是确定性模板（通过 `--max-pairs`、`--max-temporal` 限制数量），也可以使用 `--candidate-file` 固定 JSON/AST。外部候选需要调用者确保其假设在实验开始前已确定；文件哈希只能证明内容不变，不能证明来源无未来信息。当前运行器不接入历史 LLM 调用，也不提供放宽 Miner/Trader 门槛的开关。

每个子进程可用 `--step-timeout SECONDS` 限制时间。大文件会在每轮重新读取和计算，先用明确标记的低成本子实验验证流程，再安排完整运行。

## 产物与失败行为

```text
output/
  plan.json                  # 日期与移动窗口计划
  manifest.json              # 运行 ID、实验条件指纹、配置、输入/代码及依赖版本
  checkpoints/               # 初始化清单、两期 Miner 轮次、Screener 和留出期开始标记
  attempts/                  # 显式续跑时归档的上次状态、产物索引和未提交文件
  artifacts.json             # 归档产物列表，路径相对于本次运行目录
  events.jsonl               # 子进程命令、完成事件、冻结顺序、错误
  logs/                      # 每个 Miner/Screener 子进程的 stdout/stderr
  miner/YYYY-MM-DD/          # 每轮独立结果与 registry
  snapshots/YYYY-MM-DD.json  # 不可覆盖的快照
  screener_selection/        # Trader 验证期信号和 checkpoint.json
  screener_holdout/          # 留出期信号和 checkpoint.json
  holdout_evaluation/        # 一次提交的留出期收益、持仓、轮次、摘要和 checkpoint.json
  trader/
    selection_trial_N.csv    # 最多三个验证组合的逐日账本
    selection.json           # 留出生成前冻结的选择，含信号/快照哈希
    checkpoint.json          # 冻结决定、试验账本及上游输入指纹
    holdout_returns.csv
    holdout_positions.csv
  rounds.csv                 # 各轮训练/准入窗口、通过/活跃/淘汰数
  summary.json               # 留出结果与限制
  status.json                # completed 或 failed
```

- 每轮复制上一轮 registry 到**新目录**再让 Miner 读写，不修改历史轮次的 registry。
- 计划、manifest、旧快照、旧 registry、已冻结的选择和信号会进行哈希核验；输入和代码也会在关键阶段重新核验。快照目录还必须与已完成计划严格一致，不能插入额外快照或在验证阶段使用未来快照。
- 新运行使用新建或空目录。失败保留日志、计划和已有快照，`status.json` 记下失败阶段。
- 成功状态在结果、产物清单与事件日志写完后才落盘。失败记录会尽可能保存当时已有的产物。读取收益时以 `status.json` 的 `completed` 为准。
- `--dry-run` 只检查计划及基本配置，不替代真实的候选 AST 校验、数据质量检查或收益评估。

## 续跑流水线

如果进程中断或失败，使用原命令、原输出目录和原参数，追加 `--resume`。续跑保留原运行 ID，要求相同版本的 `checkpoints/run.json`。程序会校验计划、配置、代码、依赖版本、输入和已提交产物的指纹。

- Miner 先在暂存目录执行。核验快照、注册表和标签边界后，再发布产物和轮次检查点。续跑只跳过已有检查点且指纹仍一致的轮次。
- 两期 Screener 的预测、注册表和检查点分别一同发布。Trader 冻结的试验账本与决定也一同发布。留出期开始标记绑定冻结检查点，恢复评估时继续使用这份策略，包括空仓决定。
- 留出期的收益、持仓、轮次和摘要先作为 `holdout_evaluation/` 一组提交，再发布到原有产物路径。若中断发生在评估提交后，恢复时只核验结果并补齐原有产物路径、索引和完成状态。
- 未提交的轮次或阶段移到 `attempts/` 下并重新执行。过去的日志继续保留，本次日志写到 `logs/resume-*/`。
- 运行目录需要支持 POSIX 文件锁和硬链接。子进程仍在使用目录时，另一进程会收到锁冲突。
- 已完成的运行可直接查阅和归档。要采用不同的代码、配置或数据运行，使用新目录。

## 查找和归档实验

```bash
python quant-method/alphacrafter/list_experiments.py quant-method/machine-learn/data/qlib
```

该命令以 JSON 列出指定目录中的主流水线、CSI300 因子研究和中低频研究。可以传入单个运行目录，或扫描包含多次运行的上级目录。每条记录的 `directory` 指向当前运行目录，`artifacts` 中的文件路径相对于它。产物索引和最终状态的位置由 `manifest.json` 的 `records` 字段给出。移动归档目录后可以继续查询。旧版运行通过已有的计划、状态、清单和摘要读取，缺失字段显示为 `null`，缺少有效状态的运行显示为 `unknown`。

- `run_id` 标识一次执行。即使条件相同，重跑也会产生新的 ID。
- `experiment_fingerprint` 标识规范化配置、输入内容、参与执行的源码和运行环境版本的组合。输入按用途记录，路径和输出目录不参与计算。Git 提交号用于追溯，实际执行的源码通过文件哈希识别，包括工作区中的修改。
- `experiment_type` 标识实验类型。主流水线展示组合收益和风险指标，CSI300 因子研究展示候选数、严格准入数和留出评估状态。中低频研究展示 Miner 来源、运行模式、分频入池和策略状态。
- `data_version` 在主流水线中是特征面板的 SHA-256，在 CSI300 与中低频中是原始行情、成分表和日历指纹的组合。中低频的可选特征面板也参与实验条件指纹。日志和清单保留运行当时的来源路径。

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

## 已执行：完整窗口固定 3 候选收益诊断

使用新建的显式标签日期特征缓存与报价并集日历，固定先前冒烟实验的 `return_5d`、`volatility_20d`、`volume_ratio_20d` 三个候选；默认 90 日维护、严格 Miner/Trader 门槛，完整输出在 `machine-learn/data/qlib/alphacrafter_pipeline_fixed3_full/`（`status.json` 为 `completed`，本地未纳入 Git）。除 `--allow-feature-warmup-sessions 120` 和固定候选文件外，窗口与交易成本使用运行器默认值。

**13 轮因子准入均为 0，验证期 249 个执行日、留出期 540 个执行日均无信号。** 三个 Trader 配置验证收益均为 0、Sharpe 未定义，没有可行配置，故留出期全现金：累计/年化收益 **0%**、最大回撤 **0%**、Sharpe **未定义**、实际投资天数 0。同一日内收益窗口、**未扣交易费用**的逐日有效报价股票等权参考值累计收益 **9.82%**，相对此参考值的差额 **-9.82 个百分点**。这是完整流程的拒绝结果，不是论文策略收益，也不能通过看完留出结果后改候选/门槛仍声称同一留出集未使用。只使用 Trader 选择期指标的逐项失败原因见 [`GATE_DIAGNOSIS.md`](GATE_DIAGNOSIS.md)。

## 回归测试与下一步

```bash
cd quant-method/alphacrafter
python3 -m unittest discover -v
# 58 tests, OK
```

新增测试覆盖维护日历、移动窗口、窗口冲突、数据不足、dry-run 无副作用、失败停止、旧产物不被覆盖、空库现金、延迟标签；合成数据中确实有因子和策略通过门槛，并验证**修改留出结果不会改变冻结选择文件的任何字节**，同时允许留出收益随之变化。

下一优先级：保存精确的原始标签结束日及交易日历，补持续持仓/保证金执行模型；然后接入有预算和可重放缓存的历史 LLM 候选生成与筛选，再开展完整 90 日维护、跨模型和重复试验。当前确定性模板冒烟结果不能代替这些实验。
