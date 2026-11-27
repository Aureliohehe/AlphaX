# AlphaX 接口与运行开关清单

本地 Web 工作台有 8 个页面，提供系统设置、自然语言挖掘、因子检索和基于固定候选的全周期回测。项目另有 46 个研究与数据脚本入口，其中 42 个声明了 CLI 参数。数据准备、逐阶段诊断及多数专项实验通过 CLI 完成。

## Web 页面与 HTTP 接口

| 页面 | 页面上可用的能力 |
| --- | --- |
| 研究总览 `#overview` | 查看沪深 300 冻结四因子净值与摘要，切换日期及各条曲线，查看预选的周频 XGBoost 配对案例 |
| 因子挖掘 `#miner` | 填写研究假设并启动 Miner，查看阶段和日志，打开本次任务生成的因子定义 |
| 因子库 `#factors` | 按关键词、实验来源和准入状态筛选本机 Miner 产物 |
| 独立回测 `#backtest` | 选定因子来源和质量范围，运行完整美股主线，查看历史回测的留出期指标、净值和冻结策略 |
| Agent loop `#workflow` | 查看工作流说明、历史搜索逐轮统计及机制优先实验的分频候选计数 |
| 实验对照 `#results` | 查看指定因子的基线与增强对照、冻结检验及冻结四因子两档成本指标 |
| 运行中心 `#runs` | 启动离线示例、冒烟测试和固定 S&P 500 预检，查看日志及复制三条研究命令 |
| 系统设置 `#settings` | 配置模型、API 地址、密钥、数据路径、实验窗口、搜索预算及策略参数 |

| HTTP 操作 | 请求与返回 |
| --- | --- |
| `GET /api/dashboard` | 返回固定历史实验的 `holdout`、`paired`、`search`、`mechanism` |
| `GET /api/jobs` | 列出会话中的任务、进度、命令、日志及退出码 |
| `POST /api/jobs` | 接收同源 JSON。固定任务的 `kind` 为 `demo`、`smoke` 或 `preflight`。挖掘使用 `miner` 和 `brief`，回测使用 `backtest`、`library` 和可选 `grade`。成功返回 HTTP 201 和任务 ID，已有运行中任务时返回 HTTP 409 |
| `GET /api/settings` · `POST /api/settings` | 读取或保存配置。保存请求使用 `values` 和可选 `api_key`，响应只返回配置及密钥存在状态 |
| `GET /api/factors` | 返回可用因子库及候选定义的展示字段 |
| `GET /api/backtests` · `GET /api/backtest?id=…` | 列出已完成回测，或读取指定回测的留出期指标及净值 |

服务入口为 `python report/web/server.py`，`--port` 默认 8765，监听 `127.0.0.1`。同一时刻执行一个任务，内存保存最近 20 个任务，每任务最多 300 行子进程日志。密钥仅保存在当前服务进程内；其余设置和研究产物保存在 `report/web/.runs/`。[工作台说明](report/web/README.md) · [HTTP 后端](report/web/server.py)

## 能力与 Web 覆盖

| 能力 | 实际入口与产物 | Web 覆盖 |
| --- | --- | --- |
| 实验目录、运行状态与复现信息 | `list_experiments.py`、`alphax.experiment.list_runs`、`manifest.json`、`status.json` | 本次服务任务日志、可检索因子和已完成 Web 回测。通用实验产物浏览在本地文件中 |
| 数据准备与审计 | 行情下载、历史成分补数、独立日历、特征和标签、成分覆盖与补数审计 | 页面配置输入路径，下载、构建、审计在 CLI |
| 美股主流水线 | `run_pipeline.py`。滚动准入与因子快照、验证期信号、策略冻结、留出期收益和检查点 | 可配置的固定候选全周期回测，另有预检与合成示例。续跑在 CLI |
| Miner | `miner_lite.py`。模板、固定候选、LangGraph 提案、训练折搜索、机制优先、研究记忆和外部 skill | 可输入研究假设并设置模型、搜索预算和部分窗口参数；专项机制、研究记忆、外部 skill 与门槛覆盖在 CLI |
| Screener 与 Trader | `screener_lite.py`、`trader_lite.py`。按日回放因子库，合成股票评分，冻结策略并评估留出期 | 由独立回测调用，逐阶段独立运行及诊断在 CLI |
| 沪深 300 与周频研究 | 四因子、分频候选池、日周月策略、配对增强、归因和冻结留出 | 指定实验的历史展示和研究命令；运行专项实验使用 CLI |
| Qlib 与 XGBoost 基线 | 模型训练、组合回测、前推、消融、Alpha158 对照与因子收益排序 | 独立实验在 CLI |

### 会改变研究口径的开关

| 入口 | 参数语义与组合约束 |
| --- | --- |
| `run_pipeline.py` | 默认用模板，也可用 `--candidate-file` 固定候选。日期参数分别控制首轮信息截点、训练、验证和留出期。维护间隔默认 90 个自然日，准入窗口与最少训练期默认各 252 个交易日。`--dry-run` 预检，`--resume` 复用已提交检查点，两者互斥 |
| `miner_lite.py` | 默认无外部模型。`--llm-model` 启用 LangGraph 提案，Agent 默认 3 个。`--research-brief-file` 追加 UTF-8 研究假设。`--search-rounds` 默认 1，超过 1 时要求模型且排除固定候选，训练折默认 3，最少 2 折。`--llm-mechanism-first` 先冻结经济机制，`--llm-seed-file` 和外部 skill 各自要求单轮 LLM 模式。配对反馈要求种子模式或 `--llm-xgboost-complement`。SQLite 记忆要求市场为 `us` 或 `cn` 并指定 `--memory-universe` |
| `miner_lite.py` 评估 | `--as-of` 控制信息截点，`--label-mode` 默认 `execution`，也可选 `paper_close` 做论文口径诊断。覆盖率下限默认 0.90，换手上限默认 0.40，训练与准入期的 IC、ICIR、命中率门槛可分别覆盖。`--factor-return-model` 要求固定候选和截点，走独立因子排序入口 |
| `screener_lite.py` | `--factor-library` 与 `--library-history` 互斥，后者按历史信号日回放。仅诊断用的 `--allow-retrospective-library` 可回填静态库。默认回看 10 天，最低适用度 0.02，相似性阈值 0.80，最多 20 个因子。三个 `--llm-*-model` 开关只用于最新日期 |
| `trader_lite.py` | 验证期与回测期独立设置。最多 3 组 `beta,gamma,nlong,nshort` 配置，成本率默认 0.0001，每次入场与出场均收费。可用研究池约束策略验证时点并选择 1 日或 5 日研究周期 |
| `run_medium_frequency.py` | 默认执行单因子研究。频率强度规则默认 `all`，可切换 `any`。两种周频 XGBoost 比较均依赖 `--evaluate-strategies` 及 XGBoost。候选加法比较支持 `refit`、`factor_rank`、`residual_rank`，权重在 (0, 1] 内，默认 1。该入口读取既有沪深 300 本地数据和已完成 Miner 产物 |
| `run_frozen_positive_holdout.py` | 行情和策略约定由实验脚本固定。`--resume-prefix-order-failure` 仅恢复已知的前缀序列化断言失败，恢复前校验冻结输入和收益文件状态 |
| 数据准备和 Qlib | `build_features.py --overwrite` 显式替换派生产物与旁路文件，`--calendar` 改用独立会话日历，`--fundamentals` 与 `--news` 加入可选输入。`qlib_portfolio_backtest.py --rebuild-provider` 重建 provider，持仓数量、替换数量、资金比例和各项交易成本分别配置 |

### 进程环境与 Python 接口

- `DEEPSEEK_API_KEY` 由内置 DeepSeek 客户端读取，`DEEPSEEK_API_BASE` 可覆盖服务地址，默认 `https://api.deepseek.com`。显式设置 `ALPHAX_LLM_API_BASE` 和 `ALPHAX_LLM_API_KEY` 可接入兼容 Chat Completions 的模型服务。其他 Miner 模型交由 LiteLLM，凭据由对应提供商配置，例如 Anthropic 的 `ANTHROPIC_API_KEY`。`make test-llm` 设置 `LITELLM_LOCAL_MODEL_COST_MAP=True` 与 `LANGSMITH_TRACING=false`。工作台子进程继承启动环境，额外设置 `PYTHONUNBUFFERED=1` 和 `PYTHONDONTWRITEBYTECODE=1`。依赖组包括核心环境、`llm`、`qlib`、`dev`。[依赖说明](requirements/README.md) · [Miner 使用说明](quant-method/alphacrafter/README.md)
- 已打包的 `alphax` 提供因子 DSL（`Candidate`、`parse_expression`、`validate_candidate`、`evaluate_expression`）、实验清单与发现（`build_manifest`、`list_runs`）、检查点（`lock_run`、`publish_json`）和 SQLite 记忆（`MinerMemory.record`、`recall`）。仓库内可复用的研究函数还包括 `run_pipeline.build_plan`、`miner_search.run_search`、`screener_lite.select_ensemble`、`trader_lite.simulate`、`trader_lite.select_configuration`、`cn_portfolio.simulate_cn`。这些是 Python 调用点，工作台的 HTTP 接口见上表。[共享包](quant-method/alphacrafter/alphax/) · [研究代码](quant-method/alphacrafter/)
- Makefile 入口为 `test`、`test-alphacrafter`、`test-machine-learn`、`test-smoke`、`lint`、`test-llm`、`test-qlib`，可用 `PYTHON=...` 指定解释器。[Makefile](Makefile)

## CLI 参数索引

参数索引覆盖现有研究与数据入口。带参数的入口使用 `--help` 查看类型、说明和用法。标注「参数写在脚本中」的实验没有 CLI 参数解析，直接调用脚本会启动实验。参数必填性、默认值和校验以链接中的解析器为准。

### 主流程与实验管理

| 入口 | 能力 | 参数 |
| --- | --- | --- |
| [`run_pipeline.py`](quant-method/alphacrafter/run_pipeline.py) | 滚动 Miner → Screener → Trader | `--input`、`--output-dir`、`--candidate-file`、`--calendar`、`--allow-feature-warmup-sessions`、`--require-explicit-label-dates`、`--initial-as-of`、`--train-start`、`--selection-start`、`--selection-end`、`--holdout-start`、`--holdout-end`、`--maintenance-days`、`--admission-sessions`、`--min-train-sessions`、`--max-pairs`、`--max-temporal`、`--configs`、`--cost-rate`、`--annual-risk-free-rate`、`--initial-cash`、`--step-timeout`、`--dry-run`、`--resume` |
| [`run_offline_example.py`](quant-method/alphacrafter/run_offline_example.py) | 使用合成数据完成全链路 | `--output-dir` |
| [`list_experiments.py`](quant-method/alphacrafter/list_experiments.py) | 枚举实验和状态 | `root` |
| [`miner_lite.py`](quant-method/alphacrafter/miner_lite.py) | 因子生成、统计准入与维护 | `--input`、`--output-dir`、`--candidate-file`、`--factor-return-model`、`--research-market`、`--registry`、`--maintenance-interval-days`、`--as-of`、`--snapshot-dir`、`--require-explicit-label-dates`、`--calendar`、`--label-mode`、`--train-start`、`--train-end`、`--validation-start`、`--validation-end`、`--backtest-start`、`--backtest-end`、`--llm-model`、`--llm-count`、`--research-brief-file`、`--llm-factor-mine-skill-dir`、`--llm-weekly-feedback-dir`、`--llm-weekly-xgboost-feedback`、`--memory-db`、`--memory-universe`、`--memory-limit`、`--llm-xgboost-complement`、`--llm-mechanism-first`、`--llm-seed-file`、`--search-rounds`、`--search-budget`、`--search-batch-size`、`--search-folds`、`--search-reserve`、`--llm-orchestrator`、`--llm-agents`、`--max-pairs`、`--novelty-baseline`、`--max-temporal`、`--min-train-ic`、`--min-valid-ic`、`--min-train-icir`、`--min-valid-icir`、`--min-train-hit`、`--min-valid-hit`、`--min-coverage`、`--max-turnover` |
| [`screener_lite.py`](quant-method/alphacrafter/screener_lite.py) | 按时点筛选因子和合成股票评分 | `--input`、`--factor-library`、`--output-dir`、`--library-history`、`--allow-retrospective-library`、`--as-of`、`--signal-start`、`--require-explicit-label-dates`、`--calendar`、`--lookback-days`、`--min-suitability`、`--similarity-threshold`、`--max-factors`、`--llm-regime-model`、`--llm-relevance-model`、`--llm-similarity-model` |
| [`trader_lite.py`](quant-method/alphacrafter/trader_lite.py) | 策略验证、冻结和后续日内回测 | `--predictions`、`--input`、`--output-dir`、`--selection-start`、`--selection-end`、`--backtest-start`、`--backtest-end`、`--configs`、`--cost-rate`、`--annual-risk-free-rate`、`--initial-cash`、`--calendar`、`--require-explicit-label-dates`、`--research-pool`、`--research-horizon` |
| [`alphax/miner_memory.py`](quant-method/alphacrafter/alphax/miner_memory.py) | 从搜索反馈导入 SQLite 记忆 | `--db`、`--search-feedback`、`--market`、`--universe`、`--data-version` |

### 因子准入和策略研究

| 入口 | 能力 | 参数 |
| --- | --- | --- |
| [`run_agent_execution_demo.py`](quant-method/alphacrafter/run_agent_execution_demo.py) | 回顾性 Agent 执行演示 | `--miner-dir`、`--input`、`--calendar`、`--output-dir`、`--admission-as-of`、`--selection-start`、`--selection-end` |
| [`run_candidate_batch.py`](quant-method/alphacrafter/run_candidate_batch.py) | 固定预算批量评估候选 | `--input`、`--calendar`、`--llm-candidates`、`--output-dir`、`--budget`、`--as-of`、`--train-start`、`--train-end`、`--validation-start`、`--validation-end`、`--skip-trades` |
| [`sweep_llm_admission.py`](quant-method/alphacrafter/sweep_llm_admission.py) | 扫描 LLM 候选准入门槛 | `--batch-dir`、`--output-dir` |
| [`compare_gate_ladder.py`](quant-method/alphacrafter/compare_gate_ladder.py) | 比较不同准入门槛对后续策略的影响 | `--sweep-dir`、`--input`、`--calendar`、`--output-dir`、`--start`、`--end` |
| [`run_research_pool.py`](quant-method/alphacrafter/run_research_pool.py) | 生成分周期研究池与策略验证计划 | `--metrics`、`--candidates`、`--temporal-audit`、`--market`、`--horizons`、`--output-dir`、`--strategy-validation-start`、`--strategy-validation-end`、`--holdout-start`、`--holdout-end` |
| [`run_miner_walk_forward.py`](quant-method/alphacrafter/run_miner_walk_forward.py) | 因子前推研究 | `--input`、`--output-dir`、`--max-factors`、`--factor-counts`、`--topk`、`--n-drop`、`--validation-cost-rate`、`--max-pairs`、`--min-abs-train-ic` |
| [`run_csi300_research.py`](quant-method/alphacrafter/run_csi300_research.py) | 构建沪深 300 研究面板并评估候选 | `--output-dir` |
| [`run_cn_portfolio_research.py`](quant-method/alphacrafter/run_cn_portfolio_research.py) | 沪深 300 跨日持仓及交易成本模拟 | `--output-dir` |
| [`run_four_factor_template_research.py`](quant-method/alphacrafter/run_four_factor_template_research.py) | 四因子模板及候选替换研究 | `--output-dir` |
| [`run_core_agent_blend.py`](quant-method/alphacrafter/run_core_agent_blend.py) | 四因子组合与 Agent 资金袖套对照 | `--core-dir`、`--approved-agent-signals`、`--agent-cash`、`--agent-weight`、`--output-dir` |
| [`run_combination_metrics.py`](quant-method/alphacrafter/run_combination_metrics.py) | 评估候选加入四因子组合后的指标 | `--miner-dir`、`--panel`、`--calendar`、`--start`、`--end`、`--output`、`--topk` |
| [`run_frozen_positive_holdout.py`](quant-method/alphacrafter/run_frozen_positive_holdout.py) | 重放冻结的沪深 300 四因子留出实验 | `--output-dir`、`--resume-prefix-order-failure` |
| [`verify_positive_holdout.py`](quant-method/alphacrafter/verify_positive_holdout.py) | 独立校验留出期产物 | `--experiment-dir` |
| [`run_positive_robustness.py`](quant-method/alphacrafter/run_positive_robustness.py) | 四因子组合稳健性研究 | `--source-dir`、`--output-dir` |
| [`run_medium_frequency.py`](quant-method/alphacrafter/run_medium_frequency.py) | 分频因子池、日周月策略和周频 XGBoost 配对 | `--output-dir`、`--miner-dir`、`--feature-panel`、`--evaluate-strategies`、`--factor-strength-rule`、`--compare-weekly-xgboost`、`--compare-weekly-xgboost-additions`、`--weekly-xgboost-augmentation-weight`、`--weekly-xgboost-augmentation-method` |
| [`run_medium_diagnostic.py`](quant-method/alphacrafter/run_medium_diagnostic.py) | 分频研究诊断 | `--research-dir`、`--output-dir` |
| [`run_medium_attribution.py`](quant-method/alphacrafter/run_medium_attribution.py) | 分频组合收益归因 | `--research-dir`、`--baseline-dir`、`--output-dir` |
| [`run_signal_stability.py`](quant-method/alphacrafter/run_signal_stability.py) | 信号稳定性 | `--research-dir`、`--attribution-dir`、`--output-dir`、`--experiment` |
| [`factor_return_model.py`](quant-method/alphacrafter/factor_return_model.py) | 训练或应用因子未来收益排序模型 | 两个子命令共有 `--output-dir`、`--candidate-file`、`--panel`、`--calendar`。`train` `--source`、`--export-panel`。`score` `--model-dir`、`--as-of` |
| [`run_factor_return_review.py`](quant-method/alphacrafter/run_factor_return_review.py) | 回顾因子收益排序 | `--panel`、`--candidate-file`、`--output-dir`、`--source` |

### 数据准备与审计

| 入口 | 能力 | 参数 |
| --- | --- | --- |
| [`download_market_data.py`](quant-method/machine-learn/download_market_data.py) | 下载 S&P 500 快照成分与行情 | `--market`、`--start`、`--end`、`--workers`、`--overwrite`、`--root` |
| [`download_historical_member_prices.py`](quant-method/machine-learn/download_historical_member_prices.py) | 补充历史成员行情 | `--membership`、`--existing-quotes`、`--output-dir`、`--start`、`--end`、`--max-symbols`、`--delay` |
| [`export_xnys_calendar.py`](quant-method/machine-learn/export_xnys_calendar.py) | 导出独立 XNYS 日历 | `--observed-calendar`、`--start`、`--end`、`--output` |
| [`build_features.py`](quant-method/machine-learn/build_features.py) | 构建因子特征、标签、日历和清单 | `--input`、`--output`、`--fundamentals`、`--news`、`--calendar`、`--overwrite` |
| [`audit_sp500_universe.py`](quant-method/machine-learn/audit_sp500_universe.py) | 审计 S&P 500 特征与成分时点 | `--features`、`--snapshot`、`--calendar`、`--source-manifest`、`--output` |
| [`audit_yahoo_supplement.py`](quant-method/machine-learn/audit_yahoo_supplement.py) | 审计历史成分补充行情 | `--membership`、`--calendar`、`--quotes-dir`、`--start`、`--end`、`--output` |

### Qlib 与模型基线

| 入口 | 能力 | 参数 |
| --- | --- | --- |
| [`train_xgboost_qlib.py`](quant-method/machine-learn/train_xgboost_qlib.py) | 训练 XGBoost 股票模型并导出信号 | `--input`、`--output-dir`、`--label-column`、`--train-start`、`--train-end`、`--valid-start`、`--valid-end`、`--backtest-start`、`--backtest-end`、`--objective`、`--n-estimators`、`--max-depth`、`--learning-rate`、`--min-child-weight`、`--subsample`、`--colsample-bytree`、`--gamma`、`--reg-alpha`、`--reg-lambda`、`--early-stopping-rounds`、`--workers`、`--feature-transform` |
| [`qlib_portfolio_backtest.py`](quant-method/machine-learn/qlib_portfolio_backtest.py) | 用 Qlib 对预测信号做组合回测 | `--predictions`、`--signal-column`、`--raw-dir`、`--provider-dir`、`--output-dir`、`--start`、`--end`、`--topk`、`--n-drop`、`--risk-degree`、`--initial-cash`、`--open-cost`、`--close-cost`、`--impact-cost`、`--min-cost`、`--workers`、`--rebuild-provider` |
| [`compare_alpha158_qlib.py`](quant-method/machine-learn/compare_alpha158_qlib.py) | 校验 Alpha158 因子与 Qlib 实现的一致性 | `--input`、`--provider`、`--candidate-file`、`--output`、`--tickers`、`--start`、`--end`、`--absolute-tolerance` |
| [`run_alpha158_model_diagnostic.py`](quant-method/machine-learn/run_alpha158_model_diagnostic.py) | Alpha158 模型对照诊断 | `--input`、`--calendar`、`--candidate-file`、`--output-dir` |
| [`run_factor_optimization.py`](quant-method/machine-learn/run_factor_optimization.py) | 因子优化实验 | `--input`、`--output-dir` |
| [`run_factor_trajectories.py`](quant-method/machine-learn/run_factor_trajectories.py) | 固定预算因子研究轨迹 | `--provider-dir`、`--output-dir`、`--mutations`、`--seed` |
| [`run_qlib_three_way.py`](quant-method/machine-learn/run_qlib_three_way.py) | Qlib 三组消融对照 | `--provider-dir`、`--output-dir`、`--seed` |
| [`run_xgb_experiments.py`](quant-method/machine-learn/run_xgb_experiments.py) | 批量 XGBoost 实验 | `--input`、`--output-dir` |
| [`plot_xgb_results.py`](quant-method/machine-learn/plot_xgb_results.py) | 绘制 XGBoost 实验结果 | `--input-dir`、`--output-dir`、`--start-date` |
| [`run_market_state_experiment.py`](quant-method/machine-learn/run_market_state_experiment.py) | 市场状态实验 | 参数写在脚本中 |
| [`run_xgb_5d_experiment.py`](quant-method/machine-learn/run_xgb_5d_experiment.py) | 5 日目标实验 | 参数写在脚本中 |
| [`run_xgb_blend_experiment.py`](quant-method/machine-learn/run_xgb_blend_experiment.py) | XGBoost 预测融合 | 参数写在脚本中 |
| [`run_xgb_rank_experiment.py`](quant-method/machine-learn/run_xgb_rank_experiment.py) | XGBoost 排序模型实验 | 参数写在脚本中 |
