# AlphaX

AlphaX 是一个量化研究项目，提供因子发掘、按历史时点筛选信号、策略验证和组合回测。首次使用可以从 S&P 500 的模板因子流水线开始，跑通后再启用 LLM 或训练 Qlib 模型基线。

| 想做什么 | 入口 | 所需输入 |
| --- | --- | --- |
| 跑 Miner → Screener → Trader | `run_pipeline.py` | S&P 500 特征面板和交易日历 |
| 用 LLM 挖因子 | `miner_lite.py` | 特征面板、LangGraph 和模型 API 密钥 |
| 训练 XGBoost 并回测组合 | `train_xgboost_qlib.py`、`qlib_portfolio_backtest.py` | S&P 500 行情与特征面板 |
| 重放沪深 300 四因子实验 | `run_frozen_positive_holdout.py` | 沪深 300 行情、历史成分、交易日历及实验所需的缓存 |

## 1. 配置环境

以下命令使用 Bash 或 Zsh，并从仓库根目录执行。使用 Python 3.10 或更新版本。模板流水线使用 NumPy 和 pandas，无需 Docker、券商账户或 API 密钥。

```bash
git clone https://github.com/Aureliohehe/AlphaX.git
cd AlphaX

python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas typing-extensions

make test
```

如果系统默认的 `python3` 低于 3.10，请用较新的解释器创建虚拟环境，例如 `python3.13 -m venv .venv`。如果环境中没有 Make，也可以在两个子目录中各跑一次测试。

```bash
(cd quant-method/alphacrafter && python -m unittest discover -v)
(cd quant-method/machine-learn && python -m unittest discover -v)
```

## 2. 准备 S&P 500 数据

Git 仓库包含成分快照和行情清单。完整行情、特征缓存和回测产物保存在本地 `quant-method/machine-learn/data/` 下。如果已有下面的特征面板及日历文件，可直接进入第 3 步。下载需要联网，完整特征文件可能超过 1 GB。

```bash
ML=quant-method/machine-learn
AC=quant-method/alphacrafter
PANEL="$ML/data/processed/sp500_features_label_dates.csv.gz"
CALENDAR="$PANEL.calendar.csv"

python "$ML/download_market_data.py" --workers 8
python "$ML/build_features.py" --output "$PANEL"
```

下载器把行情写到 `data/raw/sp500/`。特征构建还会生成 `.calendar.csv` 和 `.manifest.json`。默认日历来自行情日期的并集，S&P 500 成分来自固定快照。做严格的历史时点研究时，还需单独核验交易日历和历史成员数据。[数据来源与限制](docs/alphacrafter/DATA_PROVENANCE_AUDIT.md)

## 3. 运行滚动因子流水线

在同一终端运行。为每次正式实验指定新的输出目录。

```bash
RUN="$ML/data/qlib/pipeline_first_run"

python "$AC/run_pipeline.py" \
  --input "$PANEL" \
  --calendar "$CALENDAR" \
  --require-explicit-label-dates \
  --allow-feature-warmup-sessions 120 \
  --output-dir "$RUN" \
  --dry-run
```

预检会打印滚动计划，并检查日期窗口和数据覆盖。预检通过后，重新执行上面的命令，移除 `--dry-run`。完整实验会多次读取面板并计算因子，耗时会明显长于预检。默认用模板生成候选。若要使用固定候选，可加 `--candidate-file <候选 JSON>`。

| 产物 | 查看什么 |
| --- | --- |
| `status.json` | 实验是否完成。失败时查看 `logs/` |
| `rounds.csv`、`snapshots/` | 每轮的候选准入和生效因子 |
| `screener_holdout/predictions.csv` | 留出期股票评分 |
| `trader/selection.json` | 验证期冻结的策略配置。没有合格配置时选择空仓 |
| `summary.json`、`trader/holdout_returns.csv` | 留出期汇总和逐日收益 |

[流水线参数、时点规则与故障排查](docs/alphacrafter/PIPELINE.md)

## 4. 可选运行方式

### 用 DeepSeek 和 LangGraph 挖因子

在同一个虚拟环境中安装 LangGraph，将 `DEEPSEEK_API_KEY` 设置为你的密钥，然后运行独立 Miner。以下命令沿用第 2 步的数据路径。

```bash
python -m pip install -r "$AC/requirements.txt"
export DEEPSEEK_API_KEY='替换成你的密钥'

python "$AC/miner_lite.py" \
  --input "$PANEL" \
  --calendar "$CALENDAR" \
  --require-explicit-label-dates \
  --llm-model deepseek-chat \
  --llm-agents 3 \
  --llm-count 12 \
  --output-dir "$ML/data/qlib/miner_llm_first_run"
```

结果见 `candidate_metrics.csv`、`accepted_factors.json` 和 `llm_candidate_audit.json`。使用外部因子研究 skill 时，加上 `--llm-factor-mine-skill-dir <本地检出目录>`。如需让滚动流水线评估 LLM 候选，可用 `--candidate-file` 传入在实验窗口开始前已确定的候选；对今天生成的历史候选，应按回顾性研究解读。[Miner、反馈搜索与 skill 配置](quant-method/alphacrafter/README.md)

### 训练 Qlib/XGBoost 基线并回测

```bash
python -m pip install -r "$ML/requirements.txt"

python "$ML/train_xgboost_qlib.py" \
  --input "$PANEL" \
  --output-dir "$ML/data/qlib/xgboost_baseline"

python "$ML/qlib_portfolio_backtest.py" \
  --predictions "$ML/data/qlib/xgboost_baseline/predictions.csv" \
  --output-dir "$ML/data/qlib/portfolio_backtest_model_drop10" \
  --n-drop 10
```

首次回测会根据 S&P 500 原始行情构建本地 Qlib 数据。模型输入、训练窗口和组合配置见[模型基线指南](quant-method/machine-learn/README.md)。

### 重放沪深 300 四因子实验

沪深 300 研究使用独立的数据和交易模拟。先按审计说明确认本地行情、历史成分和交易日历可用，再按留出期报告准备依赖和运行 `run_frozen_positive_holdout.py`。[沪深 300 数据审计](docs/alphacrafter/CSI300_DATA_AUDIT.md) · [四因子留出回放及产物](docs/alphacrafter/POSITIVE_HOLDOUT.md)

## 项目结构

| 路径 | 内容 |
| --- | --- |
| [quant-method/alphacrafter/](quant-method/alphacrafter/) | Miner、Screener、Trader、运行入口和因子预设 |
| [quant-method/machine-learn/](quant-method/machine-learn/) | 数据准备、特征工程、Qlib 模型和组合回测 |
| [docs/](docs/) | 架构、数据审计和历史实验报告 |

数据流、时点控制和各研究分支的关系见[架构说明](docs/ARCHITECTURE.md)。

## 常见问题

- **找不到特征文件或日历**。先完成第 2 步，并确保在同一个终端设置了 `ML`、`AC`、`PANEL` 和 `CALENDAR`。如果行情下载失败，检查 `quant-method/machine-learn/data/raw/sp500/manifest.json`。
- **预检提示日期覆盖不足**。确认面板覆盖验证期和留出期。用已有的旧缓存时，可通过 `--holdout-end` 指定文件实际覆盖的最后一个交易日。
- **输出目录已经存在**。选一个新的目录名，保留上次实验的结果和日志。
- **LLM 提示缺少模块或 API 密钥**。确认安装依赖和运行脚本使用的是同一个虚拟环境，并在运行 Miner 的终端设置 `DEEPSEEK_API_KEY`。
