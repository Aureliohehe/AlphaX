export const paths = [
  "AC=quant-method/alphacrafter", "ML=quant-method/machine-learn", 'D="$ML/data/qlib"',
];
export const recipes = {
  miner: {
    hint:"使用 LLM 环境与本机的历史面板，先设置 DEEPSEEK_API_KEY。输出目录需要是新目录。",
    lines: [
      'python "$AC/miner_lite.py" ' + "\\",
      '  --input "$D/alphacrafter_feedback_search_v1/features.csv.gz" ' + "\\",
      '  --calendar "$ML/data/processed/sp500_xnys_xcal_4_13_3.csv" ' + "\\",
      '  --require-explicit-label-dates --as-of 2022-12-30 ' + "\\",
      '  --train-start 2016-01-04 --train-end 2021-12-30 ' + "\\",
      '  --validation-start 2022-01-03 --validation-end 2022-12-30 ' + "\\",
      '  --backtest-start 2023-01-03 --backtest-end 2023-12-29 ' + "\\",
      '  --llm-model deepseek-chat --llm-agents 3 --llm-orchestrator langgraph ' + "\\",
      '  --search-rounds 3 --search-budget 30 --search-batch-size 10 --search-folds 3 ' + "\\",
      '  --output-dir "$D/web_miner_run"',
    ],
  },
  paired: {
    hint:"使用包含 XGBoost 和 HDF5 依赖的环境，以及已完成的先前 Miner 产物。",
    lines: [
      'python "$AC/run_medium_frequency.py" ' + "\\",
      '  --miner-dir "$D/llm_rank_aware_Ut07VD/miner" ' + "\\",
      '  --evaluate-strategies --compare-weekly-xgboost-additions ' + "\\",
      '  --weekly-xgboost-augmentation-method refit ' + "\\",
      '  --output-dir "$D/web_weekly_pairs"',
    ],
  },
  feedback: {
    hint:"读取上一步的已完成配对结果。使用 LLM 环境，先设置 DEEPSEEK_API_KEY。",
    lines: [
      'python "$AC/miner_lite.py" ' + "\\",
      '  --input "$D/llm_complementary_factors_nJXouK/miner_panel.csv.gz" ' + "\\",
      '  --as-of 2023-12-31 --train-start 2019-01-01 --train-end 2020-12-31 ' + "\\",
      '  --validation-start 2021-01-01 --validation-end 2021-12-31 ' + "\\",
      '  --backtest-start 2024-01-01 --backtest-end 2025-12-31 ' + "\\",
      '  --llm-model deepseek-chat --llm-orchestrator langgraph --llm-agents 3 ' + "\\",
      '  --llm-count 12 --llm-xgboost-complement ' + "\\",
      '  --llm-weekly-xgboost-feedback "$D/web_weekly_pairs/replay_summary.json" ' + "\\",
      '  --output-dir "$D/web_weekly_next"',
    ],
  },
};
