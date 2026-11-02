# XGBoost Pairwise Ranking 实验

- 目标：`rank:pairwise`
- 分组：每个交易日一个 group
- 标签：将 Qlib `CSRankNorm` 标签转换为每个交易日内的整数相关性等级，满足 XGBoost ranking 的标签要求
- 特征：逐日 `CSRankNorm`
- 最佳树轮数：`400`，仅依据 2023 验证集 Rank IC 选择

## 验证集候选轮数

 rounds  mean_daily_rank_ic  rank_icir  rank_ic_hit_ratio  mean_top_bottom_decile_spread
     25            0.012355   0.084270              0.528                       0.000267
     50            0.013349   0.092206              0.536                       0.000316
     75            0.013599   0.093780              0.528                       0.000354
    100            0.014007   0.096766              0.544                       0.000282
    150            0.015032   0.103168              0.548                       0.000407
    200            0.015283   0.105788              0.544                       0.000490
    300            0.015866   0.108748              0.540                       0.000427
    400            0.016456   0.112973              0.536                       0.000495

## 最终结果

- 验证集：`{'rows': 120956, 'mean_daily_ic': 0.01855631573864197, 'daily_ic_std': 0.13087820911761203, 'mean_daily_rank_ic': 0.0164563142811743, 'daily_rank_ic_std': 0.1456660054266051, 'rank_icir': 0.1129729220828118, 'rank_ic_hit_ratio': 0.536, 'mean_top_bottom_decile_spread': 0.0004948103637669043}`
- 回测集：`{'rows': 263525, 'mean_daily_ic': 0.012912292795884604, 'daily_ic_std': 0.13767338107915714, 'mean_daily_rank_ic': 0.011228416554613369, 'daily_rank_ic_std': 0.15233269467412436, 'rank_icir': 0.0737098268932589, 'rank_ic_hit_ratio': 0.5092592592592593, 'mean_top_bottom_decile_spread': 0.0007485280444078817}`

该实验使用 Qlib `DatasetH` 准备数据，并通过 XGBoost 原生 ranking API 设置每日 group。
