# 五交易日预测目标实验

- 标签：`target_5d = close[t+5] / close[t] - 1`
- 特征：36 个技术特征，逐日 `CSRankNorm`
- 损失函数：`reg:pseudohubererror`
- 树轮数：验证集 Rank IC 选择为 `50`
- 时间切分：训练至 2022-12-23，验证 2023，回测 2024-01-02 至 2026-02-20

## 结果

| 方法 | 验证 Rank IC | 回测 Rank IC |
| --- | ---: | ---: |
| XGBoost 5 日目标 | 0.015432 | 0.012483 |
| `-return_5d` 基线 | -0.003527 | 0.007646 |

详细结果：`data/qlib/xgb_5d_experiment/results.json`。
