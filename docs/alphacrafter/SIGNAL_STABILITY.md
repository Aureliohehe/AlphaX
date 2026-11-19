# 正收益改进：固定组合与绝对趋势

2026-11-17，在已经用于迭代的 2022–2023 开发窗口完成两轮冻结对照。第一轮冻结 6 组信号方案，每组测试标准/两倍佣金及滑点；第二轮根据第一轮结果冻结 3 组绝对趋势对照，同样测试两档成本。合计 18 条路径，均为绕过因子准入的探索性研究；未读取或评估 2024–2025 留出。

## 收益结果

以下为扣费累计收益。标准成本为每侧佣金 3bp、滑点 5bp，印花税沿用历史规则。两倍成本只翻倍佣金及滑点，其他预算、持仓期、目标敞口保持不变。

| 第一轮信号 | 标准成本 | 两倍成本 |
|---|---:|---:|
| 原盈利配置：5 因子，最近 12 个成熟周标签筛选 | +3.42% | +1.48% |
| 同 5 因子，最近 52 个成熟周标签筛选 | +1.24% | +1.78% |
| 固定 5 因子等权 | +2.55% | +2.51% |
| 固定 4 因子：低波动、低下行风险、两种长期趋势 | **+5.22%** | **+5.22%** |
| 固定 2 个防御因子 | +2.86% | +2.41% |
| 固定 2 个长期趋势因子 | -34.15% | -34.47% |

固定四因子组合比原盈利配置改善 1.80 个百分点。其改变同时包括剔除成交量稳定性与取消动态筛选，不能把差异单独归因于某一项。保持相同 5 因子时，取消动态筛选反而降低标准成本收益；延长筛选窗口也没有改善结果。

第二轮的绝对趋势条件为 `mf_momentum_120_skip20 > 0` 且 `mf_trend_60_120 > 0`，任一缺失或非正则不进入选股/保留排名。条件只使用信号日已知特征，无前瞻标签。特征排名仍在当日原成分股横截面计算，再过滤股票。

| 第二轮信号与持仓 | 标准成本 | 两倍成本 | 标准成本年化 | Sharpe | 最大回撤 | 2022 | 2023 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 固定四因子，30 只 | +5.22% | +5.22% | +2.68% | 0.30 | -11.63% | -3.96% | +9.55% |
| 加绝对趋势过滤，30 只 | -7.10% | -6.72% | -3.76% | -0.26 | -18.34% | -14.39% | +8.51% |
| 加绝对趋势过滤，15 只 | **+6.67%** | **+7.28%** | +3.42% | 0.34 | -11.89% | -6.76% | +14.41% |

15 只版本同时将保留缓冲从 30 调至 15，其他执行规则保持原设定。它比原盈利组合提高 3.26 个百分点，但比固定四因子 30 只组合的年度下行表现更差，不能仅按最高累计收益宣布为默认策略。

费用预算固定，成本变化会影响买卖完成时间、持仓路径和期末清仓预留日数。因此两倍成本偶尔产生更高收益，不代表费用对相同持仓有正贡献。

全部 18 条路径均未通过原有年化 >8%、Sharpe >0.6、最大回撤 >-8% 的数值门槛。候选由已观察到的开发期表现启发，存在研究选择偏差；这些正收益不能证明独立样本外转正，尚未构成正式策略准入。

## 可复现性修复与验证

发现原 CN 执行器按浮点分数直接排序，约 10⁻¹⁶ 的求和顺序/CSV 解析差异可能改变接近并列股票的先后。现统一将选股分数四舍五入至 12 位小数，再按股票代码打破并列。该精度规则在所有新对照中一致使用，不依据收益选择。原标准成本 +3.42% 逐日保持一致；原两倍成本结果由 +1.37% 修正为 +1.48%，详见 [成本复核](POSITIVE_ROBUSTNESS.md)。

第一轮初次运行产物 `alphacrafter_signal_stability_20261117` 为修复前诊断，最终结果以 `_v2` 为准；两者均保留源码快照。旧 Miner/1 日组合表达式源码已发生无关修正，本轮核对历史快照、原始数据、实际复用的特征/筛选源码及行情读取函数，并单独冻结新的执行器与新信号代码。原保存信号和重建信号在相同修正执行器下逐日收益、NAV、费用、现金及换手一致。

全部 137 项 AlphaCrafter 测试通过，包括新增的未来特征/标签扰动、缺失成分、绝对趋势过滤及并列分数往返读写测试。另对全部 18 条输出路径独立核对现金与库存流水、持仓估值、NAV 复利、逐笔费用、换手、预算、前一交易日信号、最短持仓、卖出日历及期末零库存；结果记录在各输出目录 `artifact_audit.json`。

复现：

```bash
/tmp/alpha-csi-inspect/bin/python quant-method/alphacrafter/run_signal_stability.py \
  --research-dir quant-method/machine-learn/data/qlib/alphacrafter_medium_frequency_20261117 \
  --attribution-dir quant-method/machine-learn/data/qlib/alphacrafter_medium_attribution_20261117 \
  --output-dir quant-method/machine-learn/data/qlib/alphacrafter_signal_stability_new

/tmp/alpha-csi-inspect/bin/python quant-method/alphacrafter/run_signal_stability.py \
  --experiment absolute_trend \
  --research-dir quant-method/machine-learn/data/qlib/alphacrafter_medium_frequency_20261117 \
  --attribution-dir quant-method/machine-learn/data/qlib/alphacrafter_medium_attribution_20261117 \
  --output-dir quant-method/machine-learn/data/qlib/alphacrafter_absolute_trend_new
```

实际完成产物为 `alphacrafter_signal_stability_20261117_v2` 与 `alphacrafter_absolute_trend_20261117`，均位于 `quant-method/machine-learn/data/qlib/`。原有数据时点认证、复权可分割单位、整手/最低佣金/流动性及近似交易限制等局限仍适用。
