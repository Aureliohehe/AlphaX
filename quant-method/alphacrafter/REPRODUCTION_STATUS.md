# AlphaCrafter 复现状态与严格口径诊断

## 结论

目前是 **Miner / Screener / Trader 的部分流程复现**，不是论文 Table 2 的收益复现。优先保证无前视、拒绝失败策略、正确计算费用与风险，而不是通过放宽门槛追赶论文数字。

依据仓库 PDF：AlphaCrafter，arXiv:2605.05580v2；重点核对 Algorithm 1–4、Table 1、Appendix B.2/B.3。

## 第一轮修正：Trader 与信号边界

| 位置 | 原问题 | 当前处理 |
|---|---|---|
| Trader Algorithm 3 | 没有可行配置时仍回退交易 | 最多三个试验；仅选可行配置，否则 `no_feasible_strategy`、全程现金 |
| Trader Algorithm 4 | 单边策略仍给另一侧预留股票名额 | 按实际暴露选股；支持 nshort=0 的纯多头及 nlong=0 的纯空头 |
| Trader 选股 | 先删除未来收益缺失股票 | 仅使用当前信号选股；持仓的执行收益缺失则失败，不替换、不视作零收益 |
| Trader 执行 | 日内收益搭配跨日仓位差计费，稳定持仓几乎不收费 | 明确日内开仓/平仓，每日两腿收费，平仓成交额随价格变化 |
| Trader 无信号 | 沿用旧仓位 | 日内策略隔夜空仓，无足够信号则保持现金 |
| Trader 指标 | 忽略无风险收益、漏算首日回撤 | 美国市场年 rf=3.81%，按几何方式换算日 rf；回撤峰值包含初始 NAV |
| Trader 日期边界 | 未限制验证与留出重叠，期末标签可能跨界 | 强制分离区间；按全股票面板下一交易日推导 execution_date 并检查期末 |
| Trader 审计 | 结果缺少输入身份 | 记录输入 SHA-256、文件大小、路径、运行参数、pandas/numpy 版本 |
| 特征生成/Screener | 依据未来标签缺失删除信号日股票 | 保留最新未标注行；仅计算 IC 时排除缺失标签 |
| Screener 合成 | 缺失成分可能被后续因子重新填零 | 要求选中成分都有效，合成结果不依赖因子顺序 |
| Screener LLM regime | “60 行”实际上可能只是同一天部分股票 | 使用最近 60 个交易日的完整股票截面 |

交易费用定义：

```text
entry_notional = sum(abs(weight_i))
exit_notional  = sum(abs(weight_i) * (1 + open_to_close_return_i))
cost           = cost_rate * (entry_notional + exit_notional)
net_return     = sum(weight_i * open_to_close_return_i) - cost
```

成交额以当日开仓前净值归一化；报告中的 `turnover` 是两腿成交额之和 / 2，费用不能再直接用 `cost_rate * turnover`。现金不计利息；零波动 Sharpe 未定义，输出 JSON `null`，不会作为可行策略。

## 本地严格诊断

使用**已有特征和已有 Screener 信号缓存**，未重新生成因子、未重新运行历史 LLM、未重建特征文件：

```bash
python3 quant-method/alphacrafter/trader_lite.py \
  --output-dir quant-method/machine-learn/data/qlib/alphacrafter_trader_strict_v2
```

输入身份：

- 特征：`machine-learn/data/processed/sp500_features.csv.gz`
  - SHA-256：`d1576247c3b7ff2d0ccacf1fa13bc50d14ef7a48c3388d6a35dfd6262dbcb64b`
- 信号：`machine-learn/data/qlib/alphacrafter_screener_lite/predictions.csv`
  - SHA-256：`74f195cab6c9003c33e0266688ff69f15c9886a49134f0e540ac5b43db0463cf`
- 环境：pandas 2.0.3、numpy 1.24.4。

验证期信号日期：2023-01-03 至 2023-12-28，共 249 天。2023-12-29 的 T+1 标签不属于验证期，已剔除。

| β | γ | Nlong / Nshort | 验证年化收益 | Sharpe（扣 rf） | 最大回撤 | 可行 |
|---|---|---|---|---|---|---|
| 0.8 | 0.5 | 50 / 25 | 7.01% | 0.440 | -8.21% | 否 |
| 0.8 | 0.0 | 50 / 50 | -0.57% | -0.856 | -8.36% | 否 |
| 0.8 | 1.0 | 50 / 0 | 16.44% | 1.019 | -10.51% | 否 |

严格门槛全部失败，留出阶段保持现金：收益与回撤均 0，Sharpe 未定义。这是拒绝规则的正确行为，不是策略复现成功，也不是调低门槛的理由。

此次缓存下可评估的留出信号到 2026-02-25，共 539 天。旧特征缓存没有完整保留最后未标注交易日；模拟器无法推导面板末行的下一交易日，因此保守排除。**不能把这一窗口声称为论文完整 542 天窗口。**

第一轮测试：

```bash
cd quant-method/alphacrafter
python3 -m unittest discover -v
# 31 tests, OK
```

覆盖严格可行域、首日亏损、无风险收益、每天双边成本、未来缺失标签、验证期末标签隔离、无信号现金、单边暴露、稳定排序、重复输入、未标注最新交易日、CLI 输出与缺失信号组合。

## 第二轮：时间隔离与快照回放

已实现（用法和本地冒烟实验见 [`TEMPORAL_REPLAY.md`](TEMPORAL_REPLAY.md)）：

- Miner `--as-of` 截断特征，屏蔽未兑现的预计算标签；1/5 日分别按标签结束日做 train/validation/backtest 期末 purge。
- 可配置独立训练/准入区间；禁止继承未来或未认证的旧 registry。
- `snapshots/YYYY-MM-DD.json` 写一次，不允许不同内容覆盖；历史回算信号单独标为 `diagnostic_predictions.csv`。
- Screener 默认按快照可用日筛选，支持快照目录回放；空快照即淘汰全部因子，未来库不会回填到过去。
- 对不规则股票交易记录，Rank IC 要等整个截面的标签结束日已知且已兑现才可使用；全量回放和截断回放的历史信号保持一致。
- 真实本地数据冒烟：484 只股票、三个预先固定技术候选，2022 年末准入，**0 个通过严格门槛**；随后 2023 年 1 月 20 个交易日没有信号，未放宽门槛。

## 尚未满足的复现条件（下一步顺序）

1. **重新生成历史快照，完成嵌套滚动实验。**
   - 回放基础已实现，但旧实验缺失的快照不能逆向恢复；旧 `alphacrafter_miner_persistent` 的 2027 年维护记录也不能用于证明 2023 年可见。
   - 需运行独立的候选生成/因子准入/Trader 调参窗口，按时间顺序产出新快照；不能修改旧时间戳冒充无泄漏数据。
   - 当前维护触发周期已存在，但评估窗口还需调用者移动；尚无自动滚动全管线运行器。
2. **补原始标签结束日与数据可见性审计。**
   - 主 Miner 的期末 purge 已实现；结束日目前由特征面板每只股票的 shift(-h) 推导。旧面板删除过中间行时可能多剔除，不能声称完整还原原始交易日。
   - 旧 `run_miner_walk_forward.py` 是单独的历史诊断脚本，尚未迁移到新快照协议，不能用它证明新流程无泄漏。
   - 需核查外部候选文件、基本面/新闻、LLM 模型记忆的时间来源；代码截止无法证明用户输入本身没有未来信息。
3. **完整执行模拟。**
   - 本轮明确修正的是日内代理，与论文 Algorithm 4 的持续持仓再平衡不同。
   - 后续需显式现金、股数、成交价格、逐日盯市、保证金（50% 初始/30% 维持）、借券/交易费用和成交约束；CSI 300 还需 T+1 与禁止做空。
   - 必须定义信号时点与成交时点，不能用收盘信息却默认无摩擦地成交在同一收盘价。
4. **历史多智能体工作流。**
   - Miner 已有可选 LangGraph fan-out；Screener 的 LLM 目前只影响最新日期。
   - 需要有调用预算、输入哈希、持久化缓存/重放的逐日 regime/relevance/similarity；Trader 需 regime-conditioned 三次策略提议与结构化校验。
5. **数据与论文实验矩阵。**
   - 核查 point-in-time 成分股、退市与生存者偏差；补基本面、财报、新闻及 CSI 300，而不只使用技术指标。
   - 新的特征构建逻辑不会自动修补旧缓存，应独立输出新版数据并记录数据清单、交易日完整性。
   - 完成 10 次独立试验、四分位区间内汇总、不同模型和消融；历史回放不能称作券商 paper-trading 实盘验证。

旧 README 中 Qlib 的收益来自不同持仓/执行/基准口径，不能与这里的日内诊断或论文 Table 2 直接比较。
