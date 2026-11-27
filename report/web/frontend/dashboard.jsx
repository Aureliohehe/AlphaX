import {delta, format, integer, Missing, pct, Source, Table} from "./shared";

export function holdoutTrials(holdout) {
  return [1, 2].map((multiplier) => holdout?.trials.find((trial) => trial.cost_multiplier === multiplier));
}

export function HoldoutTable({holdout}) {
  const [standard, stress] = holdoutTrials(holdout);
  if (!standard || !stress) return <Missing />;
  return <>
    <Table headers={["指标", "标准成本", "双倍佣金与滑点"]} rows={[
      ["累计净收益", pct(standard.cumulative_return), pct(stress.cumulative_return)],
      ["年化净收益", pct(standard.annualized_return), pct(stress.annualized_return)],
      ["Sharpe", format(standard.sharpe), format(stress.sharpe)],
      ["最大回撤", pct(standard.max_drawdown), pct(stress.max_drawdown)],
      ["2024 年净收益", pct(standard.calendar_year_returns?.["2024"]), pct(stress.calendar_year_returns?.["2024"])],
      ["2025 年净收益", pct(standard.calendar_year_returns?.["2025"]), pct(stress.calendar_year_returns?.["2025"])],
    ]} />
    <p className="note warning">两档成本情景的最大回撤均超过 8% 准入门槛。</p>
    <Source directory={holdout.source} />
  </>;
}

export function BestPair({pair}) {
  if (!pair) return <Missing />;
  const {variant, followup} = pair;
  const {core, augmented} = variant.results;
  const rejected = followup?.status === "rejected_on_frozen_evaluation";
  return <>
    <p className="best-factor">低波动平滑因子 · 120 日</p>
    <div className="best-impact">{delta(variant.return_delta["1"])} <small>百分点</small></div>
    <p>标准成本净收益提升</p>
    <div className="best-lines">
      <div><span>标准成本</span><strong>{pct(core.costs["1"].cumulative_return)} → {pct(augmented.costs["1"].cumulative_return)}</strong></div>
      <div><span>加压成本</span><strong>{pct(core.costs["2"].cumulative_return)} → {pct(augmented.costs["2"].cumulative_return)}</strong></div>
    </div>
    <p className={rejected ? "note warning" : "note"}>{rejected ? "2026 年冻结检验未通过" : "已观察窗口的回顾性结果"}</p>
  </>;
}

export function Comparison({pair}) {
  if (!pair) return <Missing />;
  const {variant, followup} = pair;
  const {core, augmented} = variant.results;
  const fields = [
    ["累计净收益", (r) => r.cumulative_return],
    ["最大回撤", (r) => r.max_drawdown],
    ["2024 年净收益", (r) => r.calendar_year_returns?.["2024"]],
    ["2025 年净收益", (r) => r.calendar_year_returns?.["2025"]],
  ];
  const rows = fields.map(([label, get]) => [
    label, pct(get(core.costs["1"])), pct(get(augmented.costs["1"])),
    pct(get(core.costs["2"])), pct(get(augmented.costs["2"])),
  ]);
  rows.push(["累计净收益增量", "—", delta(variant.return_delta["1"]) + " 个百分点",
    "—", delta(variant.return_delta["2"]) + " 个百分点"]);
  return <>
    <p><code>{variant.expression}</code></p>
    <Table headers={["指标", "标准 · 基线", "标准 · 增强", "加压 · 基线", "加压 · 增强"]} rows={rows} />
    <p className="technical">相同评分样本 {integer(variant.signal_rows)} 条 · 训练期 {pair.training.join(" 至 ")} · 配对窗口 {pair.window.join(" 至 ")}</p>
    {followup && <p className="note warning">2026 年冻结检验相对基线的净收益差为 {delta(followup.net_return_delta?.["1"])} 和 {delta(followup.net_return_delta?.["2"])} 个百分点，两档成本均未通过。</p>}
    <Source directory={pair.source} />
  </>;
}

export function SearchRounds({search, mechanism}) {
  return <>
    {search ? <>
      <p><span className="tag">美股历史实验</span> {integer(search.candidates)} 项去重候选 · {integer(search.strict_accepted.length)} 项严格准入</p>
      <Table headers={["迭代", "提案数", "新增候选", "代表候选最差折 Rank ICIR"]}
        rows={search.rounds.map((r) => [r.round, r.requested, r.unique, format(r.best_worst_fold_rank_icir, 4)])} />
    </> : <Missing />}
    {mechanism && <p className="note">沪深 300 机制优先实验进入研究池的候选数：日频 {integer(mechanism["1"])} · 周频 {integer(mechanism["5"])} · 20 日频率 {integer(mechanism["20"])}</p>}
  </>;
}
