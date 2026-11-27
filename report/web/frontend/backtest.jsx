import {useEffect, useRef, useState} from "react";
import Chart from "./chart";
import {gradeOptions} from "./factors";
import {format, JobMonitor, LibraryOptions, Metrics, pct, request} from "./shared";

const chartKeys = ["nav", "benchmark"];

function BacktestReport({result}) {
  const {metrics, selected_config: config, series} = result;
  const [day, setDay] = useState(Math.max(0, series.length - 1));
  const point = series[day];
  return (
    <div id="backtest-report">
      <Metrics items={[
        ["累计净收益", pct(metrics.cumulative_return)], ["年化净收益", pct(metrics.annualized_return)],
        ["Sharpe", format(metrics.sharpe)], ["最大回撤", pct(metrics.max_drawdown)],
      ]} />
      <div className="chart-legend"><span className="series-key">策略净值</span><span className="series-key benchmark-key">样本等权参考</span></div>
      {point ? <>
        <div className="chart-body"><Chart points={series} keys={chartKeys} selected={day} backtest
          label="留出期策略与基准净值曲线" /></div>
        <label className="timeline-control">查看交易日<input type="range" min={0} max={series.length - 1}
          value={day} aria-valuetext={point.date} onChange={(event) => setDay(Number(event.target.value))} /></label>
        <p className="chart-caption">{point.date} · 策略 {format(point.nav, 4)} · 参考 {format(point.benchmark, 4)}</p>
      </> : <p className="empty">该回测无逐日净值数据。</p>}
      <p className="form-hint">{config ? "冻结策略 · β " + config.beta + " · γ " + config.gamma +
        " · 多头 " + config.nlong + " · 空头 " + config.nshort + " · 持仓日 " + result.invested_days :
        "验证期无可用策略配置，留出期持有现金。"}</p>
    </div>
  );
}

function BacktestResults({active, completed}) {
  const [ids, setIds] = useState([]);
  const [id, setId] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [listError, setListError] = useState("");
  const latest = useRef(null);

  useEffect(() => {
    if (!active) return;
    const preferred = completed && completed !== latest.current ? completed : "";
    const controller = new AbortController();
    request("/api/backtests", {signal: controller.signal}).then((results) => {
      latest.current = completed;
      setIds(results);
      setListError("");
      setId((previous) => results.includes(preferred) ? preferred :
        results.includes(previous) ? previous : results[0] ?? "");
    }).catch((err) => {
      if (!controller.signal.aborted) setListError(err.message);
    });
    return () => controller.abort();
  }, [active, completed]);

  useEffect(() => {
    if (!id) return;
    const controller = new AbortController();
    request("/api/backtest?id=" + encodeURIComponent(id), {signal: controller.signal}).then((data) => {
      setResult({id, data});
    }).catch((err) => {
      if (!controller.signal.aborted) setError({id, message: err.message});
    });
    return () => controller.abort();
  }, [id]);

  const report = result?.id === id ? result.data : null;
  const source = report?.source;
  const resultError = error?.id === id ? error.message : "";
  const status = listError || resultError || (!id ? "选择已完成的回测查看结果。" : !report ? "正在读取回测结果…" :
    "留出期 " + report.series.length + " 个交易日 · 实验 " + id +
    (source?.name ? " · " + source.name + " · " + source.count + " 个因子" : ""));

  return (
    <article className="panel section-panel research-results">
      <div className="panel-head">
        <div><h2>留出期表现</h2><p role="status" data-invalid={Boolean(listError || resultError)}>{status}</p></div>
        <label>已完成回测<select value={id} onChange={(event) => {
          setResult(null);
          setError(null);
          setId(event.target.value);
        }}>
          <option value="">选择回测</option>
          {ids.map((value) => <option key={value} value={value}>实验 {value}</option>)}
        </select></label>
      </div>
      {report && <BacktestReport key={id} result={report} />}
    </article>
  );
}

export default function Backtest({active, jobs, selected, select, launch, busy, settings, catalog, catalogError}) {
  const [library, setLibrary] = useState("");
  const [grade, setGrade] = useState("strict");
  const [error, setError] = useState("");
  const source = catalog.find(({id}) => id === library);
  const count = source?.factors.filter((factor) => !grade || factor.grade === grade).length ?? 0;
  const values = settings?.values;
  const previousJobs = jobs.filter(({kind}) => kind === "backtest");
  const completed = previousJobs[0]?.status === "passed" ? previousJobs[0].id : null;

  useEffect(() => {
    if (library && !source) setLibrary("");
  }, [library, source]);

  async function submit(event) {
    event.preventDefault();
    setError("");
    try {
      await launch("backtest", {kind: "backtest", library, grade});
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <section id="backtest" data-view>
      <div className="page-heading"><div><h1>独立回测</h1><p>固定因子定义，执行训练、滚动准入、策略验证与留出期回测。</p></div><a href="#settings" className="button secondary">回测设置</a></div>
      <div className="runs-grid">
        <form id="backtest-form" className="panel section-panel research-form" onSubmit={submit}>
          <label>因子来源<select name="library" required value={library} onChange={(event) => setLibrary(event.target.value)}>
            <option value="">选择实验因子库</option><LibraryOptions catalog={catalog} />
          </select></label>
          <label>纳入范围<select name="grade" value={grade} onChange={(event) => setGrade(event.target.value)}>{gradeOptions}<option value="">全部候选</option></select></label>
          <p className="form-hint">{values && ["符合范围 " + count + " 个", "验证 " + values.selection_start + " 至 " + values.selection_end,
            "留出 " + values.holdout_start + " 至 " + values.holdout_end].join(" · ")}</p>
          <p className="note warning">根据所选历史因子做回顾性模拟。若因子构思或筛选使用了回测窗口的信息，请按回顾性结果解读。</p>
          <div className="form-actions"><button className="button primary" disabled={busy || !count}>运行全周期回测</button>
            <p className="form-error" role="alert">{error || catalogError}</p></div>
        </form>
        <JobMonitor kind="backtest" jobs={previousJobs} selected={selected.backtest} onSelect={(id) => select("backtest", id)} />
      </div>
      <BacktestResults active={active} completed={completed} />
    </section>
  );
}
