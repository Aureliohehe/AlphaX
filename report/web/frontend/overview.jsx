import {useMemo, useState} from "react";
import Chart from "./chart";
import {BestPair, holdoutTrials} from "./dashboard";
import {format, Metrics, pct} from "./shared";

const names = {standard: "标准成本", stress: "加压成本", benchmark: "匹配指数代理"};
const ranges = {all: "全部", 2024: "2024", 2025: "2025"};
const empty = [];

function OverviewChart({series}) {
  const [range, setRange] = useState("all");
  const [pointer, setPointer] = useState(null);
  const [keys, setKeys] = useState(() => Object.keys(names));
  const points = useMemo(() => range === "all" ? series : series.filter(({date}) => date.startsWith(range)), [range, series]);
  const point = points[pointer ?? points.length - 1];
  const readings = point ? keys.map((key) => names[key] + " " + format(point[key], 3)) : [];
  function toggle(key) {
    if (!keys.includes(key)) setKeys([...keys, key]);
    else if (keys.length > 1) setKeys(keys.filter((item) => item !== key));
  }
  return (
    <article className="panel chart-panel">
      <div className="panel-head">
        <div><h2>冻结四因子净值</h2><p>沪深 300 · 2024–2025 · 留出回放</p></div>
        <div className="segmented" role="group" aria-label="图表时间范围">
          {["all", "2024", "2025"].map((id) => <button key={id} aria-pressed={range === id} onClick={() => {
            setRange(id);
            setPointer(null);
          }}>{ranges[id]}</button>)}
        </div>
      </div>
      <div className="chart-legend" role="group" aria-label="切换曲线">
        {Object.entries(names).map(([key, name]) => <button key={key} data-series={key} aria-pressed={keys.includes(key)}
          onClick={() => toggle(key)}><i aria-hidden="true" />{name}</button>)}
      </div>
      <div className="chart-body">
        {point ? <Chart id="nav-chart" points={points} keys={keys} selected={pointer} onSelect={setPointer}
          label="冻结组合标准成本、加压成本与匹配指数代理的逐日净值，使用左右方向键查看交易日" /> :
          <p className="empty">{series.length ? "该时段暂无交易数据。" : "本机缺少冻结组合的逐日收益文件。"}</p>}
        {point && pointer !== null && <div className="chart-hint" aria-hidden="true">{[point.date, ...readings].join("\n")}</div>}
      </div>
      <div className="chart-caption" aria-live={pointer === null ? "off" : "polite"}>
        {point ? [point.date, ...readings].join(" · ") : !series.length && "请确认本机包含冻结回放产物。"}
      </div>
    </article>
  );
}

export default function Overview({data}) {
  const [standard, stress] = holdoutTrials(data.holdout);
  return (
    <section id="overview" data-view>
      <div className="page-heading"><div><h1>研究总览</h1><p>查看因子研究、配对回放和本地验证结果。</p></div><div className="date-chip">沪深 300 · 2024–2025</div></div>
      <Metrics aria-label="冻结四因子关键指标" items={[
        ["累计净收益", pct(standard?.cumulative_return), "标准成本", "good"],
        ["加压净收益", pct(stress?.cumulative_return), "双倍佣金与滑点", "good"],
        ["Sharpe", format(standard?.sharpe), "无风险利率为 0"],
        ["最大回撤", pct(standard?.max_drawdown), "策略门槛 −8.00%"],
      ]} />
      <div className="overview-grid">
        <OverviewChart series={data.holdout?.series ?? empty} />
        <article className="panel best-panel">
          <div className="panel-head"><div><h2>最佳配对回放</h2><p>新增因子对周频 XGBoost 的贡献</p></div></div>
          <div id="best"><BestPair pair={data.paired} /></div>
          <a className="button secondary full" href="#results">查看实验对照</a>
        </article>
      </div>
      <article className="panel workflow-strip">
        <div className="panel-head"><div><h2>因子研究流程</h2><p>从模型提案到策略冻结</p></div><a className="text-link" href="#workflow">查看 Agent loop</a></div>
        <div className="steps"><div><b>Miner</b><span>提案与准入</span></div><div><b>Screener</b><span>时点评分</span></div><div><b>Trader</b><span>配对回放</span></div><div><b>反馈</b><span>修订下一轮候选</span></div></div>
      </article>
    </section>
  );
}
