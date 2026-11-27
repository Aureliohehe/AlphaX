import {useLayoutEffect, useRef} from "react";

export const pages = {
  overview: "研究总览", miner: "因子挖掘", factors: "因子库", backtest: "独立回测",
  workflow: "Agent loop", results: "实验对照", runs: "运行中心", settings: "系统设置",
};
export const conditions = {
  demo: "离线全链路示例", smoke: "核心冒烟测试", preflight: "美股流水线预检",
  miner: "因子挖掘", backtest: "独立回测",
};
export const statusNames = {running: "运行中", passed: "已通过", failed: "运行失败"};
export const grades = {strict: "已准入", research: "研究候选", unadmitted: "未准入", unknown: "缺少评估"};

export const format = (value, digits = 2) => Number.isFinite(value) ? value.toFixed(digits) : "—";
export const pct = (value) => Number.isFinite(value) ? format(value * 100) + "%" : "—";
export const integer = (value) => Number.isFinite(value) ? value.toLocaleString("zh-CN") : "—";
export const delta = (value) => Number.isFinite(value) ? (value >= 0 ? "+" : "") + format(value * 100) : "—";

export async function request(path, options) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error ?? "本地服务请求失败。");
  return payload;
}

export function postJson(path, payload) {
  return request(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
}

export function Missing() {
  return <p className="empty">本机缺少对应的实验产物。</p>;
}

export function Source({directory}) {
  return <p className="technical">本地产物 <code>{directory}</code></p>;
}

export function Table({headers, rows, label = headers.join("、")}) {
  return (
    <div className="table-wrap" tabIndex={0} role="region" aria-label={label}>
      <table>
        <thead><tr>{headers.map((header) => <th key={header} scope="col">{header}</th>)}</tr></thead>
        <tbody>{rows.map(([name, ...cells], index) => (
          <tr key={index}>
            <th scope="row">{name}</th>
            {cells.map((cell, column) => <td key={headers[column + 1]}>{cell}</td>)}
          </tr>
        ))}</tbody>
      </table>
    </div>
  );
}

export function Metrics({items, ...props}) {
  return (
    <div className="cards" {...props}>
      {items.map(([label, value, hint, tone]) => (
        <article className="metric" key={label}>
          <span className="metric-label">{label}</span>
          <strong className={value.startsWith("-") || value === "—" ? undefined : tone}>{value}</strong>
          {hint && <small>{hint}</small>}
        </article>
      ))}
    </div>
  );
}

export function Log({value, ...props}) {
  const log = useRef(null);
  const follow = useRef(true);
  useLayoutEffect(() => {
    if (follow.current) log.current.scrollTop = log.current.scrollHeight;
  }, [value]);
  return (
    <pre {...props} ref={log} tabIndex={0} onScroll={(event) => {
      const el = event.currentTarget;
      follow.current = el.scrollHeight - el.scrollTop - el.clientHeight < 25;
    }}>{value}</pre>
  );
}

export function LibraryOptions({catalog}) {
  return catalog.map(({id, name, factors}) => (
    <option key={id} value={id}>{name} · {factors.length} 个</option>
  ));
}

export function JobMonitor({kind, jobs, selected, onSelect, onLibrary}) {
  const miner = kind === "miner";
  const job = jobs.find(({id}) => id === selected) ?? jobs[0];
  const elapsed = job ? Math.max(0, Math.round((job.finished ?? Date.now() / 1000) - job.started)) : 0;
  const log = job ? [...(job.progress ?? []), ...job.lines] : [];
  if (job?.output && job.status === "passed") log.push("产物目录 " + job.output);
  if (job && job.exit_code !== null) log.push("退出码 " + job.exit_code);
  return (
    <article className="panel section-panel research-monitor">
      <h2>{miner ? "挖掘进度" : "回测进度"}</h2>
      {job && <label className="task-picker">任务<select value={job.id} onChange={(e) => onSelect(e.target.value)}>
        {jobs.map(({id, started}) => <option key={id} value={id}>{new Date(started * 1000).toLocaleString("zh-CN")} · {id}</option>)}
      </select></label>}
      <p role="status" data-invalid={job?.status === "failed"}>{job ? statusNames[job.status] + " · " + elapsed + " 秒" : "等待运行任务"}</p>
      <Log aria-label={miner ? "挖掘日志" : "回测日志"} value={job ? log.join("\n") || "正在准备任务…" : ""} />
      {miner && job?.status === "passed" && (
        <a className="button secondary" href="#factors" onClick={() => onLibrary(job.library)}>查看挖掘结果</a>
      )}
    </article>
  );
}
