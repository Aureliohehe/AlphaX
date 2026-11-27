import {useRef, useState} from "react";
import {paths, recipes} from "./recipes";
import {conditions, Log, statusNames} from "./shared";

export default function Runs({jobs, selected, select, busy, launch, jobError}) {
  const [error, setError] = useState("");
  const [recipe, setRecipe] = useState("miner");
  const [copyState, setCopyState] = useState("");
  const code = useRef(null);
  const job = jobs.find(({id}) => id === selected.runs) ?? jobs[0];
  const status = jobError || (job ? conditions[job.kind] + " · " + statusNames[job.status] : "等待运行任务");
  const log = job ? "$ " + job.command + "\n\n" + job.lines.join("\n") +
    (job.exit_code === null ? "" : "\n\n退出码 " + job.exit_code) : "选择任务后，日志会显示在这里。";
  const command = [...paths, "", ...recipes[recipe].lines].join("\n");

  async function submit(event) {
    event.preventDefault();
    setError("");
    try {
      await launch("runs", {kind: new FormData(event.currentTarget).get("kind")});
    } catch (err) {
      setError(err.message);
    }
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(command);
      setCopyState("已复制");
    } catch {
      const range = document.createRange();
      range.selectNodeContents(code.current);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      setCopyState("命令已选中，可直接复制");
    }
  }

  return (
    <section id="runs" data-view>
      <div className="page-heading"><div><h1>运行中心</h1><p>运行项目命令并查看执行日志。</p></div></div>
      <div className="runs-grid">
        <article className="panel section-panel"><h2>本地验证</h2><p>选择要执行的检查。同一时间运行一个任务。</p>
          <form id="launch" onSubmit={submit}>
            <fieldset><legend>任务</legend>
              <label className="option"><input type="radio" name="kind" value="demo" defaultChecked /><span><b>离线全链路示例</b><small>用合成数据验证 Miner、Screener 和 Trader，并保留产物。</small></span></label>
              <label className="option"><input type="radio" name="kind" value="smoke" /><span><b>核心冒烟测试</b><small>检查完整流水线、空仓回退、失败停止与恢复。</small></span></label>
              <label className="option"><input type="radio" name="kind" value="preflight" /><span><b>美股流水线预检</b><small>核对本地 S&amp;P 500 面板、交易日历和运行计划。</small></span></label>
            </fieldset>
            <button type="submit" className="button primary" disabled={busy}><span className="play" aria-hidden="true"></span>开始运行</button>
            <p id="launch-error" className="error" role="alert" hidden={!error}>{error}</p>
          </form>
        </article>
        <article className="panel section-panel jobs-panel"><div className="panel-head"><div><h2>执行日志</h2><p id="job-status" role="status">{status}</p></div></div>{job && <label id="job-picker">任务记录 <select id="job-select" value={job.id} onChange={(event) => select("runs", event.target.value)}>
            {jobs.map(({id, kind, started}) => <option key={id} value={id}>{conditions[kind]} · {new Date(started * 1000).toLocaleTimeString("zh-CN")}</option>)}
          </select></label>}
          <Log id="job-log" aria-label="执行日志" value={log} />
          {job?.output && job.status === "passed" && <p className="technical">产物目录 {job.output}</p>}
        </article>
      </div>
      <article className="panel section-panel guide"><div className="panel-head"><div><h2>研究命令</h2><p>从仓库根目录在对应 Python 环境执行。</p></div></div>
        <label htmlFor="recipe">工作流</label><select id="recipe" value={recipe} onChange={(event) => { setRecipe(event.target.value); setCopyState(""); }}><option value="miner">DeepSeek Agent loop</option><option value="paired">周频 XGBoost 配对</option><option value="feedback">导入 XGBoost 反馈</option></select>
        <p id="recipe-hint">{recipes[recipe].hint}</p><pre id="recipe-code" ref={code} tabIndex={0}>{command}</pre><button id="copy" className="button secondary" onClick={copy}>复制命令</button><span id="copy-state" role="status">{copyState}</span>
      </article>
    </section>
  );
}
