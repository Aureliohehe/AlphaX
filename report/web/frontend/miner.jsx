import {useState} from "react";
import {JobMonitor} from "./shared";

export default function Miner({jobs, selected, select, launch, busy, settings, settingsError, onLibrary}) {
  const [error, setError] = useState("");
  const values = settings?.values;
  const context = settings?.has_key ?
    [values.model, values.agents + " 个 Agent", values.rounds + " 轮",
      "每轮最多 " + values.candidates + " 个候选", "截点 " + values.initial_as_of].join(" · ") :
    "请先在系统设置中配置 LLM API 密钥。";

  async function submit(event) {
    event.preventDefault();
    setError("");
    try {
      await launch("miner", {kind: "miner", ...Object.fromEntries(new FormData(event.currentTarget))});
    } catch (err) {
      setError(err.message);
    }
  }
  return (
    <section id="miner" data-view>
      <div className="page-heading"><div><h1>因子挖掘</h1><p>用自然语言提出研究假设，交由 Miner 生成候选并验证。</p></div><a href="#settings" className="button secondary">实验设置</a></div>
      <div className="runs-grid">
        <form id="miner-form" className="panel section-panel research-form" onSubmit={submit}>
          <label>研究假设<textarea name="brief" rows={6} maxLength={4000} required
            placeholder="例如：探索中期趋势在低波动股票中的持续性，比较不同回看窗口，并说明哪些结果会推翻假设。" /></label>
          <p className="form-hint" data-invalid={Boolean(settingsError)}>{settingsError || context}</p>
          <div className="form-actions">
            <button className="button primary" disabled={busy || !settings?.has_key}>开始挖掘</button>
            <p className="form-error" role="alert">{error}</p>
          </div>
        </form>
        <JobMonitor kind="miner" jobs={jobs.filter(({kind}) => kind === "miner")} selected={selected.miner}
          onSelect={(id) => select("miner", id)} onLibrary={onLibrary} />
      </div>
    </section>
  );
}
