import {useEffect, useState} from "react";
import {postJson} from "./shared";

function Field({draft, onChange, ...props}) {
  return <input {...props} value={draft[props.name] ?? ""} onChange={onChange} />;
}

export default function Settings({settings, error, onSave}) {
  const [draft, setDraft] = useState({});
  const [key, setKey] = useState("");
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const [invalid, setInvalid] = useState(false);

  useEffect(() => {
    if (settings) setDraft(settings.values);
  }, [settings]);

  function change(event) {
    const {name, value} = event.target;
    setDraft((previous) => ({...previous, [name]: value}));
  }

  async function save(form, clearKey = false) {
    if (!settings || saving || !form.reportValidity()) return;
    const values = Object.fromEntries(Object.entries(settings.values).map(([name, previous]) => [
      name, typeof previous === "number" ? Number(draft[name]) : draft[name],
    ]));
    const body = {values};
    if (clearKey || key) body.api_key = clearKey ? "" : key;
    setSaving(true);
    setInvalid(false);
    try {
      onSave(await postJson("/api/settings", body));
      setKey("");
      setMessage(clearKey ? "已清除界面密钥。服务环境中的密钥仍可使用。" : "设置已保存。");
    } catch (err) {
      setMessage(err.message);
      setInvalid(true);
    } finally {
      setSaving(false);
    }
  }

  return (
    <section id="settings" data-view>
      <div className="page-heading"><div><h1>系统设置</h1><p>设置新任务使用的模型、数据及实验窗口。</p></div></div>
      <form id="settings-form" className="research-form" onSubmit={(event) => {
        event.preventDefault();
        save(event.currentTarget);
      }}>
        <fieldset className="panel section-panel"><legend>LLM API</legend>
          <div className="form-grid">
            <label>模型<Field draft={draft} onChange={change} name="model" list="llm-models" required maxLength="160" placeholder="deepseek-chat" /><datalist id="llm-models"><option value="deepseek-chat"></option><option value="deepseek-reasoner"></option></datalist></label>
            <label>API 基础地址<Field draft={draft} onChange={change} name="api_base" type="url" placeholder="https://api.deepseek.com" /></label>
            <label>API 密钥<input value={key} onChange={(event) => setKey(event.target.value)} name="api_key" type="password" autoComplete="new-password" placeholder="输入后在当前服务会话使用" /></label>
          </div>
          <p className="form-hint">DeepSeek 可留空地址。自定义模型填写兼容 Chat Completions 的基础地址，例如服务的 /v1 路径。</p>
          <div className="form-actions"><p id="key-state">{settings?.has_key ? "已配置密钥。通过界面设置的密钥仅在当前服务会话保存。" : "请填写密钥，或在启动服务时设置对应的环境变量。"}</p><button type="button" className="button secondary" id="remove-key" disabled={!settings || saving} onClick={(event) => save(event.currentTarget.form, true)}>清除界面密钥</button></div>
        </fieldset>
        <fieldset className="panel section-panel"><legend>数据路径</legend>
          <div className="form-grid">
            <label className="form-wide">特征面板<Field draft={draft} onChange={change} name="panel" required /></label>
            <label className="form-wide">交易日历<Field draft={draft} onChange={change} name="calendar" required /></label>
            <label className="form-wide">因子库扫描目录<Field draft={draft} onChange={change} name="library_root" required /></label>
          </div>
          <p className="form-hint">路径相对于仓库，或填写本机绝对路径。面板需包含显式标签日期。</p>
        </fieldset>
        <fieldset className="panel section-panel"><legend>实验窗口与参数</legend>
          <div className="form-grid">
            <label>训练开始<Field draft={draft} onChange={change} type="date" name="train_start" required /></label>
            <label>首轮准入截点<Field draft={draft} onChange={change} type="date" name="initial_as_of" required /></label>
            <label>策略验证开始<Field draft={draft} onChange={change} type="date" name="selection_start" required /></label>
            <label>策略验证结束<Field draft={draft} onChange={change} type="date" name="selection_end" required /></label>
            <label>留出期开始<Field draft={draft} onChange={change} type="date" name="holdout_start" required /></label>
            <label>留出期结束<Field draft={draft} onChange={change} type="date" name="holdout_end" required /></label>
            <label>准入窗口（交易日）<Field draft={draft} onChange={change} type="number" name="admission_sessions" min="6" max="5000" required /></label>
            <label>最少训练交易日<Field draft={draft} onChange={change} type="number" name="min_train_sessions" min="6" max="5000" required /></label>
            <label>维护间隔（自然日）<Field draft={draft} onChange={change} type="number" name="maintenance_days" min="1" max="3650" required /></label>
            <label>特征预热交易日<Field draft={draft} onChange={change} type="number" name="warmup_sessions" min="0" max="5000" required /></label>
            <label>单边交易成本率<Field draft={draft} onChange={change} type="number" step="any" name="cost_rate" min="0" max="1" required /></label>
            <label>Agent 数量<Field draft={draft} onChange={change} type="number" name="agents" min="1" max="12" required /></label>
            <label>每轮候选预算<Field draft={draft} onChange={change} type="number" name="candidates" min="1" max="100" required /></label>
            <label>搜索轮数<Field draft={draft} onChange={change} type="number" name="rounds" min="1" max="10" required /></label>
            <label className="form-wide">策略配置（beta,gamma,nlong,nshort，最多三组，以分号分隔）<Field draft={draft} onChange={change} name="configs" required /></label>
          </div>
        </fieldset>
        <div className="form-actions"><button className="button primary" disabled={!settings || saving}>保存设置</button><p id="settings-status" role="status" data-invalid={invalid || Boolean(error)}>{message || error}</p></div>
      </form>
    </section>
  );
}
