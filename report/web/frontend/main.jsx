import {useCallback, useEffect, useLayoutEffect, useRef, useState} from "react";
import {createRoot} from "react-dom/client";
import {pages, postJson, request} from "./shared";
import Overview from "./overview";
import Workflow from "./workflow";
import Results from "./results";
import Runs from "./runs";
import Settings from "./settings";
import Miner from "./miner";
import Factors from "./factors";
import Backtest from "./backtest";

const icons = {
  overview: "M2 2h5v5H2zM9 2h5v5H9zM2 9h5v5H2zM9 9h5v5H9z",
  miner: "M2 8a6 6 0 1 1 2 4.5M2 4v4h4",
  factors: "M3 2h10v12H3zM5 5h6M5 8h6M5 11h4",
  backtest: "M2 2v12h12M4 10l3-3 3 2 4-5",
  workflow: "M2 8a6 6 0 1 1 2 4.5M2 4v4h4",
  results: "M2 2v12h12M5 12V8m4 4V5m4 7V3",
  runs: "M2 3h12v10H2zM4 6l2 2-2 2m4 0h3",
  settings: "M8 2v2m0 8v2M2 8h2m8 0h2M4 4l1 1m6 6 1 1m0-8-1 1m-6 6-1 1M11 8a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z",
};

function readPage() {
  const destination = location.hash.slice(1);
  return Object.hasOwn(pages, destination) ? destination : "overview";
}

function useJobs(page) {
  const [jobs, setJobs] = useState([]);
  const [jobError, setJobError] = useState("");
  const [posting, setPosting] = useState(false);
  const [selected, setSelected] = useState({});
  const pending = useRef(null);
  const submitting = useRef(false);
  const refresh = useCallback(() => {
    if (pending.current) return pending.current;
    pending.current = request("/api/jobs").then((result) => {
      setJobs(result);
      setJobError("");
    }).catch((error) => setJobError(error.message)).finally(() => { pending.current = null; });
    return pending.current;
  }, []);

  useEffect(() => {
    if (!["runs", "miner", "backtest"].includes(page)) return;
    refresh();
    const timer = setInterval(refresh, 1100);
    return () => clearInterval(timer);
  }, [page, refresh]);

  const busy = posting || jobs.some(({status}) => status === "running");
  function select(scope, id) {
    setSelected((previous) => ({...previous, [scope]: id}));
  }
  async function launch(scope, payload) {
    if (submitting.current || busy) return;
    submitting.current = true;
    setPosting(true);
    try {
      const {id} = await postJson("/api/jobs", payload);
      select(scope, id);
      // If polling started before the POST completed, fetch again after it settles.
      await pending.current;
      await refresh();
    } finally {
      submitting.current = false;
      setPosting(false);
    }
  }
  return {jobs, jobError, selected, select, launch, busy};
}

function App() {
  const [page, setPage] = useState(readPage);
  const [dashboard, setDashboard] = useState({});
  const [connectionError, setConnectionError] = useState(false);
  const [settings, setSettings] = useState(null);
  const [settingsError, setSettingsError] = useState("");
  const [catalog, setCatalog] = useState([]);
  const [catalogLoading, setCatalogLoading] = useState(false);
  const [catalogError, setCatalogError] = useState("");
  const [factorSource, setFactorSource] = useState(null);
  const catalogRequest = useRef(null);
  const settingsRequest = useRef(null);
  const main = useRef(null);
  const jobState = useJobs(page);

  useEffect(() => {
    function navigate() {
      setPage(readPage());
      window.scrollTo(0, 0);
      main.current.focus({preventScroll: true});
    }
    window.addEventListener("hashchange", navigate);
    request("/api/dashboard").then(setDashboard).catch(() => setConnectionError(true));
    return () => window.removeEventListener("hashchange", navigate);
  }, []);
  useEffect(() => { document.title = "AlphaX · " + pages[page]; }, [page]);

  useEffect(() => {
    if (settings || settingsRequest.current || !["settings", "miner", "backtest"].includes(page)) return;
    settingsRequest.current = request("/api/settings").then((value) => {
      setSettings(value);
      setSettingsError("");
    }).catch((error) => setSettingsError(error.message))
      .finally(() => { settingsRequest.current = null; });
  }, [page, settings]);

  const loadFactors = useCallback(() => {
    if (catalogRequest.current) return catalogRequest.current;
    setCatalogLoading(true);
    setCatalogError("");
    catalogRequest.current = request("/api/factors").then(setCatalog)
      .catch((error) => setCatalogError(error.message))
      .finally(() => { catalogRequest.current = null; setCatalogLoading(false); });
    return catalogRequest.current;
  }, []);

  useEffect(() => {
    if (page === "factors" || page === "backtest") loadFactors();
  }, [page, loadFactors]);

  const [theme, setTheme] = useState(() => {
    try {
      const saved = localStorage.getItem("alphax-theme");
      if (["light", "dark"].includes(saved)) return saved;
    } catch {
      // The system preference is available when browser storage is disabled.
    }
    return "system";
  });
  useLayoutEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem("alphax-theme", theme); } catch { /* Storage is optional. */ }
  }, [theme]);

  return (
    <div className="app">
      <aside className="sidebar">
        <a className="brand" href="#overview" aria-label="AlphaX 研究总览">
          <span className="brand-icon" aria-hidden="true">α</span>
          <span>Alpha<span className="brand-x">X</span><small>量化研究工作台</small></span>
        </a>
        <nav aria-label="工作台导航">
          <p className="nav-caption">工作区</p>
          {Object.entries(pages).map(([id, title]) => (
            <a key={id} href={"#" + id} data-page={id} aria-current={page === id ? "page" : undefined}>
              <svg className="icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.25" aria-hidden="true"><path d={icons[id]} /></svg>
              {title}
            </a>
          ))}
        </nav>
        <div className="sidebar-footer"><span aria-hidden="true" /><div>本地工作区<small>读取本机实验产物</small></div></div>
      </aside>
      <main ref={main} tabIndex={-1} aria-label={pages[page]}>
        <header className="topbar">
          <span className="breadcrumb">工作区 <span className="breadcrumb-divider">/</span> <strong id="breadcrumb">{pages[page]}</strong></span>
          <div className="topbar-actions">
            <label className="theme-picker"><span>主题</span><select value={theme} onChange={(event) => setTheme(event.target.value)}>
              <option value="system">跟随系统</option><option value="light">浅色</option><option value="dark">深色</option>
            </select></label>
            <a className="button primary" href="#runs"><span className="play" aria-hidden="true" />运行验证</a>
          </div>
        </header>
        {connectionError && <div className="notification" role="status">无法读取本地实验数据。请确认 AlphaX 服务正在运行。</div>}
        <div className="content">
          <div hidden={page !== "overview"}><Overview data={dashboard} /></div>
          <div hidden={page !== "workflow"}><Workflow data={dashboard} /></div>
          <div hidden={page !== "results"}><Results data={dashboard} /></div>
          <div hidden={page !== "runs"}><Runs {...jobState} /></div>
          <div hidden={page !== "settings"}><Settings settings={settings} error={settingsError} onSave={setSettings} /></div>
          <div hidden={page !== "miner"}><Miner {...jobState} settings={settings} settingsError={settingsError}
            onLibrary={(id) => setFactorSource({id})} /></div>
          <div hidden={page !== "factors"}><Factors catalog={catalog} loading={catalogLoading} error={catalogError}
            reload={loadFactors} requestedSource={factorSource} /></div>
          <div hidden={page !== "backtest"}><Backtest {...jobState} active={page === "backtest"}
            settings={settings} catalog={catalog} catalogError={catalogError} /></div>
        </div>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
