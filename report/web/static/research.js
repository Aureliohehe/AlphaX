"use strict";

const grades = {strict:"已准入", research:"研究候选", unadmitted:"未准入", unknown:"缺少评估"};
const settingsForm = $("#settings-form");
const settingsStatus = $("#settings-status");
const minerForm = $("#miner-form");
const backtestForm = $("#backtest-form");
const resultPicker = $("#backtest-results");
const backtestCanvas = $("#backtest-chart");
const dayPicker = $("#backtest-day");
const researchJobs = {miner:null, backtest:null};
let settings = null;
let settingsLoading = false;
let factorCatalog = [];
let factorPage = 0;
let factorLoading = false;
let nextFactorSource = null;
let backtestData = null;
let currentResultRequest = 0;
let researchSubmitting = false;
let latestBacktest = null;
const pageSize = 30;

function setMessage(element, message, error = false) {
  if (element.textContent !== message) element.textContent = message;
  element.dataset.invalid = String(error);
}

function postJson(path, payload) {
  return request(path, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
}

function updateLaunchControls() {
  const busy = researchSubmitting || currentJobs.some((job) => job.status === "running");
  minerForm.querySelector("button").disabled = busy || !settings?.has_key;
  backtestForm.querySelector("button").disabled = busy || !availableCandidates();
}

function showSettings(payload) {
  settings = payload;
  for (const [name, value] of Object.entries(payload.values)) {
    settingsForm.elements.namedItem(name).value = value;
  }
  const values = payload.values;
  $("#key-state").textContent = payload.has_key ?
    "已配置密钥。通过界面设置的密钥仅在当前服务会话保存。" :
    "请填写密钥，或在启动服务时设置对应的环境变量。";
  $("#miner-context").textContent = payload.has_key ?
    values.model + " · " + values.agents + " 个 Agent · " + values.rounds +
    " 轮 · 每轮最多 " + values.candidates + " 个候选 · 截点 " + values.initial_as_of :
    "请先在系统设置中配置 LLM API 密钥。";
  updateBacktestContext();
}

async function loadSettings() {
  if (settings || settingsLoading) return;
  settingsLoading = true;
  try {
    showSettings(await request("/api/settings"));
  } catch (error) {
    setMessage(settingsStatus, error.message, true);
    setMessage($("#miner-context"), error.message, true);
  } finally {
    settingsLoading = false;
    settingsForm.querySelector("button[type=submit], .form-actions button.primary").disabled = !settings;
  }
}

async function saveSettings(clearKey = false) {
  if (!settings || !settingsForm.reportValidity()) return;
  const values = {};
  for (const [name, previous] of Object.entries(settings.values)) {
    const input = settingsForm.elements.namedItem(name);
    values[name] = typeof previous === "number" ? input.valueAsNumber : input.value;
  }
  const body = {values};
  const keyInput = settingsForm.elements.namedItem("api_key");
  if (clearKey || keyInput.value) body.api_key = clearKey ? "" : keyInput.value;
  const save = settingsForm.querySelector(".button.primary");
  const clear = $("#remove-key");
  save.disabled = clear.disabled = true;
  try {
    showSettings(await postJson("/api/settings", body));
    keyInput.value = "";
    setMessage(settingsStatus, clearKey ? "已清除界面密钥。服务环境中的密钥仍可使用。" : "设置已保存。");
  } catch (error) {
    setMessage(settingsStatus, error.message, true);
  } finally {
    save.disabled = clear.disabled = false;
  }
}

function syncOptions(select, options) {
  const value = select.value;
  select.replaceChildren(select.options[0], ...options.map(([label, key]) => new Option(label, key)));
  if (options.some(([,key]) => key === value)) select.value = value;
}

function matchFactors() {
  const search = $("#factor-search").value.trim().toLocaleLowerCase();
  const source = $("#factor-source").value;
  const grade = $("#factor-grade").value;
  const matches = [];
  for (const library of factorCatalog) {
    if (source && library.id !== source) continue;
    for (const factor of library.factors) {
      if (grade && factor.grade !== grade) continue;
      if (![factor.id, factor.expression, factor.hypothesis].some(
        (value) => value.toLocaleLowerCase().includes(search))) continue;
      matches.push([factor.id, grades[factor.grade], factor.expression,
        factor.hypothesis || "—", format(factor.rank_ic, 4), library.name]);
    }
  }
  return matches;
}

function showFactors() {
  const matches = matchFactors();
  const pages = Math.max(1, Math.ceil(matches.length / pageSize));
  factorPage = Math.min(factorPage, pages - 1);
  setMessage($("#factor-state"), "找到 " + matches.length.toLocaleString("zh-CN") + " 条因子记录");
  $("#factor-table").innerHTML = matches.length ? table(["因子", "质量", "表达式", "研究假设", "Rank IC", "来源"],
    matches.slice(factorPage * pageSize, (factorPage + 1) * pageSize)) :
    '<p class="empty">没有匹配的因子。检查扫描目录或调整筛选条件。</p>';
  $("#factor-pages").hidden = pages <= 1;
  $("#factor-page").textContent = (factorPage + 1) + " / " + pages;
  $("#factors-prev").disabled = factorPage === 0;
  $("#factors-next").disabled = factorPage >= pages - 1;
}

async function loadFactors() {
  if (factorLoading) return;
  factorLoading = true;
  $("#reload-factors").disabled = true;
  setMessage($("#factor-state"), "正在读取因子产物…");
  try {
    factorCatalog = await request("/api/factors");
    const options = factorCatalog.map((library) => [library.name + " · " + library.factors.length + " 个", library.id]);
    syncOptions($("#factor-source"), options);
    syncOptions(backtestForm.elements.namedItem("library"), options);
    if (nextFactorSource) {
      const source = factorCatalog.find((library) => library.id === nextFactorSource);
      if (source) {
        $("#factor-source").value = source.id;
        $("#factor-search").value = $("#factor-grade").value = "";
      }
      nextFactorSource = null;
    }
    showFactors();
    updateBacktestContext();
  } catch (error) {
    setMessage($("#factor-state"), error.message, true);
    setMessage(backtestForm.querySelector("[data-error]"), error.message, true);
  } finally {
    factorLoading = false;
    $("#reload-factors").disabled = false;
  }
}

function availableCandidates() {
  const source = factorCatalog.find(({id}) => id === backtestForm.elements.namedItem("library").value);
  const grade = backtestForm.elements.namedItem("grade").value;
  return source?.factors.filter((factor) => !grade || factor.grade === grade).length ?? 0;
}

function updateBacktestContext() {
  const values = settings?.values;
  $("#backtest-context").textContent = values ?
    "符合范围 " + availableCandidates() + " 个 · 验证 " + values.selection_start + " 至 " + values.selection_end +
    " · 留出 " + values.holdout_start + " 至 " + values.holdout_end : "";
  updateLaunchControls();
}

async function launchResearch(event, kind) {
  event.preventDefault();
  const form = event.currentTarget;
  const errorBox = form.querySelector("[data-error]");
  if (researchSubmitting || currentJobs.some((job) => job.status === "running")) return;
  researchSubmitting = true;
  updateLaunchControls();
  setMessage(errorBox, "");
  try {
    const result = await postJson("/api/jobs", {kind, ...Object.fromEntries(new FormData(form))});
    researchJobs[kind] = result.id;
    await refreshJobs();
  } catch (error) {
    setMessage(errorBox, error.message, true);
  } finally {
    researchSubmitting = false;
    updateLaunchControls();
  }
}

function showResearchJobs() {
  updateLaunchControls();
  for (const kind of Object.keys(researchJobs)) {
    const monitor = document.querySelector('[data-monitor="' + kind + '"]');
    const jobs = currentJobs.filter((job) => job.kind === kind);
    const picker = monitor.querySelector("[data-job]");
    if (jobs.map(({id}) => id).join() !== [...picker.options].map(({value}) => value).join()) {
      picker.replaceChildren(...jobs.map((job) =>
        new Option(new Date(job.started * 1000).toLocaleString("zh-CN") + " · " + job.id, job.id)));
    }
    picker.parentElement.hidden = jobs.length === 0;
    const job = jobs.find(({id}) => id === researchJobs[kind]) ?? jobs[0];
    if (!job) continue;
    picker.value = job.id;
    const elapsed = Math.max(0, Math.round((job.finished ?? Date.now() / 1000) - job.started));
    setMessage(monitor.querySelector("[data-status]"), statusNames[job.status] + " · " + elapsed + " 秒",
      job.status === "failed");
    const log = monitor.querySelector("[data-log]");
    const lines = [...(job.progress ?? []), ...job.lines];
    if (job.output && job.status === "passed") lines.push("产物目录 " + job.output);
    if (job.exit_code !== null) lines.push("退出码 " + job.exit_code);
    const value = lines.join("\n") || "正在准备任务…";
    if (log.textContent !== value) {
      const follow = log.scrollHeight - log.clientHeight - log.scrollTop < 25;
      log.textContent = value;
      if (follow) log.scrollTop = log.scrollHeight;
    }
    if (kind === "miner") {
      const link = monitor.querySelector("[data-library]");
      link.hidden = job.status !== "passed";
      if (!link.hidden) link.dataset.source = job.library;
    }
    if (page === "backtest" && kind === "backtest" && jobs[0].status === "passed" && latestBacktest !== jobs[0].id) {
      latestBacktest = jobs[0].id;
      loadBacktests(latestBacktest);
    }
  }
}

async function loadBacktests(preferred) {
  try {
    const ids = await request("/api/backtests");
    syncOptions(resultPicker, ids.map((id) => ["实验 " + id, id]));
    if ((preferred || !resultPicker.value) && ids.length) {
      resultPicker.value = ids.includes(preferred) ? preferred : ids[0];
      await showBacktest();
    }
  } catch (error) {
    setMessage($("#backtest-result-status"), error.message, true);
  }
}

async function showBacktest() {
  const rid = ++currentResultRequest;
  const status = $("#backtest-result-status");
  const report = $("#backtest-report");
  report.hidden = true;
  const id = resultPicker.value;
  if (!id) return setMessage(status, "选择已完成的回测查看结果。");
  setMessage(status, "正在读取回测结果…");
  try {
    const result = await request("/api/backtest?id=" + encodeURIComponent(id));
    if (rid !== currentResultRequest) return;
    backtestData = result;
    const metrics = result.metrics;
    $("#backtest-metrics").innerHTML = [
      ["累计净收益", pct(metrics.cumulative_return)], ["年化净收益", pct(metrics.annualized_return)],
      ["Sharpe", format(metrics.sharpe)], ["最大回撤", pct(metrics.max_drawdown)],
    ].map(([label, value]) => '<article class="metric"><span class="metric-label">' + label +
      "</span><strong>" + escapeHtml(value) + "</strong></article>").join("");
    const config = result.selected_config;
    $("#backtest-strategy").textContent = config ?
      "冻结策略 · β " + config.beta + " · γ " + config.gamma + " · 多头 " + config.nlong +
      " · 空头 " + config.nshort + " · 持仓日 " + result.invested_days :
      "验证期无可用策略配置，留出期持有现金。";
    dayPicker.max = Math.max(0, result.series.length - 1);
    dayPicker.value = dayPicker.max;
    const source = result.source?.name ? " · " + result.source.name + " · " + result.source.count + " 个因子" : "";
    setMessage(status, "留出期 " + result.series.length + " 个交易日 · 实验 " + id + source);
    report.hidden = false;
    drawBacktest();
  } catch (error) {
    if (rid === currentResultRequest) setMessage(status, error.message, true);
  }
}

function drawBacktest() {
  if (!backtestData?.series.length || backtestCanvas.clientWidth === 0 || $("#backtest-report").hidden) return;
  const points = backtestData.series;
  const selected = Number(dayPicker.value);
  const point = points[selected];
  $("#backtest-day-values").textContent = point.date + " · 策略 " + format(point.nav, 4) +
    " · 参考 " + format(point.benchmark, 4);
  const width = backtestCanvas.clientWidth;
  const height = backtestCanvas.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  backtestCanvas.width = Math.round(width * ratio);
  backtestCanvas.height = Math.round(height * ratio);
  const ctx = backtestCanvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const left = 57, right = width - 16, top = 14, bottom = height - 27;
  const values = points.flatMap(({nav, benchmark}) => [nav, benchmark]);
  const min = Math.min(0.995, ...values), max = Math.max(1.005, ...values);
  const x = (index) => left + (index + 1) / points.length * (right - left);
  const y = (value) => bottom - (value - min) / (max - min) * (bottom - top);
  ctx.font = "12px system-ui";
  ctx.fillStyle = "#58647a";
  ctx.strokeStyle = "#e7e9f1";
  for (let tick = 0; tick <= 4; tick++) {
    const value = min + (max - min) * tick / 4;
    const ordinate = y(value);
    ctx.beginPath();
    ctx.moveTo(left, ordinate);
    ctx.lineTo(right, ordinate);
    ctx.stroke();
    ctx.fillText(format(value, 3), 4, ordinate + 4);
  }
  ctx.fillText(points[0].date, left, height - 6);
  const last = points.at(-1).date;
  ctx.fillText(last, Math.max(left, right - ctx.measureText(last).width), height - 6);
  for (const [key, color] of [["benchmark","#54827e"], ["nav","#6253e7"]]) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(left, y(1));
    points.forEach((row, index) => ctx.lineTo(x(index), y(row[key])));
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x(selected), y(point[key]), 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

function openResearchPage() {
  if (["settings", "miner", "backtest"].includes(page)) loadSettings();
  if (["factors", "backtest"].includes(page)) loadFactors();
  if (page === "backtest") loadBacktests();
}

settingsForm.addEventListener("submit", (event) => { event.preventDefault(); saveSettings(); });
$("#remove-key").addEventListener("click", () => saveSettings(true));
minerForm.addEventListener("submit", (event) => launchResearch(event, "miner"));
backtestForm.addEventListener("submit", (event) => launchResearch(event, "backtest"));
backtestForm.addEventListener("change", updateBacktestContext);
$("[data-library]").addEventListener("click", (event) => { nextFactorSource = event.currentTarget.dataset.source; });
for (const select of all("[data-job]")) {
  select.addEventListener("change", () => {
    researchJobs[select.closest("[data-monitor]").dataset.monitor] = select.value;
    showResearchJobs();
  });
}
$("#reload-factors").addEventListener("click", loadFactors);
for (const id of ["#factor-search", "#factor-source", "#factor-grade"]) {
  $(id).addEventListener("input", () => { factorPage = 0; showFactors(); });
}
$("#factors-prev").addEventListener("click", () => { factorPage--; showFactors(); });
$("#factors-next").addEventListener("click", () => { factorPage++; showFactors(); });
resultPicker.addEventListener("change", showBacktest);
dayPicker.addEventListener("input", drawBacktest);
new ResizeObserver(drawBacktest).observe(backtestCanvas);
window.addEventListener("hashchange", openResearchPage);
window.addEventListener("alphax:jobs", showResearchJobs);
openResearchPage();
