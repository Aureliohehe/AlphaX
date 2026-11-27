"use strict";

const $ = (selector) => document.querySelector(selector);
const all = (selector) => [...document.querySelectorAll(selector)];
const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (c) =>
  ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"})[c]);
const format = (value, digits = 2) => Number.isFinite(value) ? value.toFixed(digits) : "—";
const pct = (value) => Number.isFinite(value) ? format(value * 100) + "%" : "—";
const integer = (value) => Number.isFinite(value) ? value.toLocaleString("zh-CN") : "—";
const delta = (value) => Number.isFinite(value) ? (value >= 0 ? "+" : "") + format(value * 100) : "—";
const missing = '<p class="empty">本机缺少对应的实验产物。</p>';
const pages = {overview:"研究总览", workflow:"Agent loop", results:"实验对照", runs:"运行中心",
  settings:"系统设置", miner:"因子挖掘", factors:"因子库", backtest:"独立回测"};
let data = {};
let page = "overview";
let range = "all";
let pointer = null;
let series = [];
let plot = null;
let activeJob = null;
let posting = false;
let jobsRefreshing = false;
let currentJobs = [];
const chart = $("#nav-chart");
const names = {standard:"标准成本", stress:"加压成本", benchmark:"匹配指数代理"};
const colors = {standard:"#6253e7", stress:"#0b9471", benchmark:"#78869c"};
const enabled = new Set(Object.keys(names));
const conditions = {demo:"离线全链路示例", smoke:"核心冒烟测试", preflight:"美股流水线预检",
  miner:"因子挖掘", backtest:"独立回测"};
const statusNames = {running:"运行中", passed:"已通过", failed:"运行失败"};

function navigate() {
  const destination = location.hash.slice(1);
  page = Object.hasOwn(pages, destination) ? destination : "overview";
  all("[data-view]").forEach((view) => { view.hidden = view.id !== page; });
  all("[data-page]").forEach((link) => {
    if (link.dataset.page === page) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  $("#breadcrumb").textContent = pages[page];
  document.title = "AlphaX · " + pages[page];
  window.scrollTo(0, 0);
  if (page === "overview") requestAnimationFrame(drawChart);
  if (["runs", "miner", "backtest"].includes(page)) refreshJobs();
}

function table(headers, rows) {
  return '<div class="table-wrap"><table><thead><tr>' +
    headers.map((cell) => "<th scope='col'>" + escapeHtml(cell) + "</th>").join("") +
    "</tr></thead><tbody>" + rows.map((row) =>
      "<tr><th scope='row'>" + escapeHtml(row[0]) + "</th>" +
      row.slice(1).map((cell) => "<td>" + escapeHtml(cell) + "</td>").join("") + "</tr>").join("") +
    "</tbody></table></div>";
}

function populate() {
  const holdout = data.holdout;
  const standard = holdout?.trials.find((trial) => trial.cost_multiplier === 1);
  const stress = holdout?.trials.find((trial) => trial.cost_multiplier === 2);
  const cards = [
    ["累计净收益", pct(standard?.cumulative_return), "标准成本", "good"],
    ["加压净收益", pct(stress?.cumulative_return), "双倍佣金与滑点", "good"],
    ["Sharpe", format(standard?.sharpe), "无风险利率为 0", ""],
    ["最大回撤", pct(standard?.max_drawdown), "策略门槛 −8.00%", ""],
  ];
  $("#metrics").innerHTML = cards.map(([label, number, hint, tone]) =>
    '<article class="metric"><span class="metric-label">' + label + '</span><strong class="' +
    (number.startsWith("-") ? "" : tone) + '">' + number + "</strong><small>" + hint + "</small></article>").join("");
  $("#holdout-table").innerHTML = standard && stress ? table(
    ["指标", "标准成本", "双倍佣金与滑点"], [
      ["累计净收益", pct(standard.cumulative_return), pct(stress.cumulative_return)],
      ["年化净收益", pct(standard.annualized_return), pct(stress.annualized_return)],
      ["Sharpe", format(standard.sharpe), format(stress.sharpe)],
      ["最大回撤", pct(standard.max_drawdown), pct(stress.max_drawdown)],
      ["2024 年净收益", pct(standard.calendar_year_returns?.["2024"]), pct(stress.calendar_year_returns?.["2024"])],
      ["2025 年净收益", pct(standard.calendar_year_returns?.["2025"]), pct(stress.calendar_year_returns?.["2025"])],
    ]) + '<p class="note warning">两档成本情景的最大回撤均超过 8% 准入门槛。</p>' + source(holdout.source) : missing;

  const pair = data.paired;
  if (pair) {
    const variant = pair.variant;
    const core = variant.results.core.costs;
    const augmented = variant.results.augmented.costs;
    $("#best").innerHTML = '<p class="best-factor">低波动平滑因子 · 120 日</p><div class="best-impact">' +
      delta(variant.return_delta["1"]) + " <small>百分点</small></div>" +
      '<p>标准成本净收益提升</p><div class="best-lines"><div><span>标准成本</span><strong>' +
      pct(core["1"].cumulative_return) + " → " + pct(augmented["1"].cumulative_return) +
      '</strong></div><div><span>加压成本</span><strong>' + pct(core["2"].cumulative_return) + " → " +
      pct(augmented["2"].cumulative_return) + '</strong></div></div>' +
      (pair.followup?.status === "rejected_on_frozen_evaluation"
        ? '<div class="note warning">2026 年冻结检验未通过</div>' : '<div class="note">已观察窗口的回顾性结果</div>');
    const fields = [
      ["累计净收益", (r) => r.cumulative_return],
      ["最大回撤", (r) => r.max_drawdown],
      ["2024 年净收益", (r) => r.calendar_year_returns?.["2024"]],
      ["2025 年净收益", (r) => r.calendar_year_returns?.["2025"]],
    ];
    const comparison = fields.map(([name, get]) => [name, pct(get(core["1"])), pct(get(augmented["1"])),
      pct(get(core["2"])), pct(get(augmented["2"]))]);
    comparison.push(["累计净收益增量", "—", delta(variant.return_delta["1"]) + " 个百分点",
      "—", delta(variant.return_delta["2"]) + " 个百分点"]);
    $("#comparison").innerHTML = "<p><code>" + escapeHtml(variant.expression) + "</code></p>" +
      table(["指标", "标准 · 基线", "标准 · 增强", "加压 · 基线", "加压 · 增强"], comparison) +
      '<p class="technical">相同评分样本 ' + integer(variant.signal_rows) + " 条 · 训练期 " +
      escapeHtml(pair.training.join(" 至 ")) + " · 配对窗口 " + escapeHtml(pair.window.join(" 至 ")) + "</p>" +
      (pair.followup ? '<p class="note warning">2026 年冻结检验相对基线的净收益差为 ' +
        delta(pair.followup.net_return_delta?.["1"]) + " 和 " + delta(pair.followup.net_return_delta?.["2"]) +
        " 个百分点，两档成本均未通过。</p>" : "") + source(pair.source);
  } else {
    $("#best").innerHTML = missing;
    $("#comparison").innerHTML = missing;
  }

  const search = data.search;
  const mechanism = data.mechanism;
  $("#rounds").innerHTML = (search ? '<p><span class="tag">美股历史实验</span> ' +
    integer(search.candidates) + " 项去重候选 · " + integer(search.strict_accepted.length) + " 项严格准入</p>" +
    table(["迭代", "提案数", "新增候选", "代表候选最差折 Rank ICIR"],
      search.rounds.map((r) => [r.round, r.requested, r.unique, format(r.best_worst_fold_rank_icir, 4)]))
    : missing) + (mechanism ? '<p class="note">沪深 300 机制优先实验进入研究池的候选数：日频 ' +
      integer(mechanism["1"]) + " · 周频 " + integer(mechanism["5"]) + " · 20 日频率 " +
      integer(mechanism["20"]) + "</p>" : "");
  series = holdout?.series ?? [];
  chart.hidden = !series.length;
  $("#chart-empty").hidden = Boolean(series.length);
  if (!series.length) $("#chart-caption").textContent = "请确认本机包含冻结回放产物。";
  drawChart();
}

function source(directory) {
  return '<p class="technical">本地产物 <code>' + escapeHtml(directory) + "</code></p>";
}

function visiblePoints() {
  return range === "all" ? series : series.filter((p) => p.date.startsWith(range));
}

function drawChart() {
  const points = visiblePoints();
  if (page !== "overview" || !points.length || !chart.clientWidth) return;
  const keys = [...enabled];
  const width = chart.clientWidth;
  const height = chart.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  const context = chart.getContext("2d");
  chart.width = Math.round(width * ratio);
  chart.height = Math.round(height * ratio);
  context.scale(ratio, ratio);

  const left = 43, top = 15, right = width - 10, bottom = height - 28;
  const values = points.flatMap((point) => keys.map((key) => point[key]));
  const low = Math.floor(Math.min(...values) * 20) / 20;
  const high = Math.ceil(Math.max(...values) * 20) / 20 || 1;
  const maxIndex = Math.max(1, points.length - 1);
  const x = (i) => left + i * (right - left) / maxIndex;
  const y = (value) => bottom - (value - low) / (high - low || 1) * (bottom - top);
  plot = {points, left, width: right - left};
  context.font = "12px system-ui, sans-serif";
  context.fillStyle = "#58647a";
  context.strokeStyle = "#e9ebf3";
  context.textAlign = "right";
  context.lineWidth = 1;
  for (let tick = 0; tick <= 4; tick++) {
    const value = low + (high - low) * tick / 4;
    const ordinate = y(value);
    context.beginPath();
    context.moveTo(left, ordinate);
    context.lineTo(right, ordinate);
    context.stroke();
    context.fillText(value.toFixed(2), left - 7, ordinate + 4);
  }
  const tickCount = width < 400 ? 3 : 5;
  for (let tick = 0; tick < tickCount; tick++) {
    const index = Math.round(maxIndex * tick / (tickCount - 1));
    context.textAlign = tick === 0 ? "left" : tick === tickCount - 1 ? "right" : "center";
    context.fillText(points[index].date.slice(0, 7), x(index), height - 8);
  }
  function path(key) {
    context.moveTo(x(0), y(points[0][key]));
    for (let i = 1; i < points.length; i++) context.lineTo(x(i), y(points[i][key]));
  }
  if (enabled.has("standard")) {
    const fill = context.createLinearGradient(0, top, 0, bottom);
    fill.addColorStop(0, "#6050dd2b");
    fill.addColorStop(1, "#6050dd04");
    context.beginPath();
    path("standard");
    context.lineTo(right, bottom);
    context.lineTo(left, bottom);
    context.closePath();
    context.fillStyle = fill;
    context.fill();
  }
  for (const key of keys.toReversed()) {
    context.strokeStyle = colors[key];
    context.lineWidth = key === "standard" ? 2.15 : 1.55;
    context.setLineDash(key === "benchmark" ? [4, 4] : []);
    context.beginPath();
    path(key);
    context.stroke();
  }
  context.setLineDash([]);
  const index = pointer === null ? maxIndex : Math.max(0, Math.min(maxIndex, pointer));
  const point = points[index];
  const reading = keys.map((key) => names[key] + " " + format(point[key], 3)).join(" · ");
  $("#chart-caption").textContent = point.date + " · " + reading;
  $("#chart-hint").hidden = pointer === null;
  if (pointer !== null) {
    const abscissa = x(index);
    context.strokeStyle = "#8179b7";
    context.lineWidth = 1;
    context.setLineDash([3, 3]);
    context.beginPath();
    context.moveTo(abscissa, top);
    context.lineTo(abscissa, bottom);
    context.stroke();
    context.setLineDash([]);
    for (const key of keys) {
      context.fillStyle = colors[key];
      context.beginPath();
      context.arc(abscissa, y(point[key]), 3, 0, 2 * Math.PI);
      context.fill();
    }
    $("#chart-hint").textContent = point.date + "\n" +
      keys.map((key) => names[key] + " " + format(point[key], 3)).join("\n");
  }
}

async function request(path, options) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error ?? "本地服务请求失败。");
  return payload;
}

async function refreshJobs() {
  if (jobsRefreshing) return;
  jobsRefreshing = true;
  try {
    currentJobs = await request("/api/jobs");
    if (!currentJobs.some((job) => job.id === activeJob)) activeJob = currentJobs[0]?.id;
    const select = $("#job-select");
    const known = [...select.options].map((option) => option.value).join(",");
    if (known !== currentJobs.map((job) => job.id).join(",")) {
      select.replaceChildren(...currentJobs.map((job) => {
        const label = conditions[job.kind] + " · " + new Date(job.started * 1000).toLocaleTimeString("zh-CN");
        return new Option(label, job.id);
      }));
    }
    $("#job-picker").hidden = currentJobs.length === 0;
    if (activeJob) select.value = activeJob;
    showJob();
    window.dispatchEvent(new Event("alphax:jobs"));
  } catch (error) {
    $("#job-status").textContent = error.message;
  } finally {
    jobsRefreshing = false;
  }
}

function showJob() {
  const job = currentJobs.find((entry) => entry.id === activeJob);
  $("#launch button").disabled = posting || currentJobs.some((entry) => entry.status === "running");
  if (!job) return;
  const status = conditions[job.kind] + " · " + statusNames[job.status];
  if ($("#job-status").textContent !== status) $("#job-status").textContent = status;
  const log = $("#job-log");
  const fullLog = "$ " + job.command + "\n\n" + job.lines.join("\n") +
    (job.exit_code === null ? "" : "\n\n退出码 " + job.exit_code);
  if (log.textContent !== fullLog) {
    const follow = log.scrollHeight - log.scrollTop - log.clientHeight < 24;
    log.textContent = fullLog;
    if (follow) log.scrollTop = log.scrollHeight;
  }
  $("#job-output").textContent = job.output && job.status === "passed" ? "产物目录 " + job.output : "";
}

const paths = [
  "AC=quant-method/alphacrafter", "ML=quant-method/machine-learn", 'D="$ML/data/qlib"',
];
const recipes = {
  miner: {
    hint:"使用 LLM 环境与本机的历史面板，先设置 DEEPSEEK_API_KEY。输出目录需要是新目录。",
    lines: [
      'python "$AC/miner_lite.py" ' + "\\",
      '  --input "$D/alphacrafter_feedback_search_v1/features.csv.gz" ' + "\\",
      '  --calendar "$ML/data/processed/sp500_xnys_xcal_4_13_3.csv" ' + "\\",
      '  --require-explicit-label-dates --as-of 2022-12-30 ' + "\\",
      '  --train-start 2016-01-04 --train-end 2021-12-30 ' + "\\",
      '  --validation-start 2022-01-03 --validation-end 2022-12-30 ' + "\\",
      '  --backtest-start 2023-01-03 --backtest-end 2023-12-29 ' + "\\",
      '  --llm-model deepseek-chat --llm-agents 3 --llm-orchestrator langgraph ' + "\\",
      '  --search-rounds 3 --search-budget 30 --search-batch-size 10 --search-folds 3 ' + "\\",
      '  --output-dir "$D/web_miner_run"',
    ],
  },
  paired: {
    hint:"使用包含 XGBoost 和 HDF5 依赖的环境，以及已完成的先前 Miner 产物。",
    lines: [
      'python "$AC/run_medium_frequency.py" ' + "\\",
      '  --miner-dir "$D/llm_rank_aware_Ut07VD/miner" ' + "\\",
      '  --evaluate-strategies --compare-weekly-xgboost-additions ' + "\\",
      '  --weekly-xgboost-augmentation-method refit ' + "\\",
      '  --output-dir "$D/web_weekly_pairs"',
    ],
  },
  feedback: {
    hint:"读取上一步的已完成配对结果。使用 LLM 环境，先设置 DEEPSEEK_API_KEY。",
    lines: [
      'python "$AC/miner_lite.py" ' + "\\",
      '  --input "$D/llm_complementary_factors_nJXouK/miner_panel.csv.gz" ' + "\\",
      '  --as-of 2023-12-31 --train-start 2019-01-01 --train-end 2020-12-31 ' + "\\",
      '  --validation-start 2021-01-01 --validation-end 2021-12-31 ' + "\\",
      '  --backtest-start 2024-01-01 --backtest-end 2025-12-31 ' + "\\",
      '  --llm-model deepseek-chat --llm-orchestrator langgraph --llm-agents 3 ' + "\\",
      '  --llm-count 12 --llm-xgboost-complement ' + "\\",
      '  --llm-weekly-xgboost-feedback "$D/web_weekly_pairs/replay_summary.json" ' + "\\",
      '  --output-dir "$D/web_weekly_next"',
    ],
  },
};

function showRecipe() {
  const recipe = recipes[$("#recipe").value];
  $("#recipe-hint").textContent = recipe.hint;
  $("#recipe-code").textContent = [...paths, "", ...recipe.lines].join("\n");
  $("#copy-state").textContent = "";
}

window.addEventListener("hashchange", navigate);
new ResizeObserver(drawChart).observe(chart);
all("[data-range]").forEach((button) => button.addEventListener("click", () => {
  range = button.dataset.range;
  pointer = null;
  all("[data-range]").forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
  drawChart();
}));
all("[data-series]").forEach((button) => button.addEventListener("click", () => {
  const key = button.dataset.series;
  if (enabled.has(key) && enabled.size === 1) return;
  if (enabled.has(key)) enabled.delete(key);
  else enabled.add(key);
  button.setAttribute("aria-pressed", String(enabled.has(key)));
  drawChart();
}));
chart.addEventListener("pointermove", (event) => {
  if (!plot) return;
  pointer = Math.round((event.offsetX - plot.left) / plot.width * (plot.points.length - 1));
  drawChart();
});
chart.addEventListener("pointerleave", () => { pointer = null; drawChart(); });
chart.tabIndex = 0;
chart.addEventListener("keydown", (event) => {
  if (!plot || !["ArrowLeft", "ArrowRight"].includes(event.key)) return;
  event.preventDefault();
  const last = plot.points.length - 1;
  pointer = Math.min(last, Math.max(0, (pointer ?? last) + (event.key === "ArrowLeft" ? -1 : 1)));
  drawChart();
});
chart.addEventListener("blur", () => { pointer = null; drawChart(); });

$("#launch").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (posting || currentJobs.some((entry) => entry.status === "running")) return;
  const errorBox = $("#launch-error");
  errorBox.hidden = true;
  posting = true;
  showJob();
  const kind = new FormData(event.currentTarget).get("kind");
  try {
    const result = await request("/api/jobs", {
      method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({kind}),
    });
    activeJob = result.id;
    await refreshJobs();
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
  } finally {
    posting = false;
    showJob();
  }
});
$("#job-select").addEventListener("change", (event) => { activeJob = event.target.value; showJob(); });
$("#recipe").addEventListener("change", showRecipe);
$("#copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("#recipe-code").textContent);
    $("#copy-state").textContent = "已复制";
  } catch {
    const selection = document.createRange();
    selection.selectNodeContents($("#recipe-code"));
    window.getSelection().removeAllRanges();
    window.getSelection().addRange(selection);
    $("#copy-state").textContent = "命令已选中，可直接复制";
  }
});
setInterval(() => { if (["runs", "miner", "backtest"].includes(page)) refreshJobs(); }, 1100);

navigate();
showRecipe();
request("/api/dashboard").then((result) => {
  data = result;
  populate();
}).catch(() => {
  $("#connection").textContent = "无法读取本地实验数据。请确认 AlphaX 服务正在运行。";
  $("#connection").hidden = false;
  populate();
});
