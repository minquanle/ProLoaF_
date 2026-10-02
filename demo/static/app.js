const $ = (selector) => document.querySelector(selector);
const state = { meta: null, data: null, showOriginal: true };
const formatMW = (value) => `${Math.round(value).toLocaleString("vi-VN")} MW`;
const formatDate = (timestamp) => new Intl.DateTimeFormat("vi-VN", { weekday: "short", day: "2-digit", month: "2-digit", year: "numeric" }).format(new Date(timestamp));
const formatTime = (timestamp) => new Intl.DateTimeFormat("vi-VN", { hour: "2-digit", minute: "2-digit" }).format(new Date(timestamp));

async function fetchJSON(path) {
  const response = await fetch(path); const data = await response.json();
  if (!response.ok || data.error) throw new Error(data.error || "Không thể tải dữ liệu.");
  return data;
}

async function waitForModel() {
  try {
    const status = await fetchJSON("/api/status");
    if (!status.ready) { if (status.error) throw new Error(status.error); setTimeout(waitForModel, 900); return; }
    state.meta = status; $("#dot").parentElement.classList.add("ready"); $("#status-text").textContent = "Checkpoint sẵn sàng";
    $("#loading").classList.add("hidden"); $("#dashboard").classList.remove("hidden"); $("#summary").classList.remove("hidden");
    const slider = $("#origin-slider"); slider.max = status.origins - 1; slider.value = Math.min(1000, status.origins - 1);
    $("#c-value").textContent = `c = ${status.c.toFixed(6)}`;
    await loadSummary(); await loadForecast(Number(slider.value));
  } catch (error) {
    $("#loading").innerHTML = `<div><b>Không thể khởi động demo</b><p>${error.message}</p></div>`;
    $("#status-text").textContent = "Lỗi khi nạp dữ liệu";
  }
}

async function loadSummary() {
  const result = await fetchJSON("/api/summary");
  $("#summary-body").innerHTML = result.summary.map((row) => `<tr><td>${row.interval.includes("original") ? "LSTM gốc" : "LSTM calibrated"}</td><td>${Number(row.c).toFixed(6)}</td><td>${Number(row.PICP_percent).toFixed(2)}%</td><td>${Number(row.Sharpness).toFixed(6)}</td><td>${Number(row.RMSE).toFixed(6)}</td><td>${Number(row.MAE).toFixed(6)}</td></tr>`).join("");
}

async function loadForecast(origin) {
  try { state.data = await fetchJSON(`/api/forecast?origin=${origin}`); render(); }
  catch (error) { console.error(error); }
}

function render() {
  const { origin, origin_count, metrics, c, forecast } = state.data;
  $("#origin-number").textContent = `${origin + 1} / ${origin_count}`; $("#origin-slider").value = origin;
  $("#origin-day").textContent = formatDate(forecast.timestamps[0]); $("#origin-time").textContent = `Dự báo bắt đầu lúc ${formatTime(forecast.timestamps[0])}`;
  $("#rmse").textContent = formatMW(metrics.rmse); $("#mae").textContent = formatMW(metrics.mae);
  $("#picp").textContent = `${metrics.original.picp.toFixed(1)}% → ${metrics.calibrated.picp.toFixed(1)}%`;
  $("#sharpness").textContent = `${formatMW(metrics.original.sharpness)} → ${formatMW(metrics.calibrated.sharpness)}`;
  $("#c-value").textContent = `c = ${c.toFixed(6)}`;
  document.querySelectorAll(".legend-original").forEach((node) => node.style.opacity = state.showOriginal ? "1" : ".32");
  drawChart();
}

function drawChart() {
  const canvas = $("#forecast-chart"), wrap = canvas.parentElement, ratio = window.devicePixelRatio || 1;
  const width = wrap.clientWidth, height = wrap.clientHeight; canvas.width = width * ratio; canvas.height = height * ratio;
  const ctx = canvas.getContext("2d"); ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, width, height);
  const { history, forecast } = state.data, pad = { l: 62, r: 20, t: 22, b: 48 }, count = history.load_mw.length + forecast.actual_mw.length;
  const values = [...history.load_mw, ...forecast.actual_mw, ...forecast.mean_mw, ...forecast.calibrated_lower_mw, ...forecast.calibrated_upper_mw, ...(state.showOriginal ? [...forecast.original_lower_mw, ...forecast.original_upper_mw] : [])];
  const min = Math.floor(Math.min(...values) / 2000) * 2000, max = Math.ceil(Math.max(...values) / 2000) * 2000;
  const x = (i) => pad.l + (i / (count - 1)) * (width - pad.l - pad.r), y = (v) => pad.t + (1 - (v - min) / (max - min)) * (height - pad.t - pad.b);
  ctx.font = "11px system-ui"; ctx.fillStyle = "#7d91aa"; ctx.strokeStyle = "rgba(145,163,189,.16)"; ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) { const value = min + ((max - min) * i) / 5, yy = y(value); ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(width - pad.r, yy); ctx.stroke(); ctx.fillText(`${Math.round(value / 1000)}k`, 15, yy + 4); }
  const start = history.load_mw.length;
  const area = (lower, upper, color) => { ctx.beginPath(); upper.forEach((v, i) => i ? ctx.lineTo(x(start + i), y(v)) : ctx.moveTo(x(start + i), y(v))); for (let i = lower.length - 1; i >= 0; i--) ctx.lineTo(x(start + i), y(lower[i])); ctx.closePath(); ctx.fillStyle = color; ctx.fill(); };
  if (state.showOriginal) area(forecast.original_lower_mw, forecast.original_upper_mw, "rgba(251,113,133,.18)"); area(forecast.calibrated_lower_mw, forecast.calibrated_upper_mw, "rgba(79,140,255,.28)");
  const line = (series, color, dash = [], from = 0) => { ctx.beginPath(); series.forEach((v, i) => i ? ctx.lineTo(x(i + from), y(v)) : ctx.moveTo(x(i + from), y(v))); ctx.strokeStyle = color; ctx.lineWidth = 2.35; ctx.setLineDash(dash); ctx.stroke(); ctx.setLineDash([]); };
  line(history.load_mw, "#a6b2c5"); line(forecast.actual_mw, "#ffffff", [], start); line(forecast.mean_mw, "#4f8cff", [8, 5], start);
  ctx.beginPath(); ctx.moveTo(x(start), pad.t); ctx.lineTo(x(start), height - pad.b); ctx.strokeStyle = "rgba(94,234,212,.65)"; ctx.setLineDash([4, 5]); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = "#5eead4"; ctx.fillText("Bắt đầu dự báo", x(start) + 7, pad.t + 14);
  const labels = [0, Math.floor(history.load_mw.length / 2), start, count - 1]; ctx.fillStyle = "#7d91aa";
  labels.forEach((i) => { const timestamp = i < start ? history.timestamps[i] : forecast.timestamps[i - start]; const text = new Intl.DateTimeFormat("vi-VN", { day: "2-digit", month: "2-digit" }).format(new Date(timestamp)); ctx.fillText(text, x(i) - 12, height - 19); });
}

$("#origin-slider").addEventListener("input", (event) => loadForecast(Number(event.target.value)));
$("#previous").addEventListener("click", () => loadForecast(Math.max(0, state.data.origin - 24)));
$("#next").addEventListener("click", () => loadForecast(Math.min(state.data.origin_count - 1, state.data.origin + 24)));
$("#random").addEventListener("click", () => loadForecast(Math.floor(Math.random() * state.data.origin_count)));
$("#original-toggle").addEventListener("change", (event) => { state.showOriginal = event.target.checked; drawChart(); });
window.addEventListener("resize", () => state.data && drawChart());
waitForModel();
