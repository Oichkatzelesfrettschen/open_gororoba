const pipelineGrid = document.getElementById("pipeline-grid");
const timelineBody = document.getElementById("timeline-body");
const profileEl = document.getElementById("profile");
const runSuiteBtn = document.getElementById("run-suite");
const healthStatusEl = document.getElementById("health-status");
const healthClockEl = document.getElementById("health-clock");
const healthSummaryEl = document.getElementById("health-summary");
const notificationsEl = document.getElementById("notifications");
const template = document.getElementById("pipeline-template");

const benchmarkPipelineEl = document.getElementById("benchmark-pipeline");
const benchmarkIterationsEl = document.getElementById("benchmark-iterations");
const runBenchmarkBtn = document.getElementById("run-benchmark");
const benchmarkOutputEl = document.getElementById("benchmark-output");

const reproPipelineEl = document.getElementById("repro-pipeline");
const reproIterationsEl = document.getElementById("repro-iterations");
const reproToleranceEl = document.getElementById("repro-tolerance");
const runReproBtn = document.getElementById("run-repro");
const reproOutputEl = document.getElementById("repro-output");

const state = {
  pipelines: [],
  history: [],
};

function currentProfile() {
  return profileEl.value === "full" ? "full" : "quick";
}

function clampInteger(value, fallback, min, max) {
  const numeric = Number.parseInt(value, 10);
  if (!Number.isFinite(numeric)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, numeric));
}

function formatUtc(unixSeconds) {
  if (!unixSeconds) {
    return "-";
  }
  return new Date(unixSeconds * 1000).toISOString().replace("T", " ").replace(".000Z", "Z");
}

function describeProfile(profile) {
  return profile === "full" ? "full" : "quick";
}

function noticeKind(kind) {
  if (kind === "error") {
    return "notice--error";
  }
  if (kind === "warn") {
    return "notice--warn";
  }
  return "notice--ok";
}

function pushNotice(message, kind = "ok") {
  const node = document.createElement("article");
  node.className = `notice ${noticeKind(kind)}`;
  node.textContent = message;
  notificationsEl.prepend(node);
  const maxNotices = 5;
  while (notificationsEl.children.length > maxNotices) {
    notificationsEl.removeChild(notificationsEl.lastElementChild);
  }
  window.setTimeout(() => {
    if (node.parentElement) {
      node.parentElement.removeChild(node);
    }
  }, 7000);
}

async function getJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let payload = {};
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch (_) {
      payload = { raw: text };
    }
  }
  if (!response.ok) {
    const reason = payload.error || payload.raw || response.statusText;
    throw new Error(`${response.status} ${response.statusText}: ${reason}`);
  }
  return payload;
}

function cardForPipelineId(id) {
  return pipelineGrid.querySelector(`[data-pipeline-id="${id}"]`);
}

function setPipelineStatus(id, text, isError = false) {
  const card = cardForPipelineId(id);
  if (!card) {
    return;
  }
  const statusNode = card.querySelector(".pipeline-status");
  statusNode.textContent = text;
  statusNode.classList.toggle("gate-fail", isError);
  statusNode.classList.toggle("gate-pass", !isError);
}

function setButtonBusy(button, busy, busyLabel, idleLabel) {
  button.disabled = busy;
  button.textContent = busy ? busyLabel : idleLabel;
}

function addCell(row, content, className = "") {
  const cell = document.createElement("td");
  cell.textContent = content;
  if (className) {
    cell.className = className;
  }
  row.appendChild(cell);
}

function renderHistory() {
  timelineBody.innerHTML = "";
  state.history.forEach((entry) => {
    const row = document.createElement("tr");
    const gateClass = entry.passes_gate ? "gate-pass" : "gate-fail";
    addCell(row, String(entry.run_id ?? "-"), "mono");
    addCell(row, formatUtc(entry.unix_seconds), "mono");
    addCell(row, entry.experiment_id ?? "-");
    addCell(row, describeProfile(entry.profile ?? "quick"));
    addCell(row, Number(entry.metric_value ?? 0).toFixed(6), "mono");
    addCell(row, Number(entry.threshold ?? 0).toFixed(6), "mono");
    addCell(row, entry.passes_gate ? "PASS" : "FAIL", gateClass);
    addCell(row, String(entry.duration_ms ?? "-"), "mono");
    timelineBody.appendChild(row);
  });
}

function pushHistory(items) {
  const entries = Array.isArray(items) ? items : [items];
  state.history = [...entries, ...state.history];
  state.history = state.history.slice(0, 120);
  renderHistory();
  if (state.history.length > 0) {
    const newest = state.history[0];
    healthSummaryEl.textContent = `Last run ${newest.experiment_id} (${describeProfile(newest.profile)}) :: ${newest.passes_gate ? "PASS" : "FAIL"} :: ${newest.duration_ms} ms`;
  }
}

function addPipelineCard(pipeline) {
  const node = template.content.firstElementChild.cloneNode(true);
  node.dataset.pipelineId = pipeline.id;
  node.querySelector("h3").textContent = pipeline.title;
  node.querySelector(".badge").textContent = pipeline.id;
  node.querySelector(".pipeline-hypothesis").textContent = pipeline.hypothesis;
  node.querySelector(".metric").textContent = `Metric: ${pipeline.primary_metric}`;
  node.querySelector(".quick").textContent = `Quick: ${pipeline.quick_profile}`;
  node.querySelector(".full").textContent = `Full: ${pipeline.full_profile}`;
  const button = node.querySelector(".run-pipeline");
  const idleLabel = "Run";
  button.addEventListener("click", async () => {
    setButtonBusy(button, true, "Running...", idleLabel);
    setPipelineStatus(pipeline.id, "Running...");
    try {
      const result = await getJson(`/api/run/${pipeline.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile: currentProfile() }),
      });
      pushHistory(result);
      const gateLabel = result.passes_gate ? "PASS" : "FAIL";
      setPipelineStatus(pipeline.id, `${gateLabel} in ${result.duration_ms} ms`, !result.passes_gate);
      pushNotice(`Pipeline ${pipeline.id} finished: ${gateLabel}.`, result.passes_gate ? "ok" : "warn");
    } catch (error) {
      setPipelineStatus(pipeline.id, "Run failed", true);
      pushNotice(`Pipeline ${pipeline.id} failed: ${error.message}`, "error");
    } finally {
      setButtonBusy(button, false, "Running...", idleLabel);
    }
  });
  pipelineGrid.appendChild(node);
}

function fillPipelineSelect(selectEl) {
  selectEl.innerHTML = "";
  state.pipelines.forEach((pipeline) => {
    const option = document.createElement("option");
    option.value = pipeline.id;
    option.textContent = `${pipeline.id} :: ${pipeline.title}`;
    selectEl.appendChild(option);
  });
}

async function loadPipelines() {
  const pipelines = await getJson("/api/pipelines");
  state.pipelines = pipelines;
  pipelineGrid.innerHTML = "";
  pipelines.forEach(addPipelineCard);
  fillPipelineSelect(benchmarkPipelineEl);
  fillPipelineSelect(reproPipelineEl);
}

async function loadHistory() {
  const history = await getJson("/api/history");
  state.history = Array.isArray(history) ? history : [];
  renderHistory();
}

async function loadHealth() {
  try {
    const health = await getJson("/api/health");
    healthStatusEl.textContent = `${health.service} :: ${health.status.toUpperCase()}`;
    healthClockEl.textContent = `Last refresh: ${formatUtc(health.unix_seconds)}`;
  } catch (error) {
    healthStatusEl.textContent = "gororoba-studio :: DEGRADED";
    healthClockEl.textContent = `Health error: ${error.message}`;
  }
}

runSuiteBtn.addEventListener("click", async () => {
  setButtonBusy(runSuiteBtn, true, "Running Suite...", "Run Full Suite");
  try {
    const data = await getJson("/api/run-suite", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile: currentProfile() }),
    });
    pushHistory(data.results ?? []);
    (data.failures ?? []).forEach((failure) => {
      setPipelineStatus(failure.experiment_id, "Suite failure", true);
    });
    pushNotice(
      `Suite complete: ${data.pass_count} pass / ${data.fail_count} fail in ${data.total_duration_ms} ms.`,
      data.failures && data.failures.length > 0 ? "warn" : "ok",
    );
  } catch (error) {
    pushNotice(`Suite failed: ${error.message}`, "error");
  } finally {
    setButtonBusy(runSuiteBtn, false, "Running Suite...", "Run Full Suite");
  }
});

runBenchmarkBtn.addEventListener("click", async () => {
  const pipelineId = benchmarkPipelineEl.value;
  const iterations = clampInteger(benchmarkIterationsEl.value, 5, 1, 25);
  benchmarkIterationsEl.value = String(iterations);
  setButtonBusy(runBenchmarkBtn, true, "Benchmarking...", "Run Benchmark");
  try {
    const data = await getJson(`/api/benchmark/${pipelineId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile: currentProfile(), iterations }),
    });
    pushHistory(data.runs ?? []);
    benchmarkOutputEl.textContent = [
      `Pipeline: ${data.experiment_id}`,
      `Profile: ${describeProfile(data.profile)}`,
      `Iterations: ${data.iterations_completed}/${data.iterations_requested}`,
      `Pass/Fail: ${data.pass_count}/${data.fail_count}`,
      `Duration mean/median ms: ${data.mean_duration_ms.toFixed(2)} / ${data.median_duration_ms.toFixed(2)}`,
      `Duration min/max ms: ${data.min_duration_ms} / ${data.max_duration_ms}`,
      `Metric mean/stdev: ${data.mean_metric_value.toFixed(6)} / ${data.metric_stddev.toFixed(6)}`,
    ].join("\n");
    pushNotice(`Benchmark completed for ${pipelineId}.`, "ok");
  } catch (error) {
    benchmarkOutputEl.textContent = `Benchmark failed: ${error.message}`;
    pushNotice(`Benchmark failed for ${pipelineId}: ${error.message}`, "error");
  } finally {
    setButtonBusy(runBenchmarkBtn, false, "Benchmarking...", "Run Benchmark");
  }
});

runReproBtn.addEventListener("click", async () => {
  const pipelineId = reproPipelineEl.value;
  const iterations = clampInteger(reproIterationsEl.value, 3, 2, 20);
  const tolerance = Number.parseFloat(reproToleranceEl.value);
  const normalizedTolerance = Number.isFinite(tolerance) && tolerance >= 0 ? tolerance : 1e-9;
  reproIterationsEl.value = String(iterations);
  reproToleranceEl.value = String(normalizedTolerance);
  setButtonBusy(runReproBtn, true, "Checking...", "Run Repro Check");
  try {
    const data = await getJson(`/api/reproducibility/${pipelineId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        profile: currentProfile(),
        iterations,
        tolerance: normalizedTolerance,
      }),
    });
    pushHistory(data.runs ?? []);
    reproOutputEl.textContent = [
      `Pipeline: ${data.experiment_id}`,
      `Profile: ${describeProfile(data.profile)}`,
      `Stable: ${data.stable}`,
      `Gate consistent: ${data.gate_consistent}`,
      `Tolerance: ${data.tolerance}`,
      `Baseline metric: ${Number(data.baseline_metric_value).toFixed(6)}`,
      `Max delta: ${Number(data.max_metric_delta).toExponential(3)}`,
      `Completed: ${data.iterations_completed}/${data.iterations_requested}`,
    ].join("\n");
    pushNotice(
      `Reproducibility ${data.stable ? "passed" : "requires review"} for ${pipelineId}.`,
      data.stable ? "ok" : "warn",
    );
  } catch (error) {
    reproOutputEl.textContent = `Reproducibility check failed: ${error.message}`;
    pushNotice(`Reproducibility check failed for ${pipelineId}: ${error.message}`, "error");
  } finally {
    setButtonBusy(runReproBtn, false, "Checking...", "Run Repro Check");
  }
});

async function bootstrap() {
  await Promise.all([loadPipelines(), loadHistory(), loadHealth()]);
  window.setInterval(() => {
    loadHealth().catch((error) => {
      healthStatusEl.textContent = "gororoba-studio :: DEGRADED";
      healthClockEl.textContent = error.message;
    });
  }, 15000);
  pushNotice("Studio controls loaded. Ready for benchmark and reproducibility runs.", "ok");
}

bootstrap().catch((error) => {
  healthStatusEl.textContent = "service unavailable";
  healthClockEl.textContent = error.message;
  pushNotice(`Bootstrap failed: ${error.message}`, "error");
});
