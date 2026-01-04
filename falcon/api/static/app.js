const eventList = document.getElementById("eventList");
const successRateEl = document.getElementById("successRate");
const triggerRateEl = document.getElementById("triggerRate");
const falsePosEl = document.getElementById("falsePos");
const falseNegEl = document.getElementById("falseNeg");
const decisionActionEl = document.getElementById("decisionAction");
const decisionConfidenceEl = document.getElementById("decisionConfidence");
const decisionReasoningEl = document.getElementById("decisionReasoning");
const energyUsedEl = document.getElementById("energyUsed");
const energyRemainingEl = document.getElementById("energyRemaining");
const memorySizeEl = document.getElementById("memorySize");
const memoryTypeEl = document.getElementById("memoryType");
const consensusMethodEl = document.getElementById("consensusMethod");
const sharedPoolEl = document.getElementById("sharedPool");
const swarmDeltaEl = document.getElementById("swarmDelta");
const scenarioNameEl = document.getElementById("scenarioName");
const modeNameEl = document.getElementById("modeName");
const tickRateEl = document.getElementById("tickRate");
const lastEventEl = document.getElementById("lastEvent");
const lastValueEl = document.getElementById("lastValue");
const pulseDot = document.getElementById("pulseDot");
const decisionCard = document.getElementById("decisionCard");

const scenarioSelect = document.getElementById("scenarioSelect");
const modeSelect = document.getElementById("modeSelect");
const intervalRange = document.getElementById("intervalRange");
const intervalValue = document.getElementById("intervalValue");
const toggleRun = document.getElementById("toggleRun");
const toggleShowcase = document.getElementById("toggleShowcase");
const togglePresentation = document.getElementById("togglePresentation");
const exportSnapshot = document.getElementById("exportSnapshot");

const confidenceSpark = document.getElementById("confidenceSpark");
const trendConfidence = document.getElementById("trendConfidence");
const trendTrigger = document.getElementById("trendTrigger");
const trendEnergy = document.getElementById("trendEnergy");
const trendMemory = document.getElementById("trendMemory");
const benchmarkSummary = document.getElementById("benchmarkSummary");

const consensusSegments = {
  escalate: document.getElementById("consensusEscalate"),
  intervene: document.getElementById("consensusIntervene"),
  alert: document.getElementById("consensusAlert"),
  observe: document.getElementById("consensusObserve"),
};

const state = {
  confidences: [],
  triggerRates: [],
  energyRemaining: [],
  memorySizes: [],
  successRates: [],
  running: true,
  lastPayload: null,
  report: null,
  presentation: false,
  showcase: {
    enabled: false,
    timer: null,
    index: 0,
  },
};

const scenarioCycle = ["noise", "spike", "attack", "pulse", "drift"];
const historyLimit = 80;

intervalValue.textContent = `${intervalRange.value} ms`;

intervalRange.addEventListener("input", () => {
  intervalValue.textContent = `${intervalRange.value} ms`;
});

intervalRange.addEventListener("change", () => {
  sendControl({ interval_ms: Number(intervalRange.value) });
});

scenarioSelect.addEventListener("change", () => {
  sendControl({ scenario: scenarioSelect.value });
});

modeSelect.addEventListener("change", () => {
  sendControl({ mode: modeSelect.value });
});

toggleRun.addEventListener("click", () => {
  state.running = !state.running;
  toggleRun.textContent = state.running ? "Pause" : "Resume";
  sendControl({ running: state.running });
});

toggleShowcase.addEventListener("click", () => {
  setShowcase(!state.showcase.enabled);
});

if (togglePresentation) {
  togglePresentation.addEventListener("click", () => {
    setPresentation(!state.presentation);
  });
}

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && state.presentation) {
    setPresentation(false);
  }
  if (event.key.toLowerCase() === "p") {
    setPresentation(!state.presentation);
  }
});

exportSnapshot.addEventListener("click", () => {
  if (!state.lastPayload) return;
  const payload = {
    timestamp: new Date().toISOString(),
    latest: state.lastPayload,
    series: {
      confidence: state.confidences,
      trigger_rate: state.triggerRates,
      energy_remaining: state.energyRemaining,
      memory_size: state.memorySizes,
    },
  };
  downloadJSON(payload, `falcon_snapshot_${Date.now()}.json`);
});

const eventSource = new EventSource("/api/stream");

eventSource.onmessage = (event) => {
  if (!event.data) return;
  const payload = JSON.parse(event.data);
  updateUI(payload);
};

fetch("/api/report")
  .then((res) => res.json())
  .then((report) => {
    state.report = report;
    renderBenchmark(report, scenarioSelect.value);
  })
  .catch(() => null);

fetch("/api/status")
  .then((res) => res.json())
  .then((status) => {
    scenarioNameEl.textContent = status.scenario || "-";
    modeNameEl.textContent = status.mode || "-";
    tickRateEl.textContent = `${status.interval_ms || 0} ms`;
    if (status.scenario) {
      scenarioSelect.value = status.scenario;
    }
    if (status.mode) {
      modeSelect.value = status.mode;
    }
    if (status.interval_ms) {
      intervalRange.value = status.interval_ms;
      intervalValue.textContent = `${status.interval_ms} ms`;
    }
  })
  .catch(() => null);

function updateUI(payload) {
  state.lastPayload = payload;
  scenarioNameEl.textContent = payload.scenario;
  modeNameEl.textContent = payload.mode;
  if (payload.interval_ms) {
    tickRateEl.textContent = `${payload.interval_ms} ms`;
  }

  const metrics = payload.metrics || {};
  successRateEl.textContent = formatPercent(metrics.success_rate);
  triggerRateEl.textContent = formatPercent(metrics.trigger_rate);
  falsePosEl.textContent = metrics.false_positives ?? 0;
  falseNegEl.textContent = metrics.false_negatives ?? 0;

  const label = payload.label ? "ACTION" : "IGNORE";
  lastEventEl.textContent = label;
  lastValueEl.textContent = formatValue(payload.data);
  flashPulse(payload.label || payload.decision);

  const decision = payload.decision;
  updateDecisionCard(decision);

  if (decision) {
    state.confidences.push(decision.confidence);
  } else {
    state.confidences.push(0);
  }

  state.triggerRates.push(metrics.trigger_rate || 0);
  state.successRates.push(metrics.success_rate || 0);
  state.energyRemaining.push(payload.energy ? payload.energy.remaining_fraction || 0 : 0);
  state.memorySizes.push(payload.memory ? payload.memory.size || 0 : 0);

  trimSeries();
  drawSparkline(confidenceSpark, state.confidences, "#1f9d8b");
  drawSparkline(trendConfidence, state.confidences, "#1f9d8b");
  drawSparkline(trendTrigger, state.triggerRates, "#e08a2e");
  drawSparkline(trendEnergy, state.energyRemaining, "#5a6572");
  drawSparkline(trendMemory, state.memorySizes, "#d36456");

  const energy = payload.energy || {};
  energyUsedEl.textContent = Math.round(energy.used_operations || 0);
  energyRemainingEl.textContent = formatPercent(energy.remaining_fraction || 0);

  const memory = payload.memory || {};
  memorySizeEl.textContent = memory.size ?? 0;
  memoryTypeEl.textContent = (memory.type || "none").toUpperCase();

  updateConsensus(payload.consensus, decision ? decision.action : null);
  updateSwarmDelta(payload);
  updateSharedPool(payload.shared_pool);

  addEventItem(payload);

  if (state.report) {
    renderBenchmark(state.report, payload.scenario);
  }
}

function updateDecisionCard(decision) {
  const actionClassList = [
    "action-escalate",
    "action-intervene",
    "action-alert",
    "action-observe",
  ];
  decisionCard.classList.remove(...actionClassList);

  if (decision) {
    decisionActionEl.textContent = decision.action.toUpperCase();
    decisionConfidenceEl.textContent = formatPercent(decision.confidence);
    decisionReasoningEl.textContent = decision.reasoning;
    decisionCard.classList.add(`action-${decision.action}`);
  } else {
    decisionActionEl.textContent = "-";
    decisionConfidenceEl.textContent = "-";
    decisionReasoningEl.textContent = "No action";
  }
}

function updateConsensus(consensus, activeAction) {
  Object.values(consensusSegments).forEach((segment) => {
    segment.classList.remove("active");
  });

  if (!consensus || !consensus.individual_votes) {
    Object.values(consensusSegments).forEach((segment) => {
      segment.style.opacity = 0.4;
      segment.style.flex = "1";
    });
    consensusMethodEl.textContent = "-";
    return;
  }

  const votes = consensus.individual_votes || [];
  const counts = { escalate: 0, intervene: 0, alert: 0, observe: 0 };

  votes.forEach((vote) => {
    const action = vote.action;
    if (counts[action] !== undefined) {
      counts[action] += vote.confidence || 1;
    }
  });

  const total = Object.values(counts).reduce((sum, value) => sum + value, 0) || 1;
  Object.entries(consensusSegments).forEach(([key, element]) => {
    const portion = counts[key] / total;
    element.style.opacity = 0.5 + portion * 0.5;
    element.style.flex = `${Math.max(0.2, portion)}`;
  });

  if (activeAction && consensusSegments[activeAction]) {
    consensusSegments[activeAction].classList.add("active");
  }

  consensusMethodEl.textContent = consensus.consensus_method || "weighted";
}

function updateSwarmDelta(payload) {
  if (!payload.solo) {
    swarmDeltaEl.textContent = "Delta: -";
    return;
  }
  const delta = (payload.metrics.success_rate || 0) - (payload.solo.success_rate || 0);
  const sign = delta >= 0 ? "+" : "";
  swarmDeltaEl.textContent = `Delta: ${sign}${formatPercent(delta)}`;
}

function updateSharedPool(sharedPool) {
  if (!sharedPool) {
    sharedPoolEl.textContent = "Pool: -";
    return;
  }
  sharedPoolEl.textContent = `Pool: ${sharedPool.total_experiences || 0}`;
}

function addEventItem(payload) {
  const li = document.createElement("li");
  const label = payload.label ? "ACTION" : "IGNORE";
  const action = payload.decision ? payload.decision.action : "none";
  const value = formatValue(payload.data);
  li.className = payload.label ? "event-action" : "event-ignore";
  li.innerHTML = `<strong>${label}</strong><span>${value}</span><span>${action}</span>`;
  eventList.prepend(li);

  while (eventList.children.length > 8) {
    eventList.removeChild(eventList.lastChild);
  }
}

function drawSparkline(canvas, values, color) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  ctx.beginPath();
  ctx.strokeStyle = color || "#1f9d8b";
  ctx.lineWidth = 2;

  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  const range = max - min || 1;

  values.forEach((value, index) => {
    const x = (index / Math.max(values.length - 1, 1)) * width;
    const y = height - ((value - min) / range) * height;
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
}

function sendControl(payload) {
  fetch("/api/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }).catch(() => null);
}

function formatPercent(value) {
  if (value === null || value === undefined) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function formatValue(value) {
  if (Array.isArray(value)) {
    return value.map((v) => Number(v).toFixed(2)).join(", ");
  }
  if (typeof value === "number") {
    return value.toFixed(3);
  }
  if (value === null || value === undefined) {
    return "-";
  }
  return String(value);
}

function trimSeries() {
  [
    state.confidences,
    state.triggerRates,
    state.energyRemaining,
    state.memorySizes,
    state.successRates,
  ].forEach((series) => {
    while (series.length > historyLimit) {
      series.shift();
    }
  });
}

function renderBenchmark(report, scenario) {
  if (!report || !report.results) return;
  const results = report.results || [];
  const filtered = scenario ? results.filter((r) => r.scenario === scenario) : results;
  const sorted = filtered
    .slice()
    .sort((a, b) => (b.metrics.success_rate || 0) - (a.metrics.success_rate || 0) || (b.metrics.avg_reward || 0) - (a.metrics.avg_reward || 0));

  const rows = sorted.slice(0, 5).map((r, index) => {
    const badge = index === 0 ? '<span class="benchmark-badge">Best</span>' : "";
    const rowClass = index === 0 ? "benchmark-row best" : "benchmark-row";
    return `<div class="${rowClass}">
      <strong>${r.scenario}</strong>
      <span>${r.decision} / ${r.energy} ${badge}</span>
      <span>${formatPercent(r.metrics.success_rate || 0)}</span>
    </div>`;
  });

  if (!rows.length) {
    benchmarkSummary.innerHTML = "<p>No benchmarks for this scenario yet.</p>";
    return;
  }

  benchmarkSummary.innerHTML = `
    <div class="benchmark-grid">
      ${rows.join("")}
    </div>
  `;
}

function flashPulse(active) {
  if (!pulseDot) return;
  if (!active) {
    pulseDot.classList.remove("active");
    return;
  }
  pulseDot.classList.add("active");
  setTimeout(() => pulseDot.classList.remove("active"), 800);
}

function setShowcase(enabled, options = {}) {
  state.showcase.enabled = enabled;
  toggleShowcase.textContent = enabled ? "On" : "Off";

  if (enabled) {
    startShowcase(options.intervalMs, options.cycleMs);
  } else {
    stopShowcase();
  }
}

function startShowcase(intervalMs = 300, cycleMs = 20000) {
  if (state.showcase.timer) return;
  const cycle = () => {
    state.showcase.index = (state.showcase.index + 1) % scenarioCycle.length;
    const scenario = scenarioCycle[state.showcase.index];
    scenarioSelect.value = scenario;
    sendControl({ scenario });
  };

  intervalRange.value = intervalMs;
  intervalValue.textContent = `${intervalMs} ms`;
  sendControl({ interval_ms: intervalMs });

  cycle();
  state.showcase.timer = setInterval(cycle, cycleMs);
}

function stopShowcase() {
  if (state.showcase.timer) {
    clearInterval(state.showcase.timer);
    state.showcase.timer = null;
  }
}

function setPresentation(enabled) {
  state.presentation = enabled;
  togglePresentation.textContent = enabled ? "On" : "Off";
  document.body.classList.toggle("presentation", enabled);
  setControlsDisabled(enabled);

  if (enabled) {
    if (!state.running) {
      state.running = true;
      toggleRun.textContent = "Pause";
      sendControl({ running: true });
    }
    setShowcase(true, { intervalMs: 250, cycleMs: 15000 });
  } else {
    setShowcase(false);
  }
}

function setControlsDisabled(disabled) {
  scenarioSelect.disabled = disabled;
  modeSelect.disabled = disabled;
  intervalRange.disabled = disabled;
  toggleRun.disabled = disabled;
  toggleShowcase.disabled = disabled;
}

function downloadJSON(payload, filename) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
