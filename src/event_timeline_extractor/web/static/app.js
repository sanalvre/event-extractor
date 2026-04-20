const $ = (id) => document.getElementById(id);

let lastResult = null;

function showStatus(kind, text) {
  const el = $("status");
  el.hidden = false;
  el.className = "status " + (kind === "err" ? "err" : "info");
  el.textContent = text;
}

function hideStatus() {
  const el = $("status");
  el.hidden = true;
  el.textContent = "";
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function titleCase(value) {
  return String(value || "")
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function confidenceLabel(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "Unscored";
  }
  return `${Math.round(value * 100)}% confidence`;
}

function setRawButtonLabel(showingRaw) {
  $("toggleRaw").textContent = showingRaw ? "Hide raw JSON" : "Show raw JSON";
}

/** FastAPI can return ``detail`` as a string, a list of validation errors, or an object. */
function formatHttpError(data, status, statusText) {
  if (!data || typeof data !== "object") {
    return status ? `[${status}] ${statusText}` : statusText;
  }
  const d = data.detail;
  if (typeof d === "string") {
    return `[${status}] ${d}`;
  }
  if (Array.isArray(d)) {
    const parts = d.map((item) => {
      if (typeof item === "string") return item;
      if (item && typeof item === "object") {
        const loc = Array.isArray(item.loc) ? item.loc.filter(Boolean).join(".") : "";
        const msg = item.msg || JSON.stringify(item);
        return loc ? `${loc}: ${msg}` : msg;
      }
      return String(item);
    });
    return `[${status}] ${parts.join("; ")}`;
  }
  if (d != null && typeof d === "object") {
    return `[${status}] ${JSON.stringify(d)}`;
  }
  return `[${status}] ${JSON.stringify(data)}`;
}

function syncDurationUi() {
  const mode = $("durationMode").value;
  $("customDurationWrap").hidden = mode !== "custom";
}

function resolveMaxSeconds() {
  const mode = $("durationMode").value;
  if (mode === "full") {
    return null;
  }
  if (mode === "custom") {
    const raw = $("maxsec").value.trim();
    if (!raw) {
      return { error: "Enter a number of seconds for custom clip length." };
    }
    const n = Number(raw);
    if (!Number.isFinite(n) || n < 1) {
      return { error: "Custom seconds must be a number >= 1." };
    }
    return { max_seconds: n };
  }
  const n = Number(mode);
  if (!Number.isFinite(n) || n < 1) {
    return { error: "Invalid duration preset." };
  }
  return { max_seconds: n };
}

async function submitJob(endpoint, payload) {
  const r = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    throw new Error(formatHttpError(data, r.status, r.statusText));
  }
  return data;
}

async function pollJob(jobId) {
  for (;;) {
    const r = await fetch(`/api/jobs/${jobId}`);
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      throw new Error(formatHttpError(data, r.status, r.statusText));
    }
    const status = data.status || "unknown";
    if (status === "completed") {
      return data;
    }
    if (status === "failed") {
      throw new Error(data.error || "Job failed.");
    }
    const percent = Number.isFinite(data.progress_percent) ? ` (${data.progress_percent}%)` : "";
    showStatus("info", `${data.message || `Job ${status}...`}${percent}`);
    await sleep(700);
  }
}

async function fetchJobResult(jobId) {
  const r = await fetch(`/api/jobs/${jobId}/result`);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    throw new Error(formatHttpError(data, r.status, r.statusText));
  }
  return data;
}

function renderSummary(data) {
  const meta = data.meta || {};
  const events = Array.isArray(data.events) ? data.events : [];
  const typedEvents = events.filter((event) => event.event_type && event.event_type !== "other").length;
  const averageConfidence = events
    .map((event) => (typeof event.confidence === "number" ? event.confidence : null))
    .filter((value) => value != null);
  const meanConfidence =
    averageConfidence.length > 0
      ? `${Math.round((averageConfidence.reduce((sum, value) => sum + value, 0) / averageConfidence.length) * 100)}%`
      : "N/A";
  const cards = [
    { label: "Events", value: String(events.length) },
    { label: "Typed Events", value: String(typedEvents) },
    { label: "Avg Confidence", value: meanConfidence },
    {
      label: "Batches",
      value: meta.batch_plan && Number.isFinite(meta.batch_plan.total_batches)
        ? String(meta.batch_plan.total_batches)
        : String(meta.extraction_batches || 0),
    },
  ];

  $("summary").innerHTML = cards
    .map(
      (card) => `
        <article class="summary-card">
          <p class="summary-label">${escapeHtml(card.label)}</p>
          <p class="summary-value">${escapeHtml(card.value)}</p>
        </article>
      `,
    )
    .join("");
}

function renderTimeline(data) {
  const events = Array.isArray(data.events) ? data.events : [];
  const timeline = $("timeline");
  if (events.length === 0) {
    timeline.innerHTML = `<p class="timeline-empty">No events were returned for this run.</p>`;
    return;
  }

  timeline.innerHTML = events
    .map((event) => {
      const sourceIds = Array.isArray(event.source_segment_ids) ? event.source_segment_ids.join(", ") : "";
      const sourceRange =
        event.source_start || event.source_end
          ? `${event.source_start || "?"} to ${event.source_end || "?"}`
          : null;
      const pills = [
        `<span class="pill pill-strong">${escapeHtml(titleCase(event.event_type || "other"))}</span>`,
      ];
      if (event.speaker) {
        pills.push(`<span class="pill">${escapeHtml(event.speaker)}</span>`);
      }
      if (typeof event.confidence === "number") {
        pills.push(`<span class="pill">${escapeHtml(confidenceLabel(event.confidence))}</span>`);
      }

      return `
        <article class="event-card">
          <div class="event-time">${escapeHtml(event.time || "--:--")}</div>
          <div class="event-main">
            <div class="event-topline">
              <h3 class="event-title">${escapeHtml(event.event || "(untitled event)")}</h3>
            </div>
            <div class="event-meta">${pills.join("")}</div>
            ${event.evidence ? `<p class="event-evidence">${escapeHtml(event.evidence)}</p>` : ""}
            ${
              sourceRange || sourceIds
                ? `<p class="event-source">Source: ${escapeHtml(sourceRange || "Unknown range")}${
                    sourceIds ? ` · ${escapeHtml(sourceIds)}` : ""
                  }</p>`
                : ""
            }
          </div>
        </article>
      `;
    })
    .join("");
}

function renderResult(data) {
  lastResult = data;
  $("results").hidden = false;
  $("out").textContent = JSON.stringify(data, null, 2);
  $("out").hidden = true;
  setRawButtonLabel(false);
  renderSummary(data);
  renderTimeline(data);
}

function clearResult() {
  lastResult = null;
  $("results").hidden = true;
  $("summary").innerHTML = "";
  $("timeline").innerHTML = "";
  $("out").textContent = "";
  $("out").hidden = true;
  setRawButtonLabel(false);
}

async function run() {
  const url = $("url").value.trim();
  const dry = $("dry").checked;
  const btn = $("go");

  clearResult();
  hideStatus();

  if (!url) {
    showStatus("err", "Paste a URL first.");
    return;
  }

  const dur = resolveMaxSeconds();
  if (dur && dur.error) {
    showStatus("err", dur.error);
    return;
  }

  btn.disabled = true;
  showStatus("info", dry ? "Submitting dry-run job..." : "Submitting extraction job...");

  const endpoint = dry ? "/api/jobs/dry-run" : "/api/jobs";
  const payload = { url };
  if (dur && dur.max_seconds != null) {
    payload.max_seconds = dur.max_seconds;
  }

  try {
    const job = await submitJob(endpoint, payload);
    showStatus("info", `Job queued: ${job.job_id}`);
    await pollJob(job.job_id);
    const data = await fetchJobResult(job.job_id);
    renderResult(data);
    hideStatus();
  } catch (e) {
    showStatus("err", String(e));
  } finally {
    btn.disabled = false;
  }
}

$("durationMode").addEventListener("change", syncDurationUi);
$("toggleRaw").addEventListener("click", () => {
  const out = $("out");
  const nextHidden = !out.hidden;
  out.hidden = nextHidden;
  setRawButtonLabel(!nextHidden);
  if (!nextHidden && lastResult) {
    out.textContent = JSON.stringify(lastResult, null, 2);
  }
});
syncDurationUi();

$("go").addEventListener("click", run);
$("url").addEventListener("keydown", (e) => {
  if (e.key === "Enter") run();
});
