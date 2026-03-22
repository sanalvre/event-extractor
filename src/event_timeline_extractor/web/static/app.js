const $ = (id) => document.getElementById(id);

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
      return { error: "Custom seconds must be a number ≥ 1." };
    }
    return { max_seconds: n };
  }
  const n = Number(mode);
  if (!Number.isFinite(n) || n < 1) {
    return { error: "Invalid duration preset." };
  }
  return { max_seconds: n };
}

async function run() {
  const url = $("url").value.trim();
  const dry = $("dry").checked;
  const btn = $("go");
  const out = $("out");

  out.hidden = true;
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
  showStatus("info", dry ? "Running dry pipeline (no API)…" : "Running — this can take a while…");

  const endpoint = dry ? "/api/timeline/dry-run" : "/api/timeline";

  const payload = { url };
  if (dur && dur.max_seconds != null) {
    payload.max_seconds = dur.max_seconds;
  }

  try {
    const r = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      showStatus("err", formatHttpError(data, r.status, r.statusText));
      return;
    }
    out.textContent = JSON.stringify(data, null, 2);
    out.hidden = false;
    hideStatus();
  } catch (e) {
    showStatus("err", String(e));
  } finally {
    btn.disabled = false;
  }
}

$("durationMode").addEventListener("change", syncDurationUi);
syncDurationUi();

$("go").addEventListener("click", run);
$("url").addEventListener("keydown", (e) => {
  if (e.key === "Enter") run();
});
