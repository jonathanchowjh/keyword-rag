// Shared bits of the console: data access, number and time formatting, the sidebar.
// An ES module with no dependencies; every page imports what it needs.

const MINUS = "−";
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const POLL_MS = 15000;

// -- data ---------------------------------------------------------------------

export async function api(path, { method = "GET", body } = {}) {
  const init = { method };
  if (body !== undefined) {
    init.headers = { "content-type": "application/json" };
    init.body = JSON.stringify(body);
  }
  const response = await fetch(path, init);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail;
    throw new Error(typeof detail === "string" ? detail : response.statusText || "request failed");
  }
  return data;
}

let catalogPromise = null;
let known = null; // the catalog once fetched, for the label helpers
export function catalog() {
  if (!catalogPromise) {
    catalogPromise = api("/catalog")
      .then((c) => (known = c))
      .catch((e) => {
        catalogPromise = null; // a failed fetch is retried next time
        throw e;
      });
  }
  return catalogPromise;
}

export function esc(s) {
  const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  return String(s ?? "").replace(/[&<>"']/g, (c) => map[c]);
}

// -- formatting -----------------------------------------------------------------

const NONE = "–"; // shown for a missing or invalid value

export function pct(x) {
  return Number.isFinite(x) ? `${Math.round(x * 100)}%` : NONE;
}

// Whole number, rounded away from zero on halves so -2.5 and 2.5 mirror each other.
function whole(n) {
  const r = Math.round(Math.abs(n));
  return n < 0 && r > 0 ? -r : r;
}

function signed(n) {
  return n < 0 ? `${MINUS}${Math.abs(n)}` : String(n);
}

// "+6 pts (2–10)", "−7 pts (−12 to −2)": whole points, true minus, "pt" for exactly one.
// The range is left out when its bounds are missing.
export function pts(diff, lo, hi) {
  if (!Number.isFinite(diff)) return NONE;
  const d = whole(diff);
  const unit = Math.abs(d) === 1 ? "pt" : "pts";
  const text = `${d < 0 ? MINUS : "+"}${Math.abs(d)} ${unit}`;
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return text;
  const l = whole(lo);
  const h = whole(hi);
  const range = l >= 0 && h >= 0 ? `${l}–${h}` : `${signed(l)} to ${signed(h)}`;
  return `${text} (${range})`;
}

const two = (n) => String(n).padStart(2, "0");

// Clock time in the reader's zone; anything not from today carries its date.
export function clock(iso, now = new Date()) {
  const t = iso == null ? new Date(NaN) : new Date(iso);
  if (Number.isNaN(t.getTime())) return NONE;
  const time = `${two(t.getHours())}:${two(t.getMinutes())}`;
  const today =
    t.getFullYear() === now.getFullYear() &&
    t.getMonth() === now.getMonth() &&
    t.getDate() === now.getDate();
  return today ? time : `${t.getDate()} ${MONTHS[t.getMonth()]} ${time}`;
}

// Label helpers read the catalog fetched by catalog(); until then they show the id.
export function methodLabel(id, cat = known) {
  return cat?.methods?.[id]?.label ?? id;
}

// The muted "uses credit" note beside a paid method (spec §7.1); "" for a free one. A keyword
// method spends only on a collection with summaries, and says so.
const CREDIT_NOTE = { always: "uses credit", with_summaries: "uses credit with summaries" };
export function creditMark(id, cat = known) {
  const note = CREDIT_NOTE[cat?.methods?.[id]?.paid];
  return note ? ` <span class="credit muted">${esc(note)}</span>` : "";
}

export function metricLabel(id, qaType, cat = known) {
  return cat?.metric_labels_by_type?.[qaType]?.[id] ?? cat?.metrics?.[id]?.label ?? id;
}

// Counts with thousands separators: 11,656.
export function count(n) {
  return Number(n).toLocaleString("en-US");
}

// Money as the spend line shows it: two decimals, "< $0.01" below a cent.
export function usd(x) {
  const v = Number(x ?? 0);
  return v > 0 && v < 0.01 ? "< $0.01" : `$${v.toFixed(2)}`;
}

export const FREE_SCORE_NOTE = "free score, not yet checked against answers";

// True when the headline is the free reasoning match and nothing has vouched for it (spec §7.3).
function freeScore(answer) {
  return answer.metric === "solution_sim@1" && answer.proxy_trusted !== true;
}

// "Yes: Keywords per list +6 pts (2–10) answered correctly vs Hybrid"
export function verdictText(answer, qaType, labels = known) {
  if (!answer) return "No answer yet";
  const head = { yes: "Yes", no: "No", cant_tell: "Can't tell" }[answer.verdict] ?? "Can't tell";
  const cand = methodLabel(answer.candidate, labels);
  const ref = methodLabel(answer.reference, labels);
  const metric = metricLabel(answer.metric, qaType, labels).toLowerCase();
  const text = `${head}: ${cand} ${pts(answer.diff, answer.lo, answer.hi)} ${metric} vs ${ref}`;
  return freeScore(answer) ? `${text} · ${FREE_SCORE_NOTE}` : text;
}

// An inline SVG gap bar: a centre line at 0, a bar from lo to hi, a dot at diff. All values in
// points; `span` is the axis half-width in the same units, and anything beyond it is clamped.
export function gapBar(lo, diff, hi, span) {
  const W = 220;
  const H = 18;
  const half = Number.isFinite(span) && span > 0 ? span : 10;
  const x = (v) => W / 2 + (Math.max(-half, Math.min(half, v)) / half) * (W / 2 - 4);
  const label = esc(`Gap ${pts(diff, lo, hi)}`);
  const bar =
    Number.isFinite(lo) && Number.isFinite(hi)
      ? `<rect x="${x(Math.min(lo, hi)).toFixed(1)}" y="5" width="${Math.max(
          x(Math.max(lo, hi)) - x(Math.min(lo, hi)),
          1
        ).toFixed(1)}" height="8" rx="2" fill="currentColor" opacity=".3"/>`
      : "";
  const dot = Number.isFinite(diff)
    ? `<circle cx="${x(diff).toFixed(1)}" cy="9" r="4" fill="currentColor"/>`
    : "";
  return `<svg class="gap" role="img" aria-label="${label}" viewBox="0 0 ${W} ${H}" width="${W}"
    height="${H}"><line x1="${W / 2}" x2="${W / 2}" y1="0" y2="${H}" stroke="currentColor"
    stroke-width="1" opacity=".6"/>${bar}${dot}</svg>`;
}

// -- layout ---------------------------------------------------------------------

export function pill(state) {
  const kind =
    { ready: "ok", done: "ok", loading: "run", rekeying: "run", evaluating: "run", running: "run",
      busy: "run", queued: "idle", failed: "bad", interrupted: "bad", stopped_budget: "bad",
      stopped_no_key: "bad", cancelled: "idle" }[state] ?? "idle";
  const text = String(state ?? "").replace(/_/g, " ").replace("rekeying", "re-keying");
  return `<span class="pill ${kind}">${esc(text.charAt(0).toUpperCase() + text.slice(1))}</span>`;
}

// crumb: a list of "Label" or {label, href}.
export function head({ title, crumb = [], sub = "", aside = "" }) {
  const trail = crumb
    .map((c) => (typeof c === "string" ? esc(c) : `<a href="${esc(c.href)}">${esc(c.label)}</a>`))
    .join(" / ");
  return `<div class="head"><div>
    ${trail ? `<div class="crumb">${trail}</div>` : ""}
    <h1>${esc(title)}</h1>
    ${sub ? `<p class="sub">${esc(sub)}</p>` : ""}
  </div>${aside ? `<div class="aside">${aside}</div>` : ""}</div>`;
}

// Renders "≈ $0.41 · 400 Claude calls · [Start]" into el; onStart runs only on click.
export function confirmSpend(el, estimate, onStart) {
  const usd = Number(estimate.est_usd ?? 0);
  const calls = Number(estimate.calls ?? 0);
  const what = calls === 1 ? "1 Claude call" : `${calls} Claude calls`;
  const cost = usd > 0 && usd < 0.01 ? "< $0.01" : `≈ $${usd.toFixed(2)}`;
  el.innerHTML = `<span class="confirm">${cost} · ${esc(what)} ·
    <button type="button">Start</button></span>`;
  const button = el.querySelector("button");
  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      await onStart();
    } finally {
      button.disabled = false;
    }
  });
}

function money(x) {
  return Number.isInteger(x) ? String(x) : x.toFixed(2);
}

function link(href, label, cls, active, key) {
  return `<a href="${href}" class="${cls}${active === key ? " active" : ""}">${label}</a>`;
}

function sideHtml(active, names) {
  const collections = names
    .map((n) => link(`/c/${encodeURIComponent(n)}`, esc(n), "sub", active, `c/${n}`))
    .join("");
  return `<div class="brand">rag-qa</div><div class="tag">keyword RAG test bench</div>
    ${link("/", "Collections", "", active, "collections")}${collections}
    ${link("/lab", "Lab", "", active, "lab")}
    <div class="group">BEHIND THE NUMBERS</div>
    ${link("/worker", 'Worker <span id="worker-dot" class="dot"></span>', "", active, "worker")}
    <a href="/docs">API docs</a>
    <div class="spend" id="spend"></div>`;
}

// The sidebar's spend line; pages call it after anything that may have spent.
export function refreshSpend() {
  return api("/budget")
    .then((b) => {
      const el = document.getElementById("spend");
      if (el) el.textContent = `$${b.spent_usd.toFixed(2)} of $${money(b.cap_usd)} this month`;
    })
    .catch(() => {});
}

function refreshDot() {
  return api("/status")
    .then((s) => {
      const kind = s.problems.length ? "bad" : s.worker.running ? "run" : "ok";
      const dot = document.getElementById("worker-dot");
      if (dot) dot.className = `dot ${kind}`;
    })
    .catch(() => {});
}

// Renders the sidebar into <aside id="side">. `active` is "collections", "lab", "worker",
// or "c/<name>" for a collection's page.
export function shell(active) {
  const side = document.getElementById("side");
  let names = [];
  const draw = () => {
    side.innerHTML = sideHtml(active, names);
    refreshSpend();
    refreshDot();
  };
  draw();
  api("/collections")
    .then((list) => {
      names = list.map((c) => c.name);
      draw();
    })
    .catch(() => {});
  setInterval(() => {
    refreshSpend();
    refreshDot();
  }, POLL_MS);
}
