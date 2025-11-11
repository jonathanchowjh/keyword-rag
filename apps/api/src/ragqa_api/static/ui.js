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

export function pct(x) {
  return `${Math.round(x * 100)}%`;
}

function signed(n) {
  return n < 0 ? `${MINUS}${Math.abs(n)}` : String(n);
}

// "+6 pts (2–10)", "−7 pts (−12 to −2)": whole points, true minus, "pt" for exactly one.
export function pts(diff, lo, hi) {
  const d = Math.round(diff);
  const l = Math.round(lo);
  const h = Math.round(hi);
  const unit = Math.abs(d) === 1 ? "pt" : "pts";
  const range = l >= 0 && h >= 0 ? `${l}–${h}` : `${signed(l)} to ${signed(h)}`;
  return `${d < 0 ? MINUS : "+"}${Math.abs(d)} ${unit} (${range})`;
}

const two = (n) => String(n).padStart(2, "0");

// Clock time in the reader's zone; anything not from today carries its date.
export function clock(iso, now = new Date()) {
  const t = new Date(iso);
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

export function metricLabel(id, qaType, cat = known) {
  return cat?.metric_labels_by_type?.[qaType]?.[id] ?? cat?.metrics?.[id]?.label ?? id;
}

// "Yes: Keywords per list +6 pts (2–10) answered correctly vs Hybrid"
export function verdictText(answer, labels = known) {
  if (!answer) return "No answer yet";
  const head = { yes: "Yes", no: "No", cant_tell: "Can't tell" }[answer.verdict] ?? "Can't tell";
  const cand = methodLabel(answer.candidate, labels);
  const ref = methodLabel(answer.reference, labels);
  const metric = (labels?.metrics?.[answer.metric]?.label ?? answer.metric).toLowerCase();
  return `${head}: ${cand} ${pts(answer.diff, answer.lo, answer.hi)} ${metric} vs ${ref}`;
}

// -- layout ---------------------------------------------------------------------

export function pill(state) {
  const kind =
    { ready: "ok", done: "ok", loading: "run", rekeying: "run", evaluating: "run", running: "run",
      busy: "run", queued: "idle", failed: "bad", interrupted: "bad", stopped_budget: "bad",
      stopped_no_key: "bad", cancelled: "idle" }[state] ?? "idle";
  const text = String(state ?? "").replace(/_/g, " ");
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
  el.innerHTML = `<span class="confirm">≈ $${usd.toFixed(2)} · ${esc(what)} ·
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
  const refreshSpend = () =>
    api("/budget")
      .then((b) => {
        document.getElementById("spend").textContent =
          `$${b.spent_usd.toFixed(2)} of $${money(b.cap_usd)} this month`;
      })
      .catch(() => {});
  const refreshDot = () =>
    api("/status")
      .then((s) => {
        const kind = s.problems.length ? "bad" : s.worker.running ? "run" : "ok";
        const dot = document.getElementById("worker-dot");
        if (dot) dot.className = `dot ${kind}`;
      })
      .catch(() => {});
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
