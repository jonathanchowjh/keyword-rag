// Shared bits of the console: data access, number and time formatting, the sidebar.
// An ES module with no dependencies; every page imports what it needs.

const MINUS = "−";
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const POLL_MS = 15000;

// Job statuses: still to finish, and stopped short so they may resume.
export const BUSY = new Set(["queued", "running"]);
export const RESUMABLE = new Set(["interrupted", "cancelled", "stopped_budget", "stopped_no_key", "failed"]);

export const $ = (id) => document.getElementById(id);

const painted = {}; // element id -> html last drawn, so a refresh never rebuilds what has not changed

// Draws html into the element with this id unless it is already there; true when it drew.
export function paint(id, html) {
  if (painted[id] === html) return false;
  painted[id] = html;
  $(id).innerHTML = html;
  return true;
}

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

// A whole number with a true minus: "−5"; "–" when missing.
export function num(n) {
  return Number.isFinite(n) ? signed(whole(n)) : NONE;
}

// "2–10", or "−12 to −2" once a bound is negative; "" when a bound is missing.
export function between(lo, hi) {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return "";
  const l = whole(lo);
  const h = whole(hi);
  return l >= 0 && h >= 0 ? `${l}–${h}` : `${signed(l)} to ${signed(h)}`;
}

// "+6 pts (2–10)", "−7 pts (−12 to −2)": whole points, true minus, "pt" for exactly one.
// The range is left out when its bounds are missing.
export function pts(diff, lo, hi) {
  if (!Number.isFinite(diff)) return NONE;
  const d = whole(diff);
  const unit = Math.abs(d) === 1 ? "pt" : "pts";
  const text = `${d < 0 ? MINUS : "+"}${Math.abs(d)} ${unit}`;
  const range = between(lo, hi);
  return range ? `${text} (${range})` : text;
}

export const plural = (n, noun) => `${count(n)} ${noun}${n === 1 ? "" : "s"}`;

// Runs, jobs and keyword-list versions go by their first 8 characters.
export const short = (id) => String(id ?? "").slice(0, 8);

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
// method spends only on a collection with summaries (`summarize`), and is unmarked elsewhere.
export function creditMark(id, summarize, cat = known) {
  const paid = cat?.methods?.[id]?.paid;
  const note = paid === "always" ? "uses credit" : paid === "with_summaries" && summarize ? "uses credit" : "";
  return note ? ` <span class="credit muted">${esc(note)}</span>` : "";
}

export function metricLabel(id, qaType, cat = known) {
  return cat?.metric_labels_by_type?.[qaType]?.[id] ?? cat?.metrics?.[id]?.label ?? id;
}

// What a metric measures, in a sentence; "" when the catalog has nothing.
export function metricSays(id, qaType, cat = known) {
  return cat?.metric_says_by_type?.[qaType]?.[id] ?? cat?.metrics?.[id]?.says ?? "";
}

// Counts with thousands separators: 11,656.
export function count(n) {
  return Number(n).toLocaleString("en-US");
}

// Money: two decimals, "< $0.01" below a cent, so a paid run never shows $0.00.
export function usd(x) {
  const v = Number(x ?? 0);
  return v > 0 && v < 0.01 ? "< $0.01" : `$${v.toFixed(2)}`;
}

const FREE_SCORE_NOTE = "free score, not yet checked against answers";
export const NO_CHANCE = "no chance level";
const WORDS = { yes: "Yes", no: "No", cant_tell: "Can't tell" };

// What the range in brackets and the verdict words mean, said once where verdicts appear.
export const LEGEND =
  "In brackets: the range the gap likely lies in (95%, from resampling the same questions). " +
  "Yes or No: the whole range is on one side of 0. Can't tell: it spans 0.";

export function legendHtml(extra = "") {
  return `<p class="muted legend">${esc(extra ? `${LEGEND} ${extra}` : LEGEND)}</p>`;
}

// True when the headline is the free reasoning match and nothing has vouched for it (spec §7.3).
function freeScore(answer) {
  return answer.metric === "solution_sim@1" && answer.proxy_trusted !== true;
}

// The caveats on a verdict: a free score nothing has vouched for, a reasoning match shown raw.
export function verdictNotes(answer, { raw = false } = {}) {
  const notes = [];
  if (freeScore(answer)) notes.push(FREE_SCORE_NOTE);
  if (raw) notes.push(`reasoning match shown raw: this run has no random example, so ${NO_CHANCE}`);
  return notes;
}

// The verdict word, bold and coloured by its state: "Yes", "No" or "Can't tell".
export function wordHtml(answer) {
  const kind = answer.verdict in WORDS ? answer.verdict : "cant_tell";
  return `<strong class="word ${kind}">${WORDS[kind]}</strong>`;
}

// "Keywords per list +6 pts (2–10) answered correctly vs Hybrid": the verdict after its word.
export function verdictRest(answer, qaType, labels = known) {
  const cand = methodLabel(answer.candidate, labels);
  const ref = methodLabel(answer.reference, labels);
  const metric = metricLabel(answer.metric, qaType, labels).toLowerCase();
  return `${cand} ${pts(answer.diff, answer.lo, answer.hi)} ${metric} vs ${ref}`;
}

// The verdict as plain text: "Yes: Keywords per list +6 pts (2–10) answered correctly vs Hybrid".
export function verdictText(answer, qaType, labels = known) {
  if (!answer) return "No answer yet";
  const word = WORDS[answer.verdict] ?? WORDS.cant_tell;
  const notes = verdictNotes(answer).map((n) => ` · ${n}`).join("");
  return `${word}: ${verdictRest(answer, qaType, labels)}${notes}`;
}

// "<b>Yes</b>: Keywords per list +6 pts (2–10) answered correctly vs Hybrid · free score, …"
export function verdictHtml(answer, qaType, labels = known) {
  if (!answer) return '<span class="muted">No answer yet</span>';
  const notes = verdictNotes(answer).map((n) => ` · ${n}`).join("");
  return `${wordHtml(answer)}: ${esc(verdictRest(answer, qaType, labels))}${
    notes ? `<span class="muted">${esc(notes)}</span>` : ""}`;
}

// The axis half-width for gap bars: at least 10 points, in steps of 5, wide enough for them all.
export function gapSpan(values) {
  const biggest = Math.max(0, ...values.map(Math.abs).filter(Number.isFinite));
  return Math.max(10, Math.ceil(biggest / 5) * 5);
}

// The ends and middle of a gap bar's axis, as wide as the bar: "−10   0   +10 pts".
export function axisHtml(span) {
  return `<span class="axis"><span>${MINUS}${span}</span><span class="zero">0</span><span>+${span} pts</span></span>`;
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
  const money = usd(estimate.est_usd);
  const calls = Number(estimate.calls ?? 0);
  const what = calls === 1 ? "1 Claude call" : `${calls} Claude calls`;
  const cost = money.startsWith("<") ? money : `≈ ${money}`;
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

// Resuming a job: the click asks only for the estimate, or the refusal; spending needs Start.
// `box` shows the estimate, then any refusal or error; `done` runs once the job is requeued.
export async function askResume(box, jobId, done) {
  box.innerHTML = "";
  const path = `/jobs/${encodeURIComponent(jobId)}/resume`;
  try {
    const { estimate } = await api(path, { method: "POST", body: { confirm: false } });
    confirmSpend(box, estimate, async () => {
      try {
        await api(path, { method: "POST", body: { confirm: true } });
        box.innerHTML = "";
        refreshSpend();
        await done();
      } catch (e) {
        box.innerHTML = `<p class="err" role="status">${esc(e.message)}</p>`;
      }
    });
  } catch (e) {
    box.innerHTML = `<p class="warn" role="status" data-refusal>${esc(e.message)}</p>`; // the refusal
  }
}

// Asks inline before deleting: the question, then "Yes, …" or "Keep it". Deleting goes to `gone`.
export function askDelete(box, { question, yes, path, gone }) {
  box.innerHTML = `<p>${esc(question)}</p><div class="actions">
    <button type="button" class="danger" data-yes>${esc(yes)}</button>
    <button type="button" class="secondary" data-keep>Keep it</button></div>`;
  box.querySelector("[data-keep]").addEventListener("click", () => (box.innerHTML = ""));
  box.querySelector("[data-yes]").addEventListener("click", async () => {
    try {
      await api(path, { method: "DELETE" });
      location.href = gone;
    } catch (e) {
      box.innerHTML = `<p class="err" role="status">${esc(e.message)}</p>`;
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
    ${link("/worker", 'Worker <span id="worker-dot" class="dot" role="img" aria-label="state not known yet"></span>', "", active, "worker")}
    <a id="worker-problem" class="problem" href="/worker"></a>
    <a href="/docs">API docs</a>
    <div class="spend" id="spend"></div>`;
}

// The sidebar's spend line; pages call it after anything that may have spent.
export function refreshSpend() {
  return api("/budget")
    .then((b) => {
      const el = document.getElementById("spend");
      if (el) el.textContent = `${usd(b.spent_usd)} of $${money(b.cap_usd)} this month`;
    })
    .catch(() => {});
}

function refreshDot() {
  return api("/status")
    .then((s) => {
      const kind = s.problems.length ? "bad" : s.worker.running ? "run" : "ok";
      const said = { bad: "needs attention", run: "running a job", ok: "idle" }[kind];
      const dot = document.getElementById("worker-dot");
      if (dot) {
        dot.className = `dot ${kind}`;
        dot.title = `Worker: ${said}`;
        dot.setAttribute("aria-label", said);
      }
      const box = document.getElementById("worker-problem");
      if (box) {
        const more = s.problems.length - 1;
        box.innerHTML = s.problems.length
          ? `${esc(s.problems[0])}${more ? ` and ${more} more` : ""}`
          : "";
      }
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
