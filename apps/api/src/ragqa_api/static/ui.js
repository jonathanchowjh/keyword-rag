// Shared bits of the console: data access, number and time formatting, the sidebar.
// An ES module with no dependencies; every page imports what it needs.

const MINUS = "−";
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const POLL_MS = 15000;

// Job statuses: still to finish, and stopped short so they may resume.
export const BUSY = new Set(["queued", "running"]);
export const RESUMABLE = new Set(["interrupted", "cancelled", "stopped_budget", "stopped_no_key", "stopped_replies", "failed"]);

export const $ = (id) => document.getElementById(id);

// Says `text` in a message element, styled `cls` when given ("err", "ok", …).
export function say(el, text, cls) {
  el.textContent = text;
  if (cls !== undefined) el.className = cls;
}

// Runs `fn` again and again, waiting `ms` (a number, or a function giving one) before each
// run. A hidden tab sends nothing: the wait then lasts until the tab is shown, and `fn` runs
// at once. Returns stop().
export function poll(fn, ms) {
  let stopped = false;
  let timer = null;
  let resting = false; // waiting for the tab to be shown
  const wait = () => {
    if (!stopped) timer = setTimeout(run, typeof ms === "function" ? ms() : ms);
  };
  const run = async () => {
    if (stopped) return;
    if (document.hidden) {
      resting = true;
      return;
    }
    try {
      await fn();
    } finally {
      wait();
    }
  };
  const shown = () => {
    if (document.hidden || !resting) return;
    resting = false;
    run();
  };
  document.addEventListener("visibilitychange", shown);
  wait();
  return () => {
    stopped = true;
    clearTimeout(timer);
    document.removeEventListener("visibilitychange", shown);
  };
}

const painted = {}; // element id -> html last drawn, so a refresh never rebuilds what has not changed

// Draws html into the element with this id unless it is already there; true when it drew.
export function paint(id, html) {
  if (painted[id] === html) return false;
  painted[id] = html;
  $(id).innerHTML = html;
  return true;
}

// -- data ---------------------------------------------------------------------

// A refusal's reason: a plain detail as it is, a validation list as "field: why" items.
function reasonOf(detail) {
  if (typeof detail === "string") return detail;
  if (!Array.isArray(detail)) return "";
  return detail
    .map((d) => {
      const loc = (d.loc ?? []).filter((part, i) => !(i === 0 && part === "body")).join(".");
      return loc ? `${loc}: ${d.msg}` : d.msg;
    })
    .join("; ");
}

async function send(path, init) {
  const response = await fetch(path, init);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail;
    const error = new Error(reasonOf(detail) || response.statusText || "request failed");
    error.status = response.status;
    throw error;
  }
  return data;
}

export function api(path, { method = "GET", body } = {}) {
  const init = { method };
  if (body !== undefined) {
    init.headers = { "content-type": "application/json" };
    init.body = JSON.stringify(body);
  }
  return send(path, init);
}

// POSTs a File as the raw body, for the upload endpoints.
export function upload(path, file) {
  return send(path, { method: "POST", body: file });
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

// A source as people read it: "upload:leave" is "leave (uploaded)", "hand" is "added by hand".
export function sourceLabel(source) {
  if (source === "hand") return "added by hand";
  if (typeof source === "string" && source.startsWith("upload:")) return `${source.slice(7)} (uploaded)`;
  return source;
}

// A retrieved passage's text. Up to `limit` characters it is shown whole; longer, its first
// `limit` characters sit in a summary and the rest opens from it. `meta` is HTML for a small
// line above. `key` names the toggle (data-expand) so a page can keep it open across redraws,
// and `open` says whether it is open now.
export function passageHtml(text, { limit = 300, meta = "", key = null, open = false } = {}) {
  const line = meta ? `<div class="small">${meta}</div>` : "";
  if (text.length <= limit) return `${line}<div>${esc(text)}</div>`;
  const toggle = key ? ` data-expand="${esc(key)}"` : "";
  return `${line}<details class="passage"${toggle}${open ? " open" : ""}><summary>${esc(text.slice(0, limit))}…</summary><div class="full">${esc(text)}</div></details>`;
}

// "file · part 2 of 3" for a passage's meta, falling back to its id, with the id as its title.
// Part only when the file has more than one.
export function whereHtml({ id, file, part, parts }) {
  const label = file ? `${file}${parts > 1 ? ` · part ${part} of ${parts}` : ""}` : id;
  return `<span title="${esc(id)}">${esc(label)}</span>`;
}

// A name a store or eval set may carry; the same rule as ragqa_core.types.slug.
export function slug(name) {
  const plain = String(name).normalize("NFKD").replace(/\p{M}/gu, "").toLowerCase();
  return plain.replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 64) || "store";
}

// The Name field of a create form (#f-name), with what is allowed and, while the entry is
// refused, a button that fills in the suggestion. Call watchName() once the html is in the page.
export function nameField(value = "") {
  return `<label>Name<input id="f-name" autocomplete="off" pattern="[A-Za-z0-9_\\-]{1,64}" value="${esc(value)}"></label>
    <p class="muted">Letters, digits, - or _; no spaces <button type="button" id="f-use" class="secondary" hidden></button></p>`;
}

export function watchName() {
  const input = $("f-name");
  const use = $("f-use");
  const show = () => {
    const bad = input.value.trim() !== "" && !/^[A-Za-z0-9_-]{1,64}$/.test(input.value.trim());
    use.hidden = !bad;
    if (bad) use.textContent = `Use ${slug(input.value.trim())}`;
  };
  input.addEventListener("input", show);
  use.addEventListener("click", () => {
    input.value = slug(input.value.trim());
    show();
    input.focus();
  });
  show();
}

export function esc(s) {
  const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  return String(s ?? "").replace(/[&<>"']/g, (c) => map[c]);
}

// -- formatting -----------------------------------------------------------------

export const NONE = "–"; // shown for a missing or invalid value

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

// The label, with a note when the method may spend on AI.
export function mayCost(id, cat = known) {
  const paid = cat?.methods?.[id]?.paid;
  return methodLabel(id, cat) + (paid && paid !== "never" ? " (may cost AI)" : "");
}

export const capital = (text) => text.charAt(0).toUpperCase() + text.slice(1);

// <option>s for the models a job may use: ready (usable and priced), by label, sending the ref.
export function modelOptions(models, chosen) {
  return models
    .filter((m) => m.ready)
    .map((m) => `<option value="${esc(m.ref)}"${m.ref === chosen ? " selected" : ""}>${esc(m.label)}</option>`)
    .join("");
}

// The muted "costs money" note beside a paid method (spec §7.1); "" for a free one. A keyword
// method spends only on a collection with summaries, and Keyword-RAG only when the AI picks
// the question's tags (search `query_tags: "model"`); unmarked elsewhere.
export function costMark(id, collection, cat = known) {
  const paid = cat?.methods?.[id]?.paid;
  const spends =
    paid === "always" ||
    (paid === "with_summaries" && collection?.summarize) ||
    (paid === "with_model_tags" && collection?.search?.query_tags === "model");
  return spends ? ` <span class="cost muted">costs money</span>` : "";
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
export const UNPRICED = "cost unknown: no AI model is set up";

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
export function verdictNotes(answer, { raw = answer.scale?.unit === "raw" } = {}) {
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

// A score and its likely range: "72% (64–80)", whole and unsigned; "–" when missing.
export function scoreRange(mean, lo, hi) {
  if (!Number.isFinite(mean)) return NONE;
  const range = Number.isFinite(lo) && Number.isFinite(hi) ? ` (${Math.round(lo)}–${Math.round(hi)})` : "";
  return `${Math.round(mean)}%${range}`;
}

// A Documents store's answer: the score of the search its chats use, and what weakens it.
// A score most of whose top 10 was never judged is withheld.
export function scoreLine(answer, labels = known) {
  const label = methodLabel(answer.method, labels);
  if (answer.unreliable) {
    return `No score yet: most of the top 10 from ${label} were never judged. Label more of its passages.`;
  }
  const notes = [];
  if (answer.before_rerank) notes.push("measured before reranking");
  if (answer.n < 10) notes.push(`only ${plural(answer.n, "question")}: label more for a firmer score`);
  if (answer.newer_without > 0) notes.push(`the newest run didn't include ${label}`);
  if (answer.unfinished_newer) notes.push(`the last run of ${label} didn't finish`);
  const line = `${metricLabel("ndcg@10", "documents", labels)}: ${scoreRange(answer.mean, answer.lo, answer.hi)} for ${
    label}, the search your chats use, on ${plural(answer.n, "question")}`;
  return [line, ...notes].join(" · ");
}

// A Documents store with no score for its chats' search: the next step its state calls for. `store`
// is a GET /collections/{n} reply.
export function noScoreLine(store, labels = known) {
  if (!store.count) return "No score yet: add your documents to start";
  if (!store.eval_set || !store.questions) return "No score yet: label a few questions to get one";
  const method = store.chat_search?.run_method;
  const why = store.methods_unavailable?.[method];
  const label = methodLabel(method, labels);
  return why ? `No score yet: ${label} can't run on this store (${why})` : `No score yet: run ${label} to get one`;
}

// "<b>Yes</b>: Keywords per list +6 pts (2–10) answered correctly vs Hybrid · free score, …"
export function verdictHtml(answer, qaType, labels = known) {
  if (!answer) return '<span class="muted">No answer yet</span>';
  if (answer.kind === "score") return esc(scoreLine(answer, labels));
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

// An embedder's option text: name, dimensions, size and note, then its timing when it has one.
export function embedderLabel(e, machine, recommended) {
  const timed = e.seconds_per_1000 != null && machine
    ? `about ${Math.round(e.seconds_per_1000)} s per 1,000 passages on ${machine}` : "";
  const parts = [e.name, e.dim == null ? "" : `${e.dim} dimensions`, e.size_gb == null ? "" : `${e.size_gb} GB`, e.note ?? "", timed]
    .filter(Boolean).join(" · ");
  return e.name === recommended ? `${parts} (recommended)` : parts;
}

// -- layout ---------------------------------------------------------------------

export function pill(state) {
  const kind =
    { ready: "ok", done: "ok", loading: "run", rekeying: "run", retagging: "run", tagging: "run", copying: "run", evaluating: "run", importing: "run", labelling: "run", rebuilding: "run", running: "run",
      busy: "run", queued: "idle", failed: "bad", interrupted: "bad", stopped_budget: "bad",
      stopped_no_key: "bad", stopped_replies: "idle", cancelled: "idle" }[state] ?? "idle";
  const text = String(state ?? "").replace(/_/g, " ").replace("rekeying", "re-keying").replace("retagging", "re-tagging");
  return `<span class="pill ${kind}">${esc(capital(text))}</span>`;
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

// A page whose subject does not exist: the whole main area says so, with a way back, and
// nothing in it can be pressed.
export function notFound({ title, crumb = [], text, links = [] }) {
  const way = links.map((l) => `<a href="${esc(l.href)}">${esc(l.label)}</a>`).join(" · ");
  $("main").innerHTML = head({ title, crumb }) + `<p class="err" data-not-found>${esc(text)}</p><p>${way}</p>`;
}

// The server's refusal of a Start whose shown price has changed (ragqa_api.guards.PRICE_CHANGED).
export const PRICE_CHANGED = "the price changed since you saw it: check it again";

// After that refusal the old Start would resend the same stale price: take it away, so the next
// click on the action prices it again. Any other error leaves the Start where it is.
function dropStaleStart(el, e) {
  if (e.status === 409 && e.message === PRICE_CHANGED) el.innerHTML = "";
}

export const CONNECT_AI = '<a href="/settings">Connect AI first</a>';

// True when a role the estimate's job needs has no usable model; a fake that omits `missing` needs none.
export const needsAI = (estimate) => (estimate?.missing?.length ?? 0) > 0;

// What was shown, as the server checks it: a price that changed since is refused.
export const shownPrice = (estimate) => ({ calls: estimate?.calls, est_usd: estimate?.est_usd });

// "≈ $0.41", "< $0.01" (never "≈ < $0.01"), or UNPRICED when no AI model is set up.
export function priceText(estimate) {
  if (estimate.est_usd == null) return UNPRICED;
  const money = usd(estimate.est_usd);
  return money.startsWith("<") ? money : `≈ ${money}`;
}

// Renders "≈ $0.41 · 400 AI calls [Start]" into el, or "Connect AI first" in Start's place when a
// role has no usable model. onStart({confirm, calls, est_usd}) runs only on click. A throw from it
// is shown in msgEl (when given); only a changed price also takes Start away.
export function confirmSpend(el, estimate, onStart, msgEl = null) {
  const calls = Number(estimate.calls ?? 0);
  const what = calls === 1 ? "1 AI call" : `${calls} AI calls`;
  const start = needsAI(estimate) ? CONNECT_AI : '<button type="button">Start</button>';
  el.innerHTML = `<span class="confirm">${esc(priceText(estimate))} · ${esc(what)} ${start}</span>`;
  const button = el.querySelector("button");
  button?.addEventListener("click", async () => {
    button.disabled = true;
    if (msgEl) say(msgEl, "");
    try {
      await onStart({ confirm: true, ...shownPrice(estimate) });
    } catch (e) {
      dropStaleStart(el, e);
      if (msgEl) say(msgEl, e.message);
    } finally {
      button.disabled = false;
    }
  });
}

// Resuming a job: the click asks only for the estimate, or the refusal; spending needs Start.
// `box` shows the estimate, then any refusal or error; `done` runs once the job is requeued.
// `current()` says whether this ask is still the one the box is for: a later click wins.
export async function askResume(box, jobId, done, current = () => true) {
  box.innerHTML = "";
  const path = `/jobs/${encodeURIComponent(jobId)}/resume`;
  try {
    const { estimate } = await api(path, { method: "POST", body: { confirm: false } });
    if (!current()) return;
    if (!estimate) { // free (a rebuild): already requeued
      await done();
      return;
    }
    box.innerHTML = '<span></span><p class="err" role="status"></p>';
    confirmSpend(box.firstChild, estimate, async (shown) => {
      await api(path, { method: "POST", body: shown });
      box.innerHTML = "";
      refreshSpend();
      await done();
    }, box.lastChild);
  } catch (e) {
    if (!current()) return;
    box.innerHTML = `<p class="warn" role="status" data-refusal>${esc(e.message)}</p>`; // the refusal
  }
}

// Asks inline before deleting: the question, then "Yes, …" or "Cancel". Deleting goes to `gone`.
export function askDelete(box, { question, yes, path, gone }) {
  box.innerHTML = `<p>${esc(question)}</p><div class="actions">
    <button type="button" class="danger" data-yes>${esc(yes)}</button>
    <button type="button" class="secondary" data-keep>Cancel</button></div>`;
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

const THEMES = ["system", "light", "dark"];
const THEME_LABEL = { system: "System", light: "Light", dark: "Dark" };

// The saved choice; a throwing store reads as System. theme.js reads the same key.
function savedTheme() {
  try {
    const t = localStorage.getItem("theme");
    return t === "light" || t === "dark" ? t : "system";
  } catch (e) {
    return "system";
  }
}

function cycleTheme() {
  const next = THEMES[(THEMES.indexOf(savedTheme()) + 1) % THEMES.length];
  try {
    if (next === "system") localStorage.removeItem("theme");
    else localStorage.setItem("theme", next);
  } catch (e) {}
  if (next === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = next;
}

function sideHtml(active, names, sets) {
  const stores = names
    .map((n) => link(`/c/${encodeURIComponent(n)}`, esc(n), "sub", active, `c/${n}`))
    .join("");
  const evalSets = sets
    .map((n) => link(`/e/${encodeURIComponent(n)}`, esc(n), "sub", active, `e/${n}`))
    .join("");
  return `<div class="brand">keyword-rag</div><div class="tag">RAG bench and chat for your docs</div>
    ${link("/", "Stores", "", active, "collections")}${stores}
    ${link("/chat", "Chat", "", active, "chat")}
    ${link("/eval-sets", "Eval sets", "", active, "eval-sets")}${evalSets}
    ${link("/lab", "Lab", "", active, "lab")}
    ${link("/settings", "Settings", "", active, "settings")}
    <div class="group">BEHIND THE NUMBERS</div>
    ${link("/worker", 'Worker <span id="worker-dot" class="dot" role="img" aria-label="state not known yet"></span>', "", active, "worker")}
    <a id="worker-problem" class="problem" href="/worker"></a>
    <a href="/docs">API docs</a>
    <button type="button" class="theme" id="theme">Theme: ${THEME_LABEL[savedTheme()]}</button>
    <div class="spend" id="spend"></div>`;
}

// Follows a job until it finishes: `onWait(job)` on each poll that finds it queued or running,
// then the finished job. A failed poll rejects, for the caller to show.
export async function followJob(jobId, onWait, every = 500) {
  for (;;) {
    const job = await api(`/jobs/${encodeURIComponent(jobId)}`);
    if (!BUSY.has(job.status)) return job;
    onWait(job);
    await new Promise((resolve) => setTimeout(resolve, every));
  }
}

// The sidebar's spend line; pages call it after anything that may have spent.
export function refreshSpend() {
  return api("/budget")
    .then((b) => {
      const el = document.getElementById("spend");
      if (el) el.textContent = `Spent ${usd(b.spent_usd)} of $${money(b.cap_usd)} this month`;
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
          ? `${esc(s.problems[0].text)}${more ? ` and ${more} more` : ""}`
          : "";
      }
    })
    .catch(() => {});
}

// Renders the sidebar into <aside id="side">. `active` is "collections" (the stores), "eval-sets",
// "chat", "lab", "settings", "worker", "c/<name>" for a store's page, or "e/<name>" for an eval set's.
export function shell(active) {
  const side = document.getElementById("side");
  let names = [];
  let sets = [];
  side.addEventListener("click", (e) => {
    if (!e.target.closest("#theme")) return;
    cycleTheme();
    draw();
    document.getElementById("theme").focus(); // the redraw replaced the button: keep the keyboard on it
  });
  // A redraw fetches nothing: the spend line and the worker dot it replaces are carried over.
  const draw = () => {
    const kept = ["spend", "worker-dot", "worker-problem"].map((id) => document.getElementById(id));
    side.innerHTML = sideHtml(active, names, sets);
    for (const old of kept) if (old) document.getElementById(old.id)?.replaceWith(old);
  };
  draw();
  refreshSpend();
  refreshDot();
  Promise.allSettled([
    api("/collections").then((list) => (names = list.map((c) => c.name))),
    api("/eval-sets").then((list) => (sets = list.map((s) => s.name))),
  ]).then(draw);
  poll(() => Promise.all([refreshSpend(), refreshDot()]), POLL_MS);
}
