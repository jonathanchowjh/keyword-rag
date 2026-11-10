// Shared bits of the console: data access, number and time formatting, the sidebar.
// An ES module with no dependencies; every page imports what it needs.

const MINUS = "−";
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const POLL_MS = 15000;

// Job statuses: still to finish, and stopped short so they may resume.
export const BUSY = new Set(["queued", "running"]);
export const RESUMABLE = new Set(["interrupted", "cancelled", "stopped_budget", "stopped_no_key", "stopped_replies", "failed"]);

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

async function send(path, init) {
  const response = await fetch(path, init);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail;
    const error = new Error(typeof detail === "string" ? detail : response.statusText || "request failed");
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
  return `${line}<details class="evidence"${toggle}${open ? " open" : ""}><summary>${esc(text.slice(0, limit))}…</summary><div class="full">${esc(text)}</div></details>`;
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
  return `<label>Name<input id="f-name" autocomplete="off" pattern="[A-Za-z0-9_-]{1,64}" value="${esc(value)}"></label>
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

// <option>s for the models a job may use: enabled and priced, by label, sending the ref.
export function modelOptions(models, chosen) {
  return models
    .filter((m) => m.enabled && m.price_in != null && m.price_out != null)
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

// The verdict as plain text: "Yes: Keywords per list +6 pts (2–10) answered correctly vs Hybrid".
export function verdictText(answer, qaType, labels = known) {
  if (!answer) return "No answer yet";
  const word = WORDS[answer.verdict] ?? WORDS.cant_tell;
  const notes = verdictNotes(answer).map((n) => ` · ${n}`).join("");
  return `${word}: ${verdictRest(answer, qaType, labels)}${notes}`;
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

// A Documents store with no score for its chats' search: run it, or why it can't run here.
export function noScoreLine(method, why, labels = known) {
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

// Renders "≈ $0.41 · 400 AI calls · [Start]" into el; onStart runs only on click.
export function confirmSpend(el, estimate, onStart) {
  const money = usd(estimate.est_usd);
  const calls = Number(estimate.calls ?? 0);
  const what = calls === 1 ? "1 AI call" : `${calls} AI calls`;
  // No price when a job has no AI model set up: Start then gets the reason from the API.
  const cost = estimate.est_usd == null ? UNPRICED : money.startsWith("<") ? money : `≈ ${money}`;
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
// `current()` says whether this ask is still the one the box is for: a later click wins.
export async function askResume(box, jobId, done, current = () => true) {
  box.innerHTML = "";
  const path = `/jobs/${encodeURIComponent(jobId)}/resume`;
  try {
    const { estimate } = await api(path, { method: "POST", body: { confirm: false } });
    if (!current()) return;
    confirmSpend(box, estimate, async () => {
      try {
        await api(path, { method: "POST", body: { confirm: true, calls: estimate.calls, est_usd: estimate.est_usd } });
        box.innerHTML = "";
        refreshSpend();
        await done();
      } catch (e) {
        box.innerHTML = `<p class="err" role="status">${esc(e.message)}</p>`;
      }
    });
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

// A /status problem line as the pages say it: the server names a job by its kind, and a tag
// build's kind reads as what it does.
export function problemText(line) {
  return String(line).replace(/^Tag_build /, "Adding keyword tags ");
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
          ? `${esc(problemText(s.problems[0]))}${more ? ` and ${more} more` : ""}`
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
  const draw = () => {
    side.innerHTML = sideHtml(active, names, sets);
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
  api("/eval-sets")
    .then((list) => {
      sets = list.map((s) => s.name);
      draw();
    })
    .catch(() => {});
  setInterval(() => {
    refreshSpend();
    refreshDot();
  }, POLL_MS);
}

// -- importing a file ----------------------------------------------------------------------------

// The upload-preview-import widget for a store's items (queues a job, priced first) or an eval set's
// questions (imported in the request, no estimate). Fills `el`; `done` runs after an import.
// Items: the price sits beside Import, and the click on Import is the confirm; the line then
// follows the job to what it added, counted in `noun` (the store type's item noun).
// A Documents store's items (`qaType` "documents") are .md and .txt files of passages: the
// preview shows the passages, with no column mapping to pick, and the guessed one is sent.
class ImportBox {
  constructor(el, { uploadPath, importPath, kind, qaType, typeLabel, noun = "item", done, next }) {
    this.el = el;
    this.noun = noun;
    this.uploadPath = uploadPath;
    this.importPath = importPath;
    this.kind = kind;
    this.done = done;
    this.next = next; // () => html appended to a finished import's line, or undefined
    this.passages = kind === "items" && qaType === "documents";
    this.guessed = {}; // the preview's own mapping, sent when there are no selects
    this.uploadId = null; // the upload being previewed, or null
    this.batch = null; // several files being previewed (a Documents store), or null
    this.ticket = 0; // the newest preview request, so an older answer is ignored
    this.own = (what) => el.querySelector(`#imp-${kind}-${what}`);
    const accept = this.passages ? ".md,.txt" : ".jsonl,.json,.csv,.tsv,.txt,.md,.zip";
    const hint = kind === "questions" && qaType ? `<p class="muted">Questions to add to this eval set (${esc(typeLabel ?? qaType)}).</p>` : "";
    el.innerHTML = `<div class="card form">${hint}
      <label>File<input type="file" id="imp-${kind}-file" accept="${accept}"${this.passages ? " multiple" : ""}></label>
      <label>Source name<input id="imp-${kind}-source" maxlength="64"></label>
      <div class="row"><button type="button" id="imp-${kind}-go">Preview</button>
        <span id="imp-${kind}-msg" class="err" role="status"></span></div>
      <div id="imp-${kind}-preview"></div></div>`;
    this.own("file").addEventListener("change", (e) => {
      // Several files: each is its own source, named by its upload.
      this.own("source").closest("label").hidden = e.target.files.length > 1;
      this.uploadId = null; // the old preview belongs to the old file
      this.batch = null;
      this.ticket++; // and no answer still on its way may paint it back
      this.own("preview").innerHTML = "";
      this.own("msg").textContent = "";
      this.own("source").value = ""; // the upload names it, the route's rule
    });
    this.own("go").addEventListener("click", () => this.upload());
    // Items are priced under the source name, so a new name or mapping needs a new estimate.
    // Questions have no estimate to redo.
    if (kind === "items") {
      this.own("source").addEventListener("input", () => this.uploadId && this.previewAgain());
      this.own("preview").addEventListener("change", (e) => {
        if (this.uploadId && e.target.closest("select[data-field]")) this.previewAgain();
      });
    }
  }

  async upload() {
    const files = [...this.own("file").files];
    const file = files[0];
    const msg = this.own("msg");
    msg.textContent = "";
    if (!file) {
      msg.textContent = "Pick a file first.";
      return;
    }
    if (files.length > 1) return this.previewMany(files);
    try {
      const path = `${this.uploadPath}?filename=${encodeURIComponent(file.name)}&kind=${this.kind}`;
      const got = await upload(path, file);
      this.uploadId = got.upload_id;
      this.ticket++; // a new preview: an import line or an estimate still on its way paints nothing
      this.show(got.preview, got.estimate);
      // The upload is priced under its cleaned file name; a name typed first is priced anew.
      const typed = this.own("source").value;
      if (!typed) this.own("source").value = got.source_name;
      else if (typed !== got.source_name && this.kind === "items") this.previewAgain();
    } catch (e) {
      this.uploadId = null;
      this.own("preview").innerHTML = "";
      msg.textContent = e.message;
    }
  }

  mapping() {
    if (this.passages) return this.guessed;
    const out = {};
    for (const select of this.own("preview").querySelectorAll("select[data-field]")) {
      out[select.dataset.field] = select.value || null;
    }
    return out;
  }

  body(confirm) {
    const body = { upload_id: this.uploadId, mapping: this.mapping(), source_name: this.own("source").value };
    return this.kind === "items" ? { ...body, kind: "items", confirm } : body;
  }

  async previewAgain() {
    const msg = this.own("msg");
    const ticket = ++this.ticket;
    msg.textContent = "";
    this.own("import")?.setAttribute("disabled", ""); // until the fresh estimate is in
    try {
      const got = await api(this.importPath, { method: "POST", body: this.body(false) });
      if (ticket === this.ticket) this.show(got.preview, got.estimate);
    } catch (e) {
      if (ticket === this.ticket) msg.textContent = e.message;
    }
  }

  // A Documents store's preview: how many passages, and the first few.
  passagesHtml(p) {
    const sample = p.sample.length
      ? `<div class="scroll"><table><thead><tr><th>Passage</th></tr></thead>
          <tbody>${p.sample.map((row) => `<tr><td>${esc(row.text ?? "")}</td></tr>`).join("")}</tbody></table></div>`
      : "";
    return `<p>${esc(p.format)} · ${esc(plural(p.rows, "passage"))} · ${esc(count(p.usable))} usable</p>${sample}`;
  }

  html(p, estimate) {
    const problems = p.problems.length
      ? `<ul class="warn">${p.problems.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`
      : "";
    if (this.passages) return this.passagesHtml(p) + problems + this.footHtml(p, estimate);
    const sample = p.sample.length
      ? `<div class="scroll"><table><thead><tr>${p.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
          <tbody>${p.sample.map((row) => `<tr>${p.columns.map((c) => `<td>${esc(row[c] ?? "")}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`
      : "";
    const selects = Object.keys(p.mapping)
      .map((field) => {
        const options = ['<option value="">(none)</option>']
          .concat(p.columns.map((c) => `<option value="${esc(c)}"${p.mapping[field] === c ? " selected" : ""}>${esc(c)}</option>`))
          .join("");
        const star = p.required.includes(field) ? " (required)" : "";
        return `<label>${esc(field)}${star}<select data-field="${esc(field)}">${options}</select></label>`;
      })
      .join("");
    const missing = p.missing.length ? `<p class="err">Map these fields first: ${esc(p.missing.join(", "))}.</p>` : "";
    return `<p>${esc(p.format)} · ${esc(plural(p.rows, "row"))} · ${esc(count(p.usable))} usable</p>${problems}${sample}
      <div class="row">${selects}</div>${missing}${this.footHtml(p, estimate)}`;
  }

  // The Import button with what the import costs.
  footHtml(p, estimate) {
    const cost = estimate
      ? `<span class="muted">${esc(estimate.est_usd == null ? UNPRICED : usd(estimate.est_usd))} · ${esc(plural(estimate.calls ?? 0, "AI call"))}</span>`
      : "";
    // A questions import cannot be previewed again under a new mapping: the server judges it.
    const open = p.ready || this.kind === "questions";
    // Items have no Start: the click on Import, beside the price, is the confirm.
    const confirm = this.kind === "questions" ? '<span id="imp-questions-confirm"></span>' : "";
    const removed = estimate?.removed > 0
      ? `<p class="warn" data-removed>${esc(`${plural(estimate.removed, "old " + this.noun)} will be removed, since the text changed.`)}</p>`
      : "";
    return `<div class="row"><button type="button" id="imp-${this.kind}-import" ${open ? "" : "disabled"}>Import</button>
        ${cost}${confirm}</div>${removed}`;
  }

  show(preview, estimate) {
    this.guessed = preview.mapping;
    this.own("preview").innerHTML = this.html(preview, estimate);
    this.own("import").addEventListener("click", () => this.start(estimate));
  }

  async start(estimate) {
    const msg = this.own("msg");
    msg.textContent = "";
    if (this.kind === "items") return this.queueItems(estimate);
    const queue = async () => {
      try {
        const got = await api(this.importPath, { method: "POST", body: this.body(true) });
        this.own("preview").innerHTML = `<p class="ok">Imported: ${esc(got.new)} new, ${esc(got.updated)} updated, ${esc(
          got.unchanged)} unchanged, ${esc(got.skipped)} skipped. Version ${esc(short(got.version))}.</p>`;
        this.uploadId = null;
        this.own("file").value = "";
        refreshSpend();
        await this.done?.();
      } catch (e) {
        msg.textContent = e.message;
      }
    };
    // An unpriced (null) estimate is not free: it is shown and confirmed, like a priced one.
    if (estimate && estimate.est_usd !== 0) confirmSpend(this.own("confirm"), estimate, queue);
    else await queue();
  }

  // Items: the click on Import, with its price beside it, queues the import (the server prices it
  // again first, and refuses it when its calls are not the `estimate` drawn here). The button stays
  // disabled, so a second click queues nothing; a refusal frees it. The click takes the upload
  // over: an edit while it is sent previews nothing, so no Import is armed for it again. The line
  // then follows the job, until a new file takes the box over.
  async queueItems(estimate) {
    const button = this.own("import");
    button.disabled = true;
    const ticket = this.ticket;
    const body = this.body(true);
    const uploadId = this.uploadId;
    this.uploadId = null;
    let got;
    try {
      const shown = { calls: estimate?.calls, est_usd: estimate?.est_usd };
      got = await api(this.importPath, { method: "POST", body: { ...body, ...shown } });
    } catch (e) {
      if (ticket === this.ticket) {
        this.uploadId = uploadId; // a refusal can be tried again
        // edited while it was sent, or refused as it stands (a 409, such as a price that
        // changed since it was drawn): priced again first
        const same = JSON.stringify(this.body(true)) === JSON.stringify(body);
        if (same && e.status !== 409) button.disabled = false;
        else this.previewAgain();
      }
      this.own("msg").textContent = e.message;
      return;
    }
    const line = (html) => {
      if (ticket === this.ticket) this.own("preview").innerHTML = html;
    };
    if (ticket === this.ticket) this.own("file").value = ""; // else a new file or upload took over
    let job;
    try {
      job = await followJob(got.job_id, (j) => line(j.status === "queued"
        ? "<p>Waiting for the store's other job…</p>"
        : `<p>Adding… ${esc(Math.round((j.progress ?? 0) * 100))}%</p>`));
    } catch (e) {
      line(`<p class="err">${esc(e.message)}</p>`);
      return;
    }
    line(this.doneHtml(job));
    refreshSpend();
    await this.done?.();
  }

  // Several files (a Documents store): each is previewed in turn, then priced and imported as one.
  async previewMany(files) {
    const ticket = ++this.ticket;
    this.uploadId = null;
    this.batch = null;
    this.own("preview").innerHTML = "";
    const entries = [];
    for (const file of files) {
      const path = `${this.uploadPath}?filename=${encodeURIComponent(file.name)}&kind=${this.kind}`;
      try {
        entries.push({ name: file.name, up: await upload(path, file) });
      } catch (e) {
        entries.push({ name: file.name, error: e.message }); // wrong type, too big, unreadable
      }
      if (ticket !== this.ticket) return; // a new file took the box over
    }
    this.showMany(entries);
  }

  // The files' previews in one summary line and a line each, with one Import for them all. Each
  // file's own price is drawn on a store as it stands, so with several the sum errs high.
  showMany(entries, before = "") {
    this.batch = entries;
    const ready = entries.filter((f) => f.up?.estimate);
    const adds = (f) => Number(f.up?.estimate?.new ?? 0) + Number(f.up?.estimate?.updated ?? 0);
    const added = ready.reduce((n, f) => n + adds(f), 0);
    const removed = ready.reduce((n, f) => n + Number(f.up.estimate.removed ?? 0), 0);
    const unpriced = ready.find((f) => f.up.estimate.est_usd === null && f.up.estimate.calls > 0);
    const total = usd(ready.reduce((n, f) => n + Number(f.up.estimate.est_usd ?? 0), 0));
    const price = unpriced ? UNPRICED : ready.length > 1 ? `at most ${total}` : total;
    const gone = removed > 0 ? `, ${count(removed)} to remove` : "";
    const each = entries
      .map((f) => `<li>${esc(f.error ? `${f.name}: ${f.error}` : `${f.name}: ${plural(adds(f), this.noun)}`)}</li>`)
      .join("");
    const why = unpriced ? `<span class="err">${esc(`Can't price ${unpriced.name}: set up a model in Settings`)}</span>` : "";
    const summary = ready.length
      ? `<p>${esc(`${plural(ready.length, "file")}: ${plural(added, this.noun)} to add${gone} · ${price}`)}</p>`
      : "";
    const foot = ready.length
      ? `<div class="row"><button type="button" id="imp-${this.kind}-import" ${unpriced ? "disabled" : ""}>${esc(
          unpriced ? "Import" : `Import (${price})`)}</button>${why}</div>`
      : "";
    this.own("preview").innerHTML = `${before}${summary}<ul>${each}</ul>${foot}`;
    this.own("import")?.addEventListener("click", () => this.importMany());
  }

  // Imports the files in order, each after the one before is in. The first sends the calls and
  // price it was shown; a later one only its price, since an earlier file can change its calls
  // (the server then refuses only a price that rose). A refusal stops there: the files not yet
  // in are priced again, for the next click.
  async importMany() {
    this.own("import").disabled = true;
    const ticket = this.ticket;
    const todo = this.batch.filter((f) => f.up?.estimate);
    this.batch = null; // the click takes the files over
    const line = (html) => {
      if (ticket === this.ticket) this.own("preview").innerHTML = html;
    };
    const sum = { new: 0, updated: 0, unchanged: 0, removed: 0 };
    const sofar = () => (sum.new + sum.updated + sum.unchanged + sum.removed > 0
      ? this.doneHtml({ status: "done", result: sum }) : "");
    // The store's page shows what is in, as the single-file import does after its job.
    const settled = async () => {
      refreshSpend();
      await this.done?.();
    };
    for (const [i, f] of todo.entries()) {
      const { calls, est_usd } = f.up.estimate;
      const shown = i === 0 ? { calls, est_usd } : { est_usd };
      let got;
      try {
        got = await api(this.importPath, { method: "POST", body: { ...manyBody(f.up, true), ...shown } });
      } catch (e) {
        if (ticket !== this.ticket) return;
        this.own("msg").textContent = e.status === 409 ? `The price of ${f.name} changed: check it again` : e.message;
        await this.repriceMany(todo.slice(i), ticket, sofar());
        if (sofar()) await settled(); // the files before this one are in
        return;
      }
      let job;
      try {
        job = await followJob(got.job_id, (j) => line(j.status === "queued"
          ? "<p>Waiting for the store's other job…</p>"
          : `<p>${esc(`Adding ${f.name}… ${Math.round((j.progress ?? 0) * 100)}%`)}</p>`));
      } catch (e) {
        line(sofar() + `<p class="err">${esc(e.message)}</p>`);
        if (sofar()) await settled();
        return;
      }
      if (job.status !== "done") {
        line(sofar() + this.doneHtml(job));
        await settled(); // a stopped job may have added some, as the single-file import allows
        return;
      }
      for (const key of Object.keys(sum)) sum[key] += Number(job.result?.[key] ?? 0);
    }
    line(sofar());
    if (ticket === this.ticket) {
      this.own("file").value = ""; // else a new file took over
      this.own("source").closest("label").hidden = false;
    }
    await settled();
  }

  // The files not yet imported, priced again on the store as it now is.
  async repriceMany(entries, ticket, before) {
    const priced = [];
    for (const f of entries) {
      try {
        const got = await api(this.importPath, { method: "POST", body: manyBody(f.up, false) });
        priced.push({ name: f.name, up: { ...f.up, ...got } });
      } catch (e) {
        priced.push({ name: f.name, error: e.message });
      }
      if (ticket !== this.ticket) return;
    }
    this.showMany(priced, before);
  }

  // What a finished import job did, in a line.
  doneHtml(job) {
    if (job.status !== "done") {
      const why = job.error ? `: ${job.error}` : ".";
      return `<p class="err">${esc(`The import ${job.status.replaceAll("_", " ")}${why}`)}</p>`;
    }
    const r = job.result ?? {};
    const added = Number(r.new ?? 0) + Number(r.updated ?? 0);
    const kept = Number(r.unchanged ?? 0);
    const already = kept > 0 ? ` ${count(kept)} already in the store.` : "";
    const removed = Number(r.removed ?? 0);
    const gone = removed > 0 ? ` ${plural(removed, "old " + this.noun)} removed.` : "";
    const next = this.next ? ` Next: ${this.next()}` : "";
    return `<p class="ok">${esc(`Added ${plural(added, this.noun)}.${already}${gone}`)}${next}</p>`;
  }
}

// One of several files' import, under the upload's own mapping and source name.
function manyBody(up, confirm) {
  return { upload_id: up.upload_id, kind: "items", mapping: up.preview.mapping, source_name: up.source_name, confirm };
}

export function importBox(el, options) {
  return new ImportBox(el, options);
}
