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
    throw new Error(typeof detail === "string" ? detail : response.statusText || "request failed");
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
    { ready: "ok", done: "ok", loading: "run", rekeying: "run", retagging: "run", copying: "run", evaluating: "run", importing: "run", labelling: "run", running: "run",
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
        await api(path, { method: "POST", body: { confirm: true } });
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
  const stores = names
    .map((n) => link(`/c/${encodeURIComponent(n)}`, esc(n), "sub", active, `c/${n}`))
    .join("");
  return `<div class="brand">rag-qa</div><div class="tag">keyword RAG test bench</div>
    ${link("/", "Stores", "", active, "collections")}${stores}
    ${link("/eval-sets", "Eval sets", "", active, "eval-sets")}
    ${link("/lab", "Lab", "", active, "lab")}
    ${link("/settings", "Settings", "", active, "settings")}
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
          ? `${esc(s.problems[0])}${more ? ` and ${more} more` : ""}`
          : "";
      }
    })
    .catch(() => {});
}

// Renders the sidebar into <aside id="side">. `active` is "collections" (the stores), "eval-sets",
// "lab", "settings", "worker", or "c/<name>" for a store's page.
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

// -- importing a file ----------------------------------------------------------------------------

// The file's name as a source name: other characters become "-", at most 64 long.
function sourceFor(file) {
  return file.name.replace(/\.[^.]*$/, "").replace(/[^A-Za-z0-9 _.-]/g, "-").slice(0, 64) || "upload";
}

// The upload-preview-import widget for a store's items (queues a job, priced first) or an eval set's
// questions (imported in the request, no estimate). Fills `el`; `done` runs after an import.
class ImportBox {
  constructor(el, { uploadPath, importPath, kind, qaType, done }) {
    this.el = el;
    this.uploadPath = uploadPath;
    this.importPath = importPath;
    this.kind = kind;
    this.done = done;
    this.uploadId = null; // the upload being previewed, or null
    this.ticket = 0; // the newest preview request, so an older answer is ignored
    this.own = (what) => el.querySelector(`#imp-${kind}-${what}`);
    const accept = ".jsonl,.json,.csv,.tsv,.txt,.md,.zip";
    const hint = kind === "questions" && qaType ? `<p class="muted">Questions for a ${esc(qaType)} eval set.</p>` : "";
    el.innerHTML = `<div class="card form">${hint}
      <label>File<input type="file" id="imp-${kind}-file" accept="${accept}"></label>
      <label>Source name<input id="imp-${kind}-source" maxlength="64"></label>
      <div class="row"><button type="button" id="imp-${kind}-go">Upload</button>
        <span id="imp-${kind}-msg" class="err" role="status"></span></div>
      <div id="imp-${kind}-preview"></div></div>`;
    this.own("file").addEventListener("change", (e) => {
      const file = e.target.files[0];
      this.uploadId = null; // the old preview belongs to the old file
      this.ticket++; // and no answer still on its way may paint it back
      this.own("preview").innerHTML = "";
      this.own("msg").textContent = "";
      if (file) this.own("source").value = sourceFor(file);
    });
    this.own("go").addEventListener("click", () => this.upload());
    // Items are priced under the source name, so a new name or mapping needs a new estimate.
    // Questions have no estimate to redo.
    if (kind === "items") {
      this.own("source").addEventListener("input", () => this.uploadId && this.previewAgain());
      this.own("preview").addEventListener("change", (e) => {
        if (e.target.closest("select[data-field]")) this.previewAgain();
      });
    }
  }

  async upload() {
    const file = this.own("file").files[0];
    const msg = this.own("msg");
    msg.textContent = "";
    if (!file) {
      msg.textContent = "Pick a file first.";
      return;
    }
    try {
      const path = `${this.uploadPath}?filename=${encodeURIComponent(file.name)}&kind=${this.kind}`;
      const got = await upload(path, file);
      this.uploadId = got.upload_id;
      this.show(got.preview, got.estimate);
    } catch (e) {
      this.uploadId = null;
      this.own("preview").innerHTML = "";
      msg.textContent = e.message;
    }
  }

  mapping() {
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

  html(p, estimate) {
    const problems = p.problems.length
      ? `<ul class="warn">${p.problems.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`
      : "";
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
    const cost = estimate
      ? `<span class="muted">${esc(estimate.est_usd == null ? UNPRICED : usd(estimate.est_usd))} · ${esc(plural(estimate.calls ?? 0, "AI call"))}</span>`
      : "";
    const missing = p.missing.length ? `<p class="err">Map these fields first: ${esc(p.missing.join(", "))}.</p>` : "";
    // A questions import cannot be previewed again under a new mapping: the server judges it.
    const open = p.ready || this.kind === "questions";
    return `<p>${esc(p.format)} · ${esc(plural(p.rows, "row"))} · ${esc(count(p.usable))} usable</p>${problems}${sample}
      <div class="row">${selects}</div>${missing}
      <div class="row"><button type="button" id="imp-${this.kind}-import" ${open ? "" : "disabled"}>Import</button>
        ${cost}<span id="imp-${this.kind}-confirm"></span></div>`;
  }

  show(preview, estimate) {
    this.own("preview").innerHTML = this.html(preview, estimate);
    this.own("import").addEventListener("click", () => this.start(estimate));
  }

  async start(estimate) {
    const msg = this.own("msg");
    msg.textContent = "";
    const queue = async () => {
      try {
        const got = await api(this.importPath, { method: "POST", body: this.body(true) });
        this.own("preview").innerHTML = this.kind === "items"
          ? `<p class="ok">Queued (job ${esc(short(got.job_id))})</p>`
          : `<p class="ok">Imported: ${esc(got.new)} new, ${esc(got.updated)} updated, ${esc(got.unchanged)} unchanged, ${esc(
            got.skipped)} skipped. Version ${esc(short(got.version))}.</p>`;
        this.uploadId = null;
        this.own("file").value = "";
        refreshSpend();
        await this.done?.();
      } catch (e) {
        msg.textContent = e.message;
      }
    };
    if (estimate && estimate.est_usd > 0) confirmSpend(this.own("confirm"), estimate, queue);
    else await queue();
  }
}

export function importBox(el, options) {
  return new ImportBox(el, options);
}
