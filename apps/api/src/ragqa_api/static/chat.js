import { $, UNPRICED, api, catalog, esc, head, methodLabel, modelOptions, plural, refreshSpend, shell, upload, usd } from "/static/ui.js";

const TURNS = 6; // earlier turns sent with the new message; the route trims to the same
const PASSAGE = 300; // characters of a source shown before it is expanded

shell("chat");
$("head").innerHTML = head({
  title: "Chat",
  sub: "Ask a store a question. Each reply cites the passages it used, listed under it.",
});

let turns = []; // what the log shows: {role: "user"} | {role: "reply", ...the /chat reply} | {role: "error"}
let answering = false;
let hasModel = false;
let adding = false; // files are being added: the store stays put until they are in
const cleared = {}; // New chat clicks per store, so a reply to a cleared conversation is dropped

// -- the conversation, kept per store in sessionStorage; the page works without it ------------

const key = (store) => `chat:${store}`;

function load(store = $("store").value) {
  try {
    const saved = JSON.parse(sessionStorage.getItem(key(store)));
    return Array.isArray(saved) ? saved : [];
  } catch {
    return [];
  }
}

function save(store = $("store").value, list = turns) {
  try {
    sessionStorage.setItem(key(store), JSON.stringify(list));
  } catch {
    // no storage: the chat lives until the page closes
  }
}

function forget() {
  try {
    sessionStorage.removeItem(key($("store").value));
  } catch {
    // nothing kept, nothing to clear
  }
}

// -- drawing ---------------------------------------------------------------------------------

// A citation mark, the route's rule: [n], [n, m, …] or [n-m] (hyphen or en dash), each
// number 1 to 4 ASCII digits.
const CITATION = /\[([0-9]{1,4}(?:\s*[-–]\s*[0-9]{1,4})?(?:\s*,\s*[0-9]{1,4}(?:\s*[-–]\s*[0-9]{1,4})?)*)\]/g;

// Escaped text with each cited number linked to its source: a lone [n] as a whole, each
// number of a grouped bracket on its own. A mark like [01] links to source 1.
function citeHtml(escaped, cited, at) {
  const have = new Set(cited.map((c) => c.n));
  const link = (n, shown) => (have.has(Number(n)) ? `<a class="cite" href="#src-${at}-${Number(n)}">${shown}</a>` : shown);
  return escaped.replace(CITATION, (mark, inner) =>
    /^\d+$/.test(inner) ? link(inner, mark) : `[${inner.replace(/\d+/g, (n) => link(n, n))}]`,
  );
}

// One escaped line: `code` spans kept as typed, bold and italic, then citation links.
// An underscore inside a word (snake_case_name) is no emphasis.
function inlineHtml(line, cited, at) {
  return line
    .split(/(`[^`]+`)/)
    .map((s, i) => {
      if (i % 2) return `<code>${s.slice(1, -1)}</code>`;
      const marked = s
        .replace(/\*\*(\S(?:.*?\S)?)\*\*/g, "<strong>$1</strong>")
        .replace(/(^|\W)__(\S(?:.*?\S)?)__(?!\w)/g, "$1<strong>$2</strong>")
        .replace(/\*(\S(?:[^*]*?\S)?)\*/g, "<em>$1</em>")
        .replace(/(^|\W)_(\S(?:[^_]*?\S)?)_(?!\w)/g, "$1<em>$2</em>");
      return citeHtml(marked, cited, at);
    })
    .join("");
}

// The reply as a safe subset of markdown. The text is escaped first, so no tag or attribute
// ever comes from the model: fenced code, `code`, bold, italic, - and 1. lists, # headings
// as bold lines, blank lines as paragraphs. Links, images, tables and HTML stay as text.
function textHtml(text, cited, at) {
  const lines = esc(text).replace(/\r\n?/g, "\n").split("\n");
  const out = [];
  let para = [];
  const flush = () => {
    if (para.length) out.push(`<p>${para.map((l) => inlineHtml(l, cited, at)).join("<br>")}</p>`);
    para = [];
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/^\s*```/.test(line)) {
      flush();
      const code = [];
      while (++i < lines.length && !/^\s*```\s*$/.test(lines[i])) code.push(lines[i]);
      out.push(`<pre><code>${code.join("\n")}</code></pre>`);
      continue;
    }
    const heading = line.match(/^#{1,3} +(.*)$/);
    if (heading) {
      flush();
      out.push(`<p><strong>${inlineHtml(heading[1], cited, at)}</strong></p>`);
      continue;
    }
    const kind = /^\s*[-*] +\S/.test(line) ? "ul" : /^\s*\d+\. +\S/.test(line) ? "ol" : null;
    if (kind) {
      flush();
      const items = [];
      const mark = kind === "ul" ? /^\s*[-*] +(.*)$/ : /^\s*\d+\. +(.*)$/;
      for (; i < lines.length && mark.test(lines[i]); i++) items.push(lines[i].match(mark)[1]);
      i--;
      out.push(`<${kind}>${items.map((s) => `<li>${inlineHtml(s, cited, at)}</li>`).join("")}</${kind}>`);
      continue;
    }
    if (line.trim()) para.push(line);
    else flush();
  }
  flush();
  return out.join("");
}

function sourceHtml(c, at) {
  const long = c.text.length > PASSAGE;
  const body = long
    ? `<details><summary>${esc(c.text.slice(0, PASSAGE))}…</summary><div class="full">${esc(c.text)}</div></details>`
    : `<div>${esc(c.text)}</div>`;
  const part = c.parts > 1 ? ` · part ${esc(c.part)} of ${esc(c.parts)}` : "";
  const from = (c.source ? ` · ${esc(c.source)}` : "") + part;
  return `<li id="src-${at}-${c.n}"><span class="what">[${c.n}] ${esc(c.id)}${from}</span>${body}</li>`;
}

function replyHtml(t, at) {
  const tags = t.query_tags?.length ? ` · question tags ${esc(t.query_tags.join(", "))}` : "";
  const sources = t.citations.length
    ? `<div class="what muted">Sources</div><ul class="sources">${t.citations.map((c) => sourceHtml(c, at)).join("")}</ul>`
    : "";
  return `<div class="msg reply"><div class="md" data-text="${esc(t.answer)}">${textHtml(t.answer, t.citations, at)}</div>${sources}
    <div class="meta muted">${esc(methodLabel(t.method))}${tags} · ${usd(t.usd)}</div></div>`;
}

function turnHtml(t, at) {
  if (t.role === "user") return `<div class="msg user">${esc(t.content)}</div>`;
  if (t.role === "note") return `<div class="msg note muted">${esc(t.content)}</div>`;
  if (t.role === "error") return `<div class="msg err">${esc(t.content)}${t.usd > 0 ? ` <span class="muted">(${usd(t.usd)})</span>` : ""}</div>`;
  return replyHtml(t, at);
}

function draw() {
  $("log").innerHTML = turns.map(turnHtml).join("");
  const last = $("log").lastElementChild;
  if (last) last.scrollIntoView({ block: "nearest" });
}

function setBusy(busy) {
  answering = busy;
  $("send").disabled = busy || !hasModel || !$("store").value;
  $("add-doc").disabled = adding || !$("store").value;
  $("status").textContent = busy ? "thinking…" : "";
}

// -- sending ---------------------------------------------------------------------------------

async function ask() {
  const content = $("box").value.trim();
  if (!content || answering || !hasModel || !$("store").value) return;
  const store = $("store").value;
  const mine = cleared[store] ?? 0;
  turns.push({ role: "user", content });
  // The conversation the model sees: the user's turns and its own answers, never an error.
  const messages = turns
    .filter((t) => t.role === "user" || t.role === "reply")
    .map((t) => (t.role === "user" ? { role: "user", content: t.content } : { role: "assistant", content: t.answer }))
    .slice(-(TURNS + 1));
  $("box").value = "";
  save();
  draw();
  setBusy(true);
  let turn;
  try {
    const r = await api("/chat", {
      method: "POST",
      body: { collection: store, messages, method: $("method").value, model: $("model").value || null },
    });
    turn = r.error || r.answer === null ? { role: "error", content: r.error ?? "no answer", usd: r.usd } : { role: "reply", ...r };
  } catch (e) {
    turn = { role: "error", content: e.message };
  } finally {
    setBusy(false);
    refreshSpend();
  }
  if ((cleared[store] ?? 0) !== mine) return; // New chat cleared its conversation meanwhile
  if (store !== $("store").value) {
    save(store, [...load(store), turn]); // the page is on another store: kept for when it's back
    return;
  }
  turns.push(turn);
  save();
  draw();
  $("spoken").textContent = turn.role === "reply" ? turn.answer : turn.content;
}

// -- adding documents -------------------------------------------------------------------------

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Asks, in #adding, whether to spend `estimate`; resolves true on Add and false on Cancel.
function askToAdd(file, estimate) {
  const cost = estimate.est_usd == null ? UNPRICED : `≈ ${usd(estimate.est_usd)}`;
  $("adding").innerHTML = `${esc(file)}: ${esc(cost)} · ${esc(plural(estimate.calls ?? 0, "AI call"))}
    <button type="button" data-add>Add</button> <button type="button" data-cancel class="secondary">Cancel</button>`;
  return new Promise((resolve) => {
    $("adding").querySelector("[data-add]").addEventListener("click", () => resolve(true));
    $("adding").querySelector("[data-cancel]").addEventListener("click", () => resolve(false));
  });
}

// Follows an import job, with the line "Adding <file>… 40%"; returns the finished job.
async function follow(file, jobId) {
  for (;;) {
    const job = await api(`/jobs/${encodeURIComponent(jobId)}`);
    if (job.status !== "queued" && job.status !== "running") return job;
    $("adding").textContent =
      job.status === "queued"
        ? `Adding ${file}: waiting for the store's other job…`
        : `Adding ${file}… ${Math.round((job.progress ?? 0) * 100)}%`;
    await wait(500);
  }
}

// One file into the store for good, as its own source: upload, estimate, confirm if it spends.
async function addFile(store, file) {
  const name = encodeURIComponent(store);
  $("adding").textContent = `Adding ${file.name}…`;
  const up = await upload(`/collections/${name}/uploads?kind=items&accept=text&filename=${encodeURIComponent(file.name)}`, file);
  const free = up.estimate?.est_usd === 0; // unpriced (null) is not free
  if (!free && !(await askToAdd(file.name, up.estimate ?? { est_usd: null }))) return null;
  $("adding").textContent = `Adding ${file.name}…`;
  const body = { upload_id: up.upload_id, kind: "items", mapping: up.preview.mapping, source_name: up.source_name, confirm: true };
  const queued = await api(`/collections/${name}/imports`, { method: "POST", body });
  const job = await follow(file.name, queued.job_id);
  if (job.status !== "done") throw new Error(job.error || `the import ${job.status}`);
  return `Added ${file.name}`;
}

async function addFiles(files) {
  const store = $("store").value;
  if (!files.length || !store || adding) return;
  adding = true;
  $("store").disabled = true;
  $("add-doc").disabled = true;
  let added = false;
  for (const file of files) {
    try {
      if (!/\.(txt|md)$/i.test(file.name)) throw new Error("only .txt and .md files can be added here");
      const said = await addFile(store, file);
      if (said) turns.push({ role: "note", content: said });
      added ||= Boolean(said);
    } catch (e) {
      turns.push({ role: "error", content: `${file.name}: ${e.message}` });
    }
    save();
    draw();
    refreshSpend();
  }
  // An add can give the store keyword lists or a vocabulary, and so more methods.
  if (added) {
    try {
      fillMethods(await api(`/collections/${encodeURIComponent(store)}`), $("method").value);
    } catch (e) {
      turns.push({ role: "error", content: e.message });
      save();
      draw();
    }
  }
  $("adding").textContent = "";
  $("store").disabled = false;
  adding = false;
  setBusy(answering);
}

// -- the pickers -----------------------------------------------------------------------------

let picking = 0; // bumped by each store pick, so only the newest one draws

// The store's methods in the picker, `keep` picked when the store has it, else the default.
function fillMethods(detail, keep) {
  const methods = detail.methods_available;
  const first = methods.includes(keep) ? keep : methods.includes("keyword_rag") ? "keyword_rag" : "hybrid";
  $("method").innerHTML = methods
    .map((m) => `<option value="${esc(m)}"${m === first ? " selected" : ""}>${esc(methodLabel(m))}</option>`)
    .join("");
}

async function pickStore() {
  const mine = ++picking;
  const name = $("store").value;
  turns = load();
  $("spoken").textContent = "";
  draw();
  try {
    const detail = await api(`/collections/${encodeURIComponent(name)}`);
    if (mine === picking) fillMethods(detail, null);
  } finally {
    setBusy(answering); // Send and Add follow the store even when its detail fails
  }
}

async function loadModels() {
  const ai = await api("/ai");
  $("model").innerHTML = modelOptions(ai.models, ai.roles.answer ?? ai.roles.default);
  hasModel = $("model").options.length > 0;
  $("note").innerHTML = hasModel ? "" : 'Chat needs an answer model: choose one on <a href="/settings">Settings</a>';
  setBusy(false);
}

async function loadStores() {
  const list = await api("/collections");
  $("store").innerHTML = list.map((c) => `<option>${esc(c.name)}</option>`).join("");
  if (list.length) await pickStore();
  else $("note").innerHTML += ' No stores yet. Create one on the <a href="/">Stores</a> page.';
  setBusy(false);
}

// pickStore sets Send and Add before it fails, so the error line set here stays.
$("store").addEventListener("change", () => pickStore().catch((e) => ($("status").textContent = e.message)));
$("new").addEventListener("click", () => {
  cleared[$("store").value] = (cleared[$("store").value] ?? 0) + 1;
  turns = [];
  $("spoken").textContent = "";
  forget();
  draw();
});
$("add-doc").addEventListener("click", () => $("doc-file").click());
$("doc-file").addEventListener("change", () => {
  const files = [...$("doc-file").files];
  $("doc-file").value = "";
  addFiles(files);
});
$("compose").addEventListener("submit", (event) => {
  event.preventDefault();
  ask();
});
$("box").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    ask();
  }
});

catalog()
  .catch(() => {})
  .then(loadModels)
  .then(loadStores)
  .catch((e) => ($("status").textContent = e.message));
