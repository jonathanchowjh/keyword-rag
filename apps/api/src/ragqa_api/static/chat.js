import { $, UNPRICED, api, esc, head, modelOptions, plural, refreshSpend, shell, upload, usd } from "/static/ui.js";

const PASSAGE = 300; // characters of a source shown before it is expanded
const MAX_EVIDENCE = 20; // the most passages a chat may answer from, the route's rule
const CHANNELS = ["meaning", "words", "keywords"]; // a search's weighted channels
const QUERY_TAGS = ["classifier", "similarity", "model"]; // oracle needs a question's gold: never in a chat

shell("chat");
$("head").innerHTML = head({
  title: "Chat",
  sub: "Ask across one or more stores. Each reply cites the passages it used, listed under it. Chats are saved.",
});

let stores = []; // every store, as GET /collections lists them
let chats = []; // the saved chats, newest first, as GET /chats lists them
let openId = ""; // the chat the page is on (the URL hash), set before it has loaded
let chat = null; // that chat once loaded: GET /chats/{id}, with its messages
let opening = 0; // bumped by each open, so only the newest one draws
let hasModel = false;
let adding = false; // files are being added
const details = {}; // store -> its GET /collections/{name}, for its chip
const pending = {}; // chat id -> the message still on its way there
const arrived = {}; // chat id -> a reply that came while its chat was loading, merged once it has
const local = {}; // chat id -> lines the server never kept: refusals and Add document's notes
const turns = {}; // "chat id:message n" -> what the reader has open or edited there, kept across redraws

// The chat's stores that still exist.
const live = () => (chat ? chat.stores.filter((s) => !chat.deleted_stores.includes(s)) : []);
const lastN = (c) => c.messages.at(-1)?.n ?? 0;

// Message n's state on the page: the attempt tab chosen (null: the latest), whether its
// Retrieval step is open, the passages expanded, each store's unsent edit and note, the store
// whose Set as default is being confirmed, and whether a request for it is on its way.
function turnOf(chatId, n) {
  return (turns[`${chatId}:${n}`] ??= {
    tab: null,
    open: false,
    expanded: new Set(),
    edits: {},
    notes: {},
    confirm: null,
    busy: false,
    error: "",
  });
}

const shownAttempt = (m, ui) => m.attempts.find((a) => a.attempt === ui.tab) ?? m.attempts.at(-1);

// The chat's messages with `message` in place of any of the same n: the server saves a message
// before its attempt, so a chat read meanwhile may already hold it with no attempt.
function withMessage(c, message) {
  const messages = [...c.messages.filter((m) => m.n !== message.n), message].sort((a, b) => a.n - b.n);
  return { ...c, messages };
}

// The chat with `attempt` in place of any of the same number on message n.
function withAttempt(c, n, attempt) {
  const messages = c.messages.map((m) =>
    m.n !== n
      ? m
      : { ...m, attempts: [...m.attempts.filter((a) => a.attempt !== attempt.attempt), attempt].sort((a, b) => a.attempt - b.attempt) },
  );
  return { ...c, messages };
}

// -- drawing a reply ---------------------------------------------------------------------------

// A citation mark, the route's rule: [n], [n, m, …] or [n-m] (hyphen or en dash), each
// number 1 to 4 ASCII digits.
const CITATION = /\[([0-9]{1,4}(?:\s*[-–]\s*[0-9]{1,4})?(?:\s*,\s*[0-9]{1,4}(?:\s*[-–]\s*[0-9]{1,4})?)*)\]/g;

// Escaped text with each cited number linked to its passage in the Retrieval step, whose id
// is ev-<at>-<n>: a lone [n] as a whole, each number of a grouped bracket on its own. A mark
// like [01] links to passage 1.
function citeHtml(escaped, cited, at) {
  const have = new Set(cited.map((c) => c.n));
  const link = (n, shown) => (have.has(Number(n)) ? `<a class="cite" href="#ev-${at}-${Number(n)}">${shown}</a>` : shown);
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
      // Items run on across a single blank line, and an indented line continues its item.
      const mark = kind === "ul" ? /^\s*[-*] +(.*)$/ : /^\s*(\d+)\. +(.*)$/;
      const items = [];
      const first = line.match(mark);
      const start = kind === "ol" && Number(first[1]) !== 1 ? ` start="${Number(first[1])}"` : "";
      for (; i < lines.length; i++) {
        const item = lines[i].match(mark);
        if (item) items.push([item[kind === "ul" ? 1 : 2]]);
        else if (/^\s+\S/.test(lines[i])) items[items.length - 1].push(lines[i].trim());
        else if (!lines[i].trim() && mark.test(lines[i + 1] ?? "")) continue;
        else break;
      }
      i--;
      const lis = items.map((parts) => `<li>${parts.map((s) => inlineHtml(s, cited, at)).join("<br>")}</li>`);
      out.push(`<${kind}${start}>${lis.join("")}</${kind}>`);
      continue;
    }
    if (line.trim()) para.push(line);
    else flush();
  }
  flush();
  return out.join("");
}

// A passage's text: the first PASSAGE characters, the rest behind a toggle whose open state
// is kept, under `key`, in the turn's `expanded`.
function passageHtml(text, key, ui) {
  if (text.length <= PASSAGE) return `<div>${esc(text)}</div>`;
  const open = ui.expanded.has(key) ? " open" : "";
  return `<details data-expand="${esc(key)}"${open}><summary>${esc(text.slice(0, PASSAGE))}…</summary><div class="full">${esc(text)}</div></details>`;
}

// One cited passage, listed under the answer.
function sourceHtml(c, at, ui) {
  const part = c.parts > 1 ? ` · part ${esc(c.part)} of ${esc(c.parts)}` : "";
  const from = (c.source ? ` · ${esc(c.source)}` : "") + part + (c.store ? ` · ${esc(c.store)}` : "");
  return `<li><span class="what">[${Number(c.n)}] ${esc(c.id)}${from}</span>${passageHtml(c.text, `src-${at}-${Number(c.n)}`, ui)}</li>`;
}

const costHtml = (x) => (x > 0 ? ` <span class="muted">(${usd(x)})</span>` : "");

// An attempt's answer with its sources and cost; with no answer, its error and cost (or that
// it is not answered yet) and Answer from this attempt. `at` is "<message n>-<attempt>".
function attemptHtml(a, at, ui) {
  if (a.answer == null) {
    const said = a.error
      ? `<div class="msg err">${esc(a.error)}${costHtml(a.usd)}</div>`
      : `<div class="msg unanswered muted">Not answered yet.${costHtml(a.usd)}</div>`;
    const error = ui.error ? `<span class="err" role="status">${esc(ui.error)}</span>` : "";
    const off = ui.busy || !hasModel ? " disabled" : "";
    return `${said}<div class="row"><button type="button" data-answer="${Number(a.attempt)}"${off}>Answer from this attempt</button>${error}</div>`;
  }
  const passages = new Set(a.evidence.map((e) => e.n));
  const cited = a.citations.filter((c) => passages.has(c.n)); // linked only where its passage is drawn
  const sources = a.citations.length
    ? `<div class="what muted">Sources</div><ul class="sources">${a.citations.map((c) => sourceHtml(c, at, ui)).join("")}</ul>`
    : "";
  const none = a.decisions.length && !a.decisions.some((d) => d.kept)
    ? '<div class="meta warn none-kept">No store was kept, so this answer used no store passages.</div>'
    : "";
  const note = a.error ? `<div class="meta warn">${esc(a.error)}</div>` : ""; // e.g. a failed rewrite
  return `<div class="msg reply"><div class="md" data-text="${esc(a.answer)}">${textHtml(a.answer, cited, at)}</div>${sources}
    ${none}${note}<div class="meta muted">Cost ${usd(a.usd)}</div></div>`;
}

// -- the retrieval step ------------------------------------------------------------------------

const two = (x) => (x == null ? "–" : Number(x).toFixed(2));

function methodText(d) {
  if (!d.method) return "–";
  return d.method === "keyword_rag" && d.search?.query_tags ? `keyword_rag · query tags: ${d.search.query_tags}` : d.method;
}

function barText(d) {
  if (d.bar_source === "none") return "none"; // no eval set
  if (d.bar_source === "few") return "too few questions";
  const said = { manual: " manual", override: " edited" }[d.bar_source] ?? "";
  return d.bar == null ? "–" : two(d.bar) + said;
}

function keptText(d) {
  if (d.reason === "deleted") return "deleted";
  if (d.error) return `dropped · error: ${d.error}`;
  return `${d.kept ? "kept" : "dropped"} · ${d.reason}`;
}

function decisionHtml(d) {
  return `<tr data-store="${esc(d.store)}"${d.kept ? "" : ' class="dropped"'}><th scope="row">${esc(d.store)}</th>
    <td class="method">${esc(methodText(d))}</td><td class="num best">${esc(two(d.best))}</td>
    <td class="num bar">${esc(barText(d))}</td><td class="kept">${esc(keptText(d))}</td></tr>`;
}

// One passage the answer was given. Its id, ev-<message n>-<attempt>-<passage n>, is where
// the answer's [n] links, so the attempts of one message never share an id.
function evidenceHtml(e, at, ui) {
  const key = `ev-${at}-${Number(e.n)}`;
  const what = [`[${Number(e.n)}] ${e.id}`, e.store, e.source, e.parts > 1 ? `part ${e.part} of ${e.parts}` : null]
    .concat(e.rank != null ? [`rank ${e.rank}`] : [], e.channels?.length ? [e.channels.join(", ")] : [])
    .filter((x) => x != null && x !== "")
    .join(" · ");
  return `<li id="${key}" data-store="${esc(e.store)}"><span class="what">${esc(what)}</span>${passageHtml(e.text, key, ui)}</li>`;
}

// A store's Edit form as its decision ran: input values, as the form holds them.
function editBase(d) {
  const w = d.search?.weights ?? {};
  const out = {
    method: d.method ?? "",
    k: d.k == null ? "" : String(d.k),
    query_tags: d.search?.query_tags ?? "similarity",
    ownBar: d.bar_source !== "override",
    bar: d.bar_source === "override" && d.bar != null ? String(d.bar) : "",
    keep: d.reason === "search anyway",
  };
  for (const c of CHANNELS) out[c] = String(w[c] ?? 1);
  return out;
}

// The override for what `edit` changes from `base`, only the fields changed; null for k or
// bar means the store's own. Throws, with what to fix, on a value the route would refuse.
function overrideOf(edit, base) {
  const out = {};
  if (edit.method !== base.method) out.method = edit.method;
  if (edit.k !== base.k) {
    const k = edit.k.trim() === "" ? null : Number(edit.k);
    if (k !== null && !(Number.isInteger(k) && k >= 1 && k <= MAX_EVIDENCE)) {
      throw new Error(`k is a whole number from 1 to ${MAX_EVIDENCE}.`);
    }
    out.k = k;
  }
  const search = {};
  if (edit.query_tags !== base.query_tags) search.query_tags = edit.query_tags;
  for (const c of CHANNELS.filter((c) => edit[c] !== base[c])) {
    const weight = edit[c].trim() === "" ? NaN : Number(edit[c]);
    if (!Number.isFinite(weight)) throw new Error("A weight is a number, 0 or more.");
    (search.weights ??= {})[c] = weight;
  }
  if (Object.keys(search).length) out.search = search;
  if (edit.ownBar !== base.ownBar || (!edit.ownBar && edit.bar !== base.bar)) {
    const bar = edit.bar.trim() === "" ? NaN : Number(edit.bar);
    if (!edit.ownBar && !(bar >= -1 && bar <= 1)) throw new Error("A bar is a number from -1 to 1.");
    out.bar = edit.ownBar ? null : bar;
  }
  if (edit.keep !== base.keep) out.keep = edit.keep;
  return out;
}

function readEdit(box) {
  const field = (name) => box.querySelector(`[name="${name}"]`);
  const out = { ownBar: field("ownBar").checked, keep: field("keep").checked };
  for (const name of ["method", "k", "query_tags", "bar", ...CHANNELS]) out[name] = field(name).value;
  return out;
}

const option = (value, chosen) => `<option value="${esc(value)}"${value === chosen ? " selected" : ""}>${esc(value)}</option>`;

// A store's Edit form: its unsent edit when there is one, else the decision as it ran.
function editHtml(d, ui) {
  const store = d.store;
  const base = editBase(d);
  const e = ui.edits[store] ?? base;
  const methods = [...new Set([...(details[store]?.methods_available ?? []), base.method, e.method])].filter(Boolean);
  const weights = CHANNELS.map(
    (c) => `<label>${c[0].toUpperCase() + c.slice(1)} weight <input name="${c}" type="number" min="0" step="0.1" value="${esc(e[c])}"></label>`,
  ).join("");
  const note = ui.notes[store];
  const off = ui.busy ? " disabled" : "";
  const confirm =
    ui.confirm === store
      ? `<div class="confirm"><span>${esc(`Save these settings as ${store}'s default? Search anyway is not saved.`)}</span>
        <button type="button" data-save${off}>Yes, save</button> <button type="button" class="secondary" data-unsave>Cancel</button></div>`
      : "";
  return `<fieldset data-edit="${esc(store)}"><legend>${esc(store)}</legend><div class="row">
      <label>Method <select name="method">${methods.map((m) => option(m, e.method)).join("")}</select></label>
      <label>k <input name="k" type="number" min="1" max="${MAX_EVIDENCE}" step="1" value="${esc(e.k)}"></label>
      <label>Query tags <select name="query_tags">${QUERY_TAGS.map((t) => option(t, e.query_tags)).join("")}</select></label>
      ${weights}
      <label>Bar <input name="bar" type="number" min="-1" max="1" step="0.01" value="${esc(e.bar)}" placeholder="${esc(two(d.bar))}"${e.ownBar ? " disabled" : ""}></label>
      <label class="check"><input name="ownBar" type="checkbox"${e.ownBar ? " checked" : ""}> Use the store's bar</label>
      <label class="check"><input name="keep" type="checkbox"${e.keep ? " checked" : ""}> Search anyway</label>
    </div><div class="row">
      <button type="button" data-rerun${off}>Re-run retrieval</button>
      <button type="button" class="secondary" data-default${off}>Set as default</button>
      <span class="edit-note${note?.bad ? " err" : " muted"}" role="status">${esc(note?.text ?? "")}</span>
    </div>${confirm}</fieldset>`;
}

// The attempt's Retrieval step, collapsed unless the reader opened it: the search text, each
// store's decision, the passages, and, on the latest attempt, an Edit form for each of the
// chat's stores still there. Only the latest: the chat's overrides build on what it ran with.
function retrievalHtml(a, at, ui, latest) {
  const passages = [...a.evidence].sort((x, y) => x.n - y.n).map((e) => evidenceHtml(e, at, ui)).join("");
  const stores = a.decisions.length
    ? `<div class="scroll"><table class="decisions"><thead><tr><th>Store</th><th>Method</th><th class="num">Best match</th>
        <th class="num">Bar</th><th>Kept</th></tr></thead><tbody>${a.decisions.map(decisionHtml).join("")}</tbody></table></div>`
    : '<p class="muted">No store was searched.</p>';
  const edits = latest ? a.decisions.filter((d) => live().includes(d.store)).map((d) => editHtml(d, ui)).join("") : "";
  const elsewhere = latest ? "" : '<p class="muted">To edit and re-run, open the latest attempt.</p>';
  return `<details class="retrieval"${ui.open ? " open" : ""}><summary>Retrieval step</summary>
    <p><strong>Searched for:</strong> <span class="searched">${esc(a.search_text)}</span></p>
    <h3>Stores</h3>${stores}
    <h3>Evidence</h3>${passages ? `<ol class="evidence">${passages}</ol>` : '<p class="muted">No passages.</p>'}
    ${edits ? `<h3>Edit</h3><div class="edits">${edits}</div>` : elsewhere}</details>`;
}

// A saved message: the user's text, then the attempt chosen (the latest unless a tab says
// otherwise), with a tab for each attempt when there are several, and its Retrieval step.
function messageHtml(m) {
  const n = Number(m.n);
  const ui = turnOf(chat.id, n);
  const shown = shownAttempt(m, ui);
  let body = "";
  if (shown) {
    const at = `${n}-${Number(shown.attempt)}`;
    const tabs =
      m.attempts.length > 1
        ? `<div class="tabs" role="tablist">${m.attempts
            .map((a) => {
              const on = a.attempt === shown.attempt;
              return `<button type="button" role="tab" class="secondary" aria-selected="${on}" data-tab="${Number(a.attempt)}">Attempt ${Number(a.attempt)}</button>`;
            })
            .join("")}</div>`
        : "";
    body = tabs + attemptHtml(shown, at, ui) + retrievalHtml(shown, at, ui, shown === m.attempts.at(-1));
  }
  return `<div class="turn" id="msg-${n}" data-n="${n}"><div class="msg user">${esc(m.content)}</div>${body}</div>`;
}

function lineHtml(l) {
  if (l.role === "user") return `<div class="msg user">${esc(l.content)}</div>`;
  if (l.role === "note") return `<div class="msg note muted">${esc(l.content)}</div>`;
  return `<div class="msg err">${esc(l.content)}</div>`;
}

// What the newest reply says aloud.
function spokenOf(m) {
  const a = m.attempts.at(-1);
  return a?.answer ?? `Error: ${a?.error ?? "no answer"}`;
}

// -- drawing the page ------------------------------------------------------------------------

function drawList() {
  $("chat-list").innerHTML = chats
    .map(
      (c) => `<li class="${c.id === openId ? "active" : ""}"><a href="#${esc(c.id)}">${esc(c.title)}
        <span class="stores">${esc(c.stores.join(", "))}</span></a>
        <button type="button" class="secondary" data-delete="${esc(c.id)}" aria-label="Delete chat ${esc(c.title)}">Delete</button></li>`,
    )
    .join("");
}

function chipHtml(store, only) {
  const gone = chat.deleted_stores.includes(store);
  const what = gone ? " · deleted" : details[store] ? ` · ${plural(details[store].count, "item")}` : "";
  return `<span class="chip${gone ? " gone" : ""}" data-store="${esc(store)}">${esc(store)}<span class="muted">${esc(what)}</span>
    <button type="button" data-remove="${esc(store)}" aria-label="Remove ${esc(store)}"${only ? " disabled" : ""}>×</button></span>`;
}

function drawChips() {
  if (chat) $("chips").innerHTML = chat.stores.map((s) => chipHtml(s, chat.stores.length === 1)).join("");
}

function drawHead() {
  $("empty").hidden = Boolean(chat);
  $("chat-head").hidden = !chat;
  if (!chat) return;
  $("chat-title").textContent = chat.title;
  drawChips();
  const others = stores.filter((s) => !chat.stores.includes(s.name));
  $("add-store").innerHTML = ['<option value="">Add a store…</option>']
    .concat(others.map((s) => `<option value="${esc(s.name)}">${esc(s.name)} · ${esc(s.qa_type)}</option>`))
    .join("");
  $("add-store").disabled = !others.length;
  $("evidence").value = chat.settings.evidence_size;
  $("rewrite").checked = Boolean(chat.settings.rewrite);
  const keep = $("doc-store").value;
  $("doc-store").innerHTML = live()
    .map((s) => `<option value="${esc(s)}"${s === keep ? " selected" : ""}>${esc(s)}</option>`)
    .join("");
}

function drawLog() {
  let html = "";
  if (chat) {
    const lines = local[chat.id] ?? [];
    const after = (n) => lines.filter((l) => l.after === n).map(lineHtml).join("");
    const waiting = pending[chat.id];
    // While a reply is on its way, the message it answers is drawn once, as the waiting line.
    const shown = waiting == null ? chat.messages : chat.messages.filter((m) => m.attempts.length);
    html = after(0) + shown.map((m) => messageHtml(m) + after(m.n)).join("");
    if (waiting != null) html += `<div class="msg user">${esc(waiting)}</div>`;
  }
  $("log").innerHTML = html;
  const last = $("log").lastElementChild;
  if (last) last.scrollIntoView({ block: "nearest" });
}

// Draws message n again, in place, from the chat and its turn's state.
function drawTurn(n) {
  const m = chat?.messages.find((x) => x.n === n);
  const el = document.getElementById(`msg-${n}`);
  if (m && el) el.outerHTML = messageHtml(m);
}

function setBusy() {
  const waiting = chat != null && pending[chat.id] != null;
  $("send").disabled = !chat || waiting || !hasModel;
  $("add-doc").disabled = !chat || adding || !live().length;
  $("doc-store").disabled = adding;
  $("status").textContent = waiting ? "thinking…" : "";
}

// -- loading -----------------------------------------------------------------------------------

async function loadList() {
  chats = await api("/chats");
  drawList();
  const listed = chats.find((c) => c.id === chat?.id);
  if (listed) {
    chat.title = listed.title; // the first message names a new chat
    $("chat-title").textContent = chat.title;
  }
}

// The stores' details for their chips; a failure is said in the status line and changes nothing else.
async function loadDetails(names) {
  const got = await Promise.allSettled(
    names.map(async (s) => (details[s] = await api(`/collections/${encodeURIComponent(s)}`))),
  );
  drawChips();
  const failed = got.find((r) => r.status === "rejected");
  if (failed) $("status").textContent = failed.reason.message;
}

async function openChat(id) {
  const mine = ++opening;
  openId = id;
  chat = null;
  $("spoken").textContent = ""; // another chat: nothing of the last one is read out
  drawList();
  drawHead();
  drawLog();
  setBusy();
  if (!id) return;
  try {
    const got = await api(`/chats/${encodeURIComponent(id)}`);
    if (mine !== opening) return;
    chat = arrived[id] ? withMessage(got, arrived[id]) : got;
    delete arrived[id];
  } catch (e) {
    if (mine === opening) $("status").textContent = e.message;
    return;
  }
  drawHead();
  drawLog();
  setBusy();
  await loadDetails(live());
}

const hashId = () => location.hash.slice(1); // a chat id is hex: nothing to decode

// -- saved chats -------------------------------------------------------------------------------

function pickStores() {
  if (!stores.length) {
    $("picking").innerHTML = '<p class="muted">No stores yet. Create one on the <a href="/">Stores</a> page.</p>';
    return;
  }
  const boxes = stores
    .map((s) => `<label class="check"><input type="checkbox" value="${esc(s.name)}"> ${esc(s.name)} <span class="muted">${esc(s.qa_type)}</span></label>`)
    .join("");
  $("picking").innerHTML = `<fieldset><legend>Stores to search</legend><div class="checks">${boxes}</div></fieldset>
    <div class="actions"><button type="button" data-start>Start chat</button>
      <button type="button" class="secondary" data-cancel>Cancel</button></div>
    <p id="pick-note" class="err" role="status"></p>`;
  $("picking").querySelector("[data-cancel]").addEventListener("click", () => ($("picking").innerHTML = ""));
  $("picking").querySelector("[data-start]").addEventListener("click", async () => {
    const picked = [...$("picking").querySelectorAll("input:checked")].map((box) => box.value);
    if (!picked.length) {
      $("pick-note").textContent = "Pick at least one store.";
      return;
    }
    try {
      const made = await api("/chats", { method: "POST", body: { stores: picked } });
      $("picking").innerHTML = "";
      await loadList();
      location.hash = made.id; // opens it
    } catch (e) {
      $("pick-note").textContent = e.message;
    }
  });
}

function askDelete(id) {
  const title = chats.find((c) => c.id === id)?.title ?? "";
  const box = $("deleting");
  box.innerHTML = `<p>${esc(`Delete chat ${title}?`)}</p><div class="actions">
    <button type="button" class="danger" data-yes>Yes, delete</button>
    <button type="button" class="secondary" data-keep>Keep it</button></div>`;
  box.querySelector("[data-keep]").addEventListener("click", () => (box.innerHTML = ""));
  box.querySelector("[data-yes]").addEventListener("click", async () => {
    try {
      await api(`/chats/${encodeURIComponent(id)}`, { method: "DELETE" });
      box.innerHTML = "";
      delete local[id];
      if (id === openId) {
        history.replaceState(null, "", location.pathname); // no hash: no chat open
        await openChat("");
      }
      await loadList();
    } catch (e) {
      box.innerHTML = `<p class="err" role="status">${esc(e.message)}</p>`;
    }
  });
}

// Saves a change to the open chat's title, stores, evidence size or rewrite.
async function patch(body) {
  if (!chat) return;
  const id = chat.id;
  try {
    const saved = await api(`/chats/${encodeURIComponent(id)}`, { method: "PATCH", body });
    if (chat?.id === id) {
      // Only stores that exist can be added, so the deleted ones are those still kept.
      const deleted = chat.deleted_stores.filter((s) => saved.stores.includes(s));
      chat = { ...chat, ...saved, deleted_stores: deleted };
      drawHead();
      setBusy();
      loadDetails(live().filter((s) => !(s in details)));
    }
  } catch (e) {
    if (chat?.id === id) drawHead(); // back to what is saved
    $("status").textContent = e.message;
  }
  loadList().catch(() => {});
}

// -- sending -----------------------------------------------------------------------------------

async function ask() {
  const content = $("box").value.trim();
  if (!content || !chat || pending[chat.id] != null || !hasModel) return;
  const id = chat.id;
  const after = lastN(chat);
  pending[id] = content;
  $("box").value = "";
  drawLog();
  setBusy();
  let message = null;
  let error = null;
  try {
    message = await api(`/chats/${encodeURIComponent(id)}/messages`, {
      method: "POST",
      body: { content, model: $("model").value || null },
    });
  } catch (e) {
    error = e.message; // refused: the server kept nothing, so the page keeps it until a reload
    (local[id] ??= []).push({ after, role: "user", content }, { after, role: "error", content: error });
  }
  delete pending[id];
  refreshSpend();
  loadList().catch(() => {}); // the first message names the chat, and it is now the newest
  if (openId !== id) {
    setBusy(); // its own chat reads it from the server when opened
    return;
  }
  if (chat?.id === id) {
    if (message) chat = withMessage(chat, message);
    drawLog();
  } else if (message) {
    arrived[id] = message; // the chat is loading: merged when it has, as its read may be older
  }
  setBusy();
  $("spoken").textContent = message ? spokenOf(message) : `Error: ${error}`;
}

// -- re-runs and defaults ----------------------------------------------------------------------

// Re-runs message n's retrieval with the store's changes as the chat's override, and shows
// the new attempt. A refusal is said beside the store and the edit stays. Returns the attempt,
// or null when there is none.
async function rerun(n, store) {
  const id = chat.id;
  const ui = turnOf(id, n);
  const m = chat.messages.find((x) => x.n === n);
  const decision = shownAttempt(m, ui)?.decisions.find((d) => d.store === store);
  if (!decision || ui.busy) return null;
  let override;
  try {
    override = overrideOf(ui.edits[store] ?? editBase(decision), editBase(decision));
  } catch (e) {
    ui.notes[store] = { text: e.message, bad: true };
    drawTurn(n);
    return null;
  }
  ui.busy = true;
  ui.notes[store] = { text: "Re-running…" };
  drawTurn(n);
  let attempt = null;
  try {
    const overrides = Object.keys(override).length ? { [store]: override } : {};
    attempt = await api(`/chats/${encodeURIComponent(id)}/messages/${Number(n)}/attempts`, { method: "POST", body: { overrides } });
    delete ui.edits[store]; // the new attempt ran with them
    ui.notes[store] = null;
    ui.tab = attempt.attempt;
    if (chat?.id === id) chat = withAttempt(chat, n, attempt);
  } catch (e) {
    ui.notes[store] = { text: e.message, bad: true };
  }
  ui.busy = false;
  refreshSpend();
  if (chat?.id === id) drawTurn(n);
  return attempt;
}

// Answers the attempt from its own passages.
async function answerAttempt(n, number) {
  const id = chat.id;
  const ui = turnOf(id, n);
  if (ui.busy) return;
  ui.busy = true;
  ui.error = "";
  drawTurn(n);
  try {
    const attempt = await api(`/chats/${encodeURIComponent(id)}/messages/${Number(n)}/attempts/${Number(number)}/answer`, {
      method: "POST",
      body: { model: $("model").value || null },
    });
    if (chat?.id === id) {
      chat = withAttempt(chat, n, attempt);
      $("spoken").textContent = attempt.answer ?? `Error: ${attempt.error ?? "no answer"}`;
    }
  } catch (e) {
    ui.error = e.message;
  }
  ui.busy = false;
  refreshSpend();
  if (chat?.id === id) drawTurn(n);
}

const unset = (x, shown = x) => (x == null ? "unset" : shown);

// Saves the store's settings as its default. An edit not yet re-run is re-run first, as the
// server saves the override the chat holds.
async function setDefault(n, store) {
  const id = chat.id;
  const ui = turnOf(id, n);
  ui.confirm = null;
  const m = chat.messages.find((x) => x.n === n);
  const decision = shownAttempt(m, ui)?.decisions.find((d) => d.store === store);
  let edited = false;
  try {
    edited = Boolean(decision) && Object.keys(overrideOf(ui.edits[store] ?? editBase(decision), editBase(decision))).length > 0;
  } catch (e) {
    ui.notes[store] = { text: e.message, bad: true };
    drawTurn(n);
    return;
  }
  if (edited && !(await rerun(n, store))) return;
  ui.busy = true;
  if (chat?.id === id) drawTurn(n);
  try {
    const saved = await api(`/chats/${encodeURIComponent(id)}/stores/${encodeURIComponent(store)}/default`, { method: "POST" });
    const what = [`method ${unset(saved.chat_method)}`, `k ${unset(saved.chat_k)}`, `bar ${unset(saved.chat_bar, two(saved.chat_bar))}`];
    ui.notes[store] = { text: `Saved as ${store}'s default: ${what.join(", ")}.` };
  } catch (e) {
    ui.notes[store] = { text: e.message, bad: true };
  }
  ui.busy = false;
  if (chat?.id === id) drawTurn(n);
}

// -- adding documents --------------------------------------------------------------------------

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
  return `Added ${file.name} to ${store}`;
}

async function addFiles(files) {
  const store = $("doc-store").value;
  if (!files.length || !store || !chat || adding) return;
  const id = chat.id;
  const start = lastN(chat);
  // A line in the chat the files went to, drawn when the page is still on it.
  const say = (role, content) => {
    (local[id] ??= []).push({ after: chat?.id === id ? lastN(chat) : start, role, content });
    if (chat?.id === id) drawLog();
  };
  adding = true;
  setBusy();
  let added = false;
  for (const file of files) {
    try {
      if (!/\.(txt|md)$/i.test(file.name)) throw new Error("only .txt and .md files can be added here");
      const said = await addFile(store, file);
      if (said) say("note", said);
      added ||= Boolean(said);
    } catch (e) {
      say("error", `${file.name}: ${e.message}`);
    }
    refreshSpend();
  }
  $("adding").textContent = "";
  adding = false;
  setBusy();
  if (added) await loadDetails([store]); // its chip shows the items it now holds
}

// -- start -------------------------------------------------------------------------------------

async function loadModels() {
  const ai = await api("/ai");
  $("model").innerHTML = modelOptions(ai.models, ai.roles.answer ?? ai.roles.default);
  hasModel = $("model").options.length > 0;
  $("note").innerHTML = hasModel ? "" : 'Chat needs an answer model: choose one on <a href="/settings">Settings</a>';
  setBusy();
}

const said = (e) => ($("status").textContent = e.message);

$("new").addEventListener("click", pickStores);
$("chat-list").addEventListener("click", (event) => {
  const button = event.target.closest("[data-delete]");
  if (button) askDelete(button.dataset.delete);
});
$("chips").addEventListener("click", (event) => {
  const button = event.target.closest("[data-remove]");
  if (button && chat) patch({ stores: chat.stores.filter((s) => s !== button.dataset.remove) });
});
$("add-store").addEventListener("change", () => {
  const store = $("add-store").value;
  if (store && chat) patch({ stores: [...chat.stores, store] });
});
$("evidence").addEventListener("change", () => {
  const size = Number($("evidence").value);
  if (Number.isInteger(size) && size >= 1 && size <= MAX_EVIDENCE) patch({ evidence_size: size });
  else {
    $("status").textContent = `Evidence size is a whole number from 1 to ${MAX_EVIDENCE}.`;
    if (chat) $("evidence").value = chat.settings.evidence_size;
  }
});
$("rewrite").addEventListener("change", () => patch({ rewrite: $("rewrite").checked }));
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
$("log").addEventListener("click", (event) => {
  const turn = event.target.closest(".turn");
  if (!turn || !chat) return;
  const n = Number(turn.dataset.n);
  const ui = turnOf(chat.id, n);
  const at = (selector) => event.target.closest(selector);
  if (at("a.cite")) {
    // Its passage is in the Retrieval step: open it before the jump.
    ui.open = true;
    turn.querySelector("details.retrieval").open = true;
    return;
  }
  if (at("[data-tab]")) {
    ui.tab = Number(at("[data-tab]").dataset.tab);
    drawTurn(n);
    return;
  }
  if (at("[data-answer]")) {
    answerAttempt(n, Number(at("[data-answer]").dataset.answer)).catch(said);
    return;
  }
  const box = at("fieldset[data-edit]");
  if (!box) return;
  const store = box.dataset.edit;
  if (at("[data-rerun]")) rerun(n, store).catch(said);
  else if (at("[data-save]")) setDefault(n, store).catch(said);
  else if (at("[data-default]") || at("[data-unsave]")) {
    ui.confirm = at("[data-default]") ? store : null;
    drawTurn(n);
  }
});
// An edit is kept as typed, so a redraw keeps it; nothing is sent until Re-run or Set as default.
const edited = (event) => {
  const box = event.target.closest("fieldset[data-edit]");
  const turn = event.target.closest(".turn");
  if (!box || !turn || !chat) return;
  const edit = readEdit(box);
  turnOf(chat.id, Number(turn.dataset.n)).edits[box.dataset.edit] = edit;
  box.querySelector('[name="bar"]').disabled = edit.ownBar;
};
$("log").addEventListener("input", edited);
$("log").addEventListener("change", edited);
// What is open is kept per turn, so a redraw keeps it. Toggle does not bubble: caught on its way down.
$("log").addEventListener(
  "toggle",
  (event) => {
    const el = event.target;
    const turn = el.closest?.(".turn");
    if (!turn || !chat) return;
    const ui = turnOf(chat.id, Number(turn.dataset.n));
    if (el.matches("details.retrieval")) ui.open = el.open;
    else if (el.dataset.expand) el.open ? ui.expanded.add(el.dataset.expand) : ui.expanded.delete(el.dataset.expand);
  },
  true,
);
window.addEventListener("hashchange", () => {
  const id = hashId();
  if (id && document.getElementById(id)) {
    // A citation's jump to its passage, not another chat: the address goes on naming the chat.
    history.replaceState(null, "", openId ? `#${openId}` : location.pathname);
    return;
  }
  openChat(id).catch(said);
});

loadModels()
  .catch(said)
  .then(async () => (stores = await api("/collections")))
  .then(loadList)
  .then(() => openChat(hashId()))
  .catch(said);
