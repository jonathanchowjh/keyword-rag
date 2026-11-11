import { $, UNPRICED, api, catalog, esc, followJob, head, manyBody, methodLabel, modelOptions, plural, refreshSpend, say, shell, upload, usd } from "/static/ui.js";
import { PASSAGES_ONLY, attemptHtml, passagesOnly, retrievalHtml } from "/static/reply.js";

const REWRITE_AGAIN = "Re-run rewrites this follow-up with AI: one AI call.";
const SOURCES = { settings: "Settings", store: "store", recommended: "recommended" }; // where a value is from

shell("chat");
$("head").innerHTML = head({
  title: "Chat",
  sub: "Ask across one or more stores. Each reply cites the passages it used, listed under it. Chats are saved.",
});

let stores = []; // every store, as GET /collections lists them
let chats = []; // the saved chats, newest first, as GET /chats lists them
let openId = ""; // the chat the page is on (the URL hash), set before it has loaded
let chat = null; // that chat once loaded: GET /chats/{id}, with its messages and what is in force
let opening = 0; // bumped by each open, so only the newest one draws
let aiFailed = false; // the last read of /ai failed
let reading = false; // a read of /ai is on its way
let hasModel = false; // an answer model can be picked: replies are written, not passages only
let defaults = null; // GET /search-defaults: what a chat field the chat holds would inherit
let cat = null; // GET /catalog: the chat methods and their labels
let adding = false; // files are being added
const details = {}; // store -> its GET /collections/{name}: its chip, its methods, its own settings
const pending = {}; // chat id -> the message still on its way there
const arrived = {}; // chat id -> the replies (a message, a re-run, an answer) that came while its chat was loading, merged once it has
const local = {}; // chat id -> lines the server never kept: refusals and Add file's notes; read and pushed through notesOf
const turns = {}; // "chat id:message n" (and "chat id:panel") -> what the reader has open or edited there, kept across redraws

// The chat's stores that still exist.
const live = () => (chat ? chat.stores.filter((s) => !chat.deleted_stores.includes(s)) : []);
const lastN = (c) => c.messages.at(-1)?.n ?? 0;
// The newest message with an attempt: the settings panel sits under it.
const latestN = () => chat?.messages.filter((m) => m.attempts.length).at(-1)?.n ?? null;

// -- the page's own lines, kept in this browser ---------------------------------------------------

const NOTES = "chat-notes:"; // localStorage key prefix, then the chat id: its lines as a JSON array

// The chat's own lines, read from this browser's storage on first touch. A storage that throws
// (a private window, blocked site data) keeps them in memory only, until a reload.
function notesOf(id) {
  if (!local[id]) {
    let kept = null;
    try {
      kept = JSON.parse(localStorage.getItem(NOTES + id));
    } catch {}
    local[id] = Array.isArray(kept) ? kept.filter((l) => l && typeof l === "object") : [];
  }
  return local[id];
}

// Adds lines to the chat's own, and writes them back.
function addNotes(id, ...lines) {
  notesOf(id).push(...lines);
  try {
    localStorage.setItem(NOTES + id, JSON.stringify(local[id]));
  } catch {}
}

function dropNotes(id) {
  delete local[id];
  try {
    localStorage.removeItem(NOTES + id);
  } catch {}
}

// Removes the kept lines of chats no longer listed, deleted from another tab or page. The open
// chat's are kept: a list read before this tab made it may land after its first note.
function pruneNotes(listed) {
  const ids = new Set([...listed.map((c) => c.id), chat?.id, openId]);
  try {
    const keys = Array.from({ length: localStorage.length }, (_, i) => localStorage.key(i));
    for (const key of keys) {
      if (key?.startsWith(NOTES) && !ids.has(key.slice(NOTES.length))) localStorage.removeItem(key);
    }
  } catch {}
}

// Message n's state on the page: the attempt tab chosen (null: the latest), whether its
// Retrieval step is open, the passages expanded, and Answer from this attempt's progress.
function turnOf(chatId, n) {
  return (turns[`${chatId}:${n}`] ??= { tab: null, open: false, expanded: new Set(), busy: false, error: "" });
}

// The settings panel's state for a chat: open or not, the unsent edit (every field as the
// panel last read it, or null), the panel's note and each store's, the confirm on show (a
// re-run's or a store's Set as default), and whether a request is on its way.
function panelOf(chatId) {
  return (turns[`${chatId}:panel`] ??= { open: false, edit: null, note: null, notes: {}, confirm: null, busy: false });
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

// -- the settings panel ------------------------------------------------------------------------

const asText = (x) => (x == null ? "" : String(x));
const onOff = (x) => (x == null ? "" : x ? "on" : "off");

// The chat's own values as the panel's fields hold them; "" inherits.
function savedFields() {
  const s = chat.settings ?? {};
  const out = { evidence_size: asText(s.evidence_size), rerank: onOff(s.rerank), rewrite: onOff(s.rewrite), stores: {} };
  for (const store of live()) {
    const own = s.stores?.[store] ?? {};
    out.stores[store] = { method: own.method ?? "", candidates: asText(own.candidates) };
  }
  return out;
}

// What the panel shows: the reader's unsent edit over the chat's own values.
function shownFields(ui) {
  const saved = savedFields();
  if (!ui.edit) return saved;
  const out = { ...saved, ...ui.edit, stores: {} };
  for (const store of live()) out.stores[store] = { ...saved.stores[store], ...ui.edit.stores?.[store] };
  return out;
}

// The edit as kept: only the fields of `fields` (every field, as the panel shows them) that
// differ from the chat's own values; null when none does.
function editOf(fields) {
  const saved = savedFields();
  const out = { stores: {} };
  for (const f of ["evidence_size", "rerank", "rewrite"]) if (fields[f] !== saved[f]) out[f] = fields[f];
  for (const [store, now] of Object.entries(fields.stores)) {
    const was = saved.stores[store];
    if (!was) continue; // off the chat since
    const changed = Object.fromEntries(["method", "candidates"].filter((f) => now[f] !== was[f]).map((f) => [f, now[f]]));
    if (Object.keys(changed).length) out.stores[store] = changed;
  }
  return Object.keys(out).length > 1 || Object.keys(out.stores).length ? out : null;
}

function readPanel(el) {
  const value = (root, name) => root.querySelector(`[name="${name}"]`)?.value ?? "";
  const out = { evidence_size: value(el, "evidence_size"), rerank: value(el, "rerank"), rewrite: value(el, "rewrite"), stores: {} };
  for (const box of el.querySelectorAll("fieldset[data-edit]")) {
    out.stores[box.dataset.edit] = { method: value(box, "method"), candidates: value(box, "candidates") };
  }
  return out;
}

// The max attribute of a number field, or nothing while the server's limit is not known.
const maxAttr = (max) => (max == null ? "" : `max="${max}"`);

// A number field as the route takes it: blank is null (inherit). Throws, with what to fix, on
// a value the route would refuse. max is the server's limit, null until the defaults load.
function whole(value, max, name) {
  if (value.trim() === "") return null;
  const x = Number(value);
  if (!(Number.isInteger(x) && x >= 1 && (max == null || x <= max))) {
    throw new Error(max == null ? `${name} is a whole number from 1.` : `${name} is a whole number from 1 to ${max}.`);
  }
  return x;
}

// The re-run's body: every field the panel changes from the chat's own values, null for one
// set back to inherit.
function changesOf(ui) {
  const saved = savedFields();
  const shown = shownFields(ui);
  const settings = {};
  const changed = {};
  if (shown.evidence_size !== saved.evidence_size) settings.evidence_size = whole(shown.evidence_size, defaults?.limits?.evidence_size?.[1] ?? null, "Evidence size");
  for (const f of ["rerank", "rewrite"]) if (shown[f] !== saved[f]) settings[f] = shown[f] === "" ? null : shown[f] === "on";
  for (const store of live()) {
    const now = shown.stores[store];
    const was = saved.stores[store];
    const out = {};
    if (now.method !== was.method) out.method = now.method || null;
    if (now.candidates !== was.candidates) out.candidates = whole(now.candidates, defaults?.limits?.candidates?.[1] ?? null, "Candidates");
    if (Object.keys(out).length) changed[store] = out;
  }
  return { settings, stores: changed };
}

const anyChange = (body) => Object.keys(body.settings).length + Object.keys(body.stores).length > 0;

// Whether the store's fields in the panel differ from the chat's values for it.
function storeEdited(ui, store) {
  const now = shownFields(ui).stores[store];
  const was = savedFields().stores[store];
  return now.method !== was.method || now.candidates !== was.candidates;
}

// Set as default has something to save: a value the chat holds for the store, or an edit.
function canSave(ui, store) {
  const own = chat.settings?.stores?.[store] ?? {};
  return own.method != null || own.candidates != null || storeEdited(ui, store);
}

// What a chat field inherits when the chat's own is blank, as {value, from}; null when that is
// unknown (Search defaults could not be read).
function chatInherit(field) {
  const force = chat.in_force?.[field];
  if (force && force.from !== "chat") return force;
  if (!defaults) return null;
  const saved = defaults.saved?.[field];
  return saved != null ? { value: saved, from: "settings" } : { value: defaults.recommended?.[field], from: "recommended" };
}

// What a store field inherits when the chat's own is blank: the store's, Settings' or the
// recommended value. Null when unknown (the store's detail could not be read).
function storeInherit(store, field) {
  const force = chat.in_force?.stores?.[store]?.[field];
  if (force && force.from !== "chat") return force;
  return details[store]?.chat_settings?.[field] ?? null;
}

// "8 · Settings": the inherited value and where it is from; with the value unknown, the
// source alone, never a made-up value.
function inheritText(inherited, show, fallback) {
  if (!inherited || inherited.value == null) return fallback;
  return `${show(inherited.value)} · ${SOURCES[inherited.from] ?? inherited.from}`;
}

const option = (value, chosen, label, disabled = false) =>
  `<option value="${esc(value)}"${value === chosen ? " selected" : ""}${disabled ? " disabled" : ""}>${esc(label)}</option>`;

// Inherit, On and Off for a chat switch.
function switchHtml(field, label, value, hidden) {
  const inherited = inheritText(chatInherit(field), (x) => (x ? "on" : "off"), "Settings");
  const options = option("", value, `Inherit (${inherited})`) + option("on", value, "On") + option("off", value, "Off");
  return `<label${hidden ? " hidden" : ""}>${esc(label)} <select name="${field}" aria-label="${esc(label)}">${options}</select></label>`;
}

// The chat methods with their catalog labels. A paid one may cost AI; one the store can't run
// is disabled with its reason.
function methodOptions(store, value) {
  const inherited = inheritText(storeInherit(store, "method"), (m) => methodLabel(m, cat), "store");
  const why = details[store]?.methods_unavailable ?? {};
  const methods = [...new Set([...(cat?.chat_methods ?? defaults?.methods ?? []), ...(value ? [value] : [])])];
  const documents = details[store]?.qa_type === "documents";
  const listed = methods.filter((m) => !(documents && cat?.methods?.[m]?.group === "keywords" && m !== value)).map((m) => {
    const paid = cat?.methods?.[m]?.paid;
    const label = methodLabel(m, cat) + (paid && paid !== "never" ? " (may cost AI)" : "") + (why[m] ? ` (${why[m]})` : "");
    return option(m, value, label, Boolean(why[m]));
  });
  return option("", value, `Inherit (${inherited})`) + listed.join("");
}

function storeFieldsHtml(store, shown, ui) {
  const f = shown.stores[store];
  const candidates = inheritText(storeInherit(store, "candidates"), String, "store");
  const note = ui.notes[store];
  const off = ui.busy || !canSave(ui, store) ? " disabled" : "";
  const confirm =
    ui.confirm?.store === store
      ? `<div class="confirm"><span class="ask">${esc(ui.confirm.text)}</span>
        <button type="button" data-save${ui.busy || pending[chat.id] != null ? " disabled" : ""}>Yes, save</button> <button type="button" class="secondary" data-unsave>Cancel</button></div>`
      : "";
  return `<fieldset data-edit="${esc(store)}"><legend>${esc(store)}</legend><div class="row">
      <label>Method <select name="method" aria-label="Method">${methodOptions(store, f.method)}</select></label>
      <label>Candidates <input name="candidates" aria-label="Candidates" type="number" min="1" ${maxAttr(defaults?.limits?.candidates?.[1])} step="1" value="${esc(f.candidates)}" placeholder="${esc(candidates)}"></label>
      <button type="button" class="secondary" data-default${off}>Set as default</button>
      <span class="edit-note${note?.bad ? " err" : " muted"}" role="status">${esc(note?.text ?? "")}</span>
    </div>${confirm}</fieldset>`;
}

// The chat's one settings panel, under its latest message: its own fields, then each store's.
// A blank field or an Inherit option inherits. Re-run sends what changed, all at once.
function panelHtml(n) {
  const ui = panelOf(chat.id);
  const shown = shownFields(ui);
  const off = ui.busy || pending[chat.id] != null ? " disabled" : "";
  const size = inheritText(chatInherit("evidence_size"), String, "Settings");
  const fields = `<div class="row">
      <label>Evidence size <input name="evidence_size" aria-label="Evidence size" type="number" min="1" ${maxAttr(defaults?.limits?.evidence_size?.[1])} step="1" value="${esc(shown.evidence_size)}" placeholder="${esc(size)}"></label>
      ${switchHtml("rerank", "Rerank", shown.rerank, false)}
      ${switchHtml("rewrite", "Rewrite follow-ups", shown.rewrite, !chat.in_force?.rewrite_available)}
    </div>`;
  const confirm = ui.confirm?.rerun
    ? `<div class="confirm"><span class="ask">${esc(REWRITE_AGAIN)}</span>
        <button type="button" data-rerun-yes${off}>Yes, re-run</button> <button type="button" class="secondary" data-rerun-no>Cancel</button></div>`
    : "";
  return `<details class="panel" data-n="${Number(n)}"${ui.open ? " open" : ""}><summary>Settings for this chat</summary>
    <p class="muted">A blank field inherits from the store, then Settings. Re-run searches message ${Number(n)} again and saves the changes on the chat.</p>
    ${fields}<div class="stores">${live().map((s) => storeFieldsHtml(s, shown, ui)).join("")}</div>
    <div class="row"><button type="button" data-rerun${off}>Re-run</button>
      <span class="panel-note${ui.note?.bad ? " err" : " muted"}" role="status">${esc(ui.note?.text ?? "")}</span></div>${confirm}</details>`;
}

// Set as default enabled only where there is something to save, as the reader types.
function refreshDefaults(el) {
  const ui = panelOf(chat.id);
  for (const box of el.querySelectorAll("fieldset[data-edit]")) {
    box.querySelector("[data-default]").disabled = ui.busy || !canSave(ui, box.dataset.edit);
  }
}

// -- a message ---------------------------------------------------------------------------------

// A saved message: the user's text, then the attempt chosen (the latest unless a tab says
// otherwise), with a tab for each attempt when there are several, its Retrieval step, and,
// under the latest message, the settings panel.
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
    body = tabs + attemptHtml(shown, at, ui, chat, hasModel) + retrievalHtml(shown, at, ui, n, chat, hasModel, cat);
    if (n === latestN()) body += panelHtml(n);
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
  if (a && passagesOnly(a, hasModel) && !a.error) return PASSAGES_ONLY;
  return a?.answer ?? `Error: ${a?.error ?? "no answer"}`;
}

// -- drawing the page ------------------------------------------------------------------------

function drawList() {
  $("chat-list").innerHTML = chats
    .map(
      (c) => `<li class="${c.id === openId ? "active" : ""}"><a href="#${esc(c.id)}">${esc(c.title)}
        <span class="stores">${esc(c.stores.join(", "))}</span></a>
        <button type="button" class="secondary" data-delete="${esc(c.id)}" aria-label="Delete chat ${esc(c.title)}">Delete chat…</button></li>`,
    )
    .join("");
}

function chipHtml(store, only) {
  const gone = chat.deleted_stores.includes(store);
  const what = gone ? " · deleted" : details[store] ? ` · ${plural(details[store].count, cat?.qa_types?.[details[store].qa_type]?.item_noun ?? "item")}` : "";
  return `<span class="chip${gone ? " gone" : ""}" data-store="${esc(store)}">${esc(store)}<span class="muted">${esc(what)}</span>
    <button type="button" data-remove="${esc(store)}" aria-label="Remove ${esc(store)}"${only ? " disabled" : ""}>×</button></span>`;
}

function drawChips() {
  if (chat) $("chips").innerHTML = chat.stores.map((s) => chipHtml(s, chat.stores.length === 1)).join("");
}

// A Copy button beside a curl line: Copied for 2 s, or the line selected when the clipboard fails.
for (const button of document.querySelectorAll("[data-copy]")) {
  button.addEventListener("click", async () => {
    const pre = $(button.dataset.copy);
    try {
      await navigator.clipboard.writeText(pre.textContent);
      button.textContent = "Copied";
    } catch {
      getSelection().selectAllChildren(pre);
      button.textContent = "Press Ctrl+C to copy";
    }
    $("copy-status").textContent = button.textContent; // the button's own change is not announced
    clearTimeout(button.timer);
    button.timer = setTimeout(() => {
      button.textContent = "Copy";
      $("copy-status").textContent = ""; // so the next Copy is announced again
    }, 2000);
  });
}

function drawHead() {
  $("empty").hidden = Boolean(chat);
  $("chat-head").hidden = !chat;
  if (!chat) return;
  $("chat-title").textContent = chat.title;
  // The curl lines for this chat, as text: an app sends its next message, or starts a copy.
  $("curl-send").textContent = `curl -s '${location.origin}/chats/${chat.id}/messages?brief=true'` +
    ` -H 'content-type: application/json' -d '{"content":"…"}'`;
  $("curl-copy").textContent = `curl -s ${location.origin}/chats -H 'content-type: application/json'` +
    ` -d '{"from":"${chat.id}"}'`;
  drawChips();
  const others = stores.filter((s) => !chat.stores.includes(s.name));
  $("add-store").innerHTML = ['<option value="">Add a store…</option>']
    .concat(others.map((s) => `<option value="${esc(s.name)}">${esc(s.name)} · ${esc(cat?.qa_types?.[s.qa_type]?.label ?? s.qa_type)}</option>`))
    .join("");
  $("add-store").disabled = !others.length;
  const keep = $("doc-store").value;
  const typeOf = (name) => stores.find((s) => s.name === name)?.qa_type;
  const label = (name) => cat?.qa_types?.[typeOf(name)]?.label ?? typeOf(name) ?? "";
  const targets = live()
    .map((name, i) => ({ name, i, docs: typeOf(name) === "documents" }))
    .sort((a, b) => b.docs - a.docs || a.i - b.i); // Documents stores first, each part in the chat's order
  $("doc-store").innerHTML = targets
    .map(({ name }) => `<option value="${esc(name)}"${name === keep ? " selected" : ""}>${esc(name)}${label(name) ? ` · ${esc(label(name))}` : ""}</option>`)
    .join("");
}

function drawLog() {
  let html = "";
  if (chat) {
    const lines = notesOf(chat.id);
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

// Draws every message again in place, as drawLog does but without scrolling: for a change
// the panel or a step shows, such as the chat's settings or a store's own.
function drawTurns() {
  for (const m of chat?.messages ?? []) drawTurn(m.n);
}

function setBusy() {
  const waiting = chat != null && pending[chat.id] != null;
  $("send").disabled = !chat || waiting;
  $("add-doc").disabled = !chat || adding || !live().length;
  $("doc-store").disabled = adding;
  $("status").textContent = waiting ? "thinking…" : "";
}

// -- loading -----------------------------------------------------------------------------------

async function loadList() {
  chats = await api("/chats");
  pruneNotes(chats);
  drawList();
  const listed = chats.find((c) => c.id === chat?.id);
  if (listed) {
    chat.title = listed.title; // the first message names a new chat
    $("chat-title").textContent = chat.title;
  }
}

// The stores' details for their chips and the panel; a failure is said in the status line
// and changes nothing else.
async function loadDetails(names) {
  const got = await Promise.allSettled(
    names.map(async (s) => (details[s] = await api(`/collections/${encodeURIComponent(s)}`))),
  );
  drawChips();
  drawTurns(); // the panel shows the stores' methods and own settings
  const failed = got.find((r) => r.status === "rejected");
  if (failed) $("status").textContent = failed.reason.message;
}

// The chat's settings and what is in force, read again after the server changed them, and
// every message redrawn with them. Only they are taken from the read: a reply landing
// meanwhile may be newer than the read's messages.
async function loadSettings(id) {
  const got = await api(`/chats/${encodeURIComponent(id)}`);
  if (chat?.id !== id) return;
  chat = { ...chat, settings: got.settings, in_force: got.in_force };
  drawTurns();
}

const EMPTY = "Start a new chat, or open one from the list.";

async function openChat(id) {
  const mine = ++opening;
  openId = id;
  chat = null;
  $("empty").textContent = EMPTY; // a note about the last chat does not outlive it
  $("spoken").textContent = ""; // another chat: nothing of the last one is read out
  $("add-doc-msg").textContent = ""; // the last chat's Add file result is not this chat's
  drawList();
  drawHead();
  drawLog();
  setBusy();
  if (!id) return;
  let late = [];
  try {
    const got = await api(`/chats/${encodeURIComponent(id)}`);
    if (mine !== opening) return;
    late = arrived[id] ?? [];
    delete arrived[id];
    chat = late.reduce((c, merge) => merge(c), got);
  } catch (e) {
    if (mine !== opening) return;
    if (e.status === 404) {
      $("empty").textContent = `No chat ${id}: it may have been deleted. ${EMPTY}`;
      $("empty").hidden = false;
      openId = ""; // the hash is cleared: the sidebar must not treat the missing chat as open
      history.replaceState(null, "", location.pathname);
    } else {
      $("status").textContent = e.message;
    }
    return;
  }
  drawHead();
  drawLog();
  setBusy();
  // A re-run that landed meanwhile saved settings on the chat, which the read may predate.
  if (late.length) loadSettings(id).catch(said);
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
    .map((s) => `<label class="check"><input type="checkbox" value="${esc(s.name)}"> ${esc(s.name)} <span class="muted">${esc(cat?.qa_types?.[s.qa_type]?.label ?? s.qa_type)}</span></label>`)
    .join("");
  $("picking").innerHTML = `<fieldset><legend>Stores to search</legend><div class="checks">${boxes}</div></fieldset>
    <div class="actions"><button type="button" data-start>Start chat</button>
      <button type="button" class="secondary" data-cancel>Cancel</button></div>
    <p id="pick-note" class="err" role="status"></p>`;
  $("picking").querySelector("[data-cancel]").addEventListener("click", () => ($("picking").innerHTML = ""));
  const start = $("picking").querySelector("[data-start]");
  start.addEventListener("click", async () => {
    const picked = [...$("picking").querySelectorAll("input:checked")].map((box) => box.value);
    if (!picked.length) {
      say($("pick-note"), "Pick at least one store.");
      return;
    }
    start.disabled = true; // until the POST settles, so a second click makes no second chat
    try {
      const made = await api("/chats", { method: "POST", body: { stores: picked } });
      $("picking").innerHTML = "";
      await loadList();
      location.hash = made.id; // opens it
    } catch (e) {
      start.disabled = false;
      say($("pick-note"), e.message);
    }
  });
}

function askDelete(id) {
  const title = chats.find((c) => c.id === id)?.title ?? "";
  const box = $("deleting");
  box.innerHTML = `<p>${esc(`Delete chat ${title.replace(/[?\s]+$/, "")}?`)}</p><div class="actions">
    <button type="button" class="danger" data-yes>Yes, delete</button>
    <button type="button" class="secondary" data-keep>Cancel</button></div>`;
  box.querySelector("[data-keep]").addEventListener("click", () => (box.innerHTML = ""));
  box.querySelector("[data-yes]").addEventListener("click", async () => {
    try {
      await api(`/chats/${encodeURIComponent(id)}`, { method: "DELETE" });
      box.innerHTML = "";
      dropNotes(id);
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

// Saves a change to the open chat's stores.
async function patch(body) {
  if (!chat) return;
  const id = chat.id;
  try {
    const saved = await api(`/chats/${encodeURIComponent(id)}`, { method: "PATCH", body });
    if (chat?.id === id) {
      // Only stores that exist can be added, so an added one is not deleted. One taken off stays
      // deleted: the chat's old rows still name it.
      const deleted = chat.deleted_stores.filter((s) => chat.stores.includes(s) || !saved.stores.includes(s));
      chat = { ...chat, ...saved, deleted_stores: deleted };
      drawHead();
      drawTurns();
      setBusy();
      loadSettings(id).catch(said); // what is in force for an added store
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
  if (!content || !chat || pending[chat.id] != null) return;
  const id = chat.id;
  const after = lastN(chat);
  pending[id] = content;
  $("box").value = "";
  drawLog();
  setBusy();
  let message = null;
  let error = null;
  if (aiFailed) await loadModels(); // the AI may be back since the page read it
  try {
    message = await api(`/chats/${encodeURIComponent(id)}/messages`, {
      method: "POST",
      body: { content, model: $("model").value || null },
    });
  } catch (e) {
    error = e.message; // refused: the server kept nothing, so the page keeps it until a reload
    addNotes(id, { after, role: "user", content }, { after, role: "error", content: error });
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
    arrive(id, (c) => withMessage(c, message));
  }
  setBusy();
  $("spoken").textContent = message ? spokenOf(message) : `Error: ${error}`;
}

// A reply for chat `id`, which is loading (open, not yet read): merged when it has, as its read
// may be older than the reply.
function arrive(id, merge) {
  (arrived[id] ??= []).push(merge);
}

// -- re-runs and defaults ----------------------------------------------------------------------

// The re-run will rewrite message n with AI and its latest attempt was not rewritten: a
// follow-up, with rewrite on after `body` and a rewrite model. An attempt from before the
// pipeline, or a re-run of one, keeps its text whatever is in force.
function rewritesAgain(n, body) {
  if (!chat.in_force?.rewrite_available || !chat.messages.some((m) => m.n < n)) return false;
  const p = chat.messages.find((m) => m.n === n)?.attempts.at(-1)?.pipeline;
  if (!p || p.search_from === "legacy" || p.search_from === "rewrite") return false;
  if (!("rewrite" in body.settings)) return Boolean(chat.in_force.rewrite?.value);
  return Boolean(body.settings.rewrite ?? chatInherit("rewrite")?.value ?? true); // unknown: say so
}

// The panel's changes as the re-run's body; a value the route would refuse is said in the
// panel instead, and null returned.
function panelChanges(n) {
  const ui = panelOf(chat.id);
  try {
    return changesOf(ui);
  } catch (e) {
    ui.note = { text: e.message, bad: true };
    drawTurn(n);
    return null;
  }
}

// Re-run: confirms first when it rewrites again, then sends.
function rerun(n) {
  const ui = panelOf(chat.id);
  if (ui.busy) return;
  const body = panelChanges(n);
  if (!body) return;
  if (rewritesAgain(n, body)) {
    ui.confirm = { rerun: true };
    drawTurn(n);
    return;
  }
  sendRerun(n, body).catch(said);
}

// Re-runs message n with `body` and shows the new attempt; the server saves the settings on
// the chat. A refusal is said in the panel and the edits stay. Returns the attempt, or null.
async function sendRerun(n, body) {
  const id = chat.id;
  const ui = panelOf(id);
  ui.busy = true;
  ui.confirm = null;
  ui.note = { text: "Re-running…" };
  drawTurn(n);
  let attempt = null;
  try {
    attempt = await api(`/chats/${encodeURIComponent(id)}/messages/${Number(n)}/attempts`, { method: "POST", body });
    ui.edit = null; // the new attempt ran with them, and the chat holds them
    ui.note = null;
    turnOf(id, n).tab = attempt.attempt;
    if (chat?.id === id) chat = withAttempt(chat, n, attempt);
    else if (openId === id) arrive(id, (c) => withAttempt(c, n, attempt));
    await loadSettings(id).catch(said);
  } catch (e) {
    ui.note = { text: e.message, bad: true };
  }
  ui.busy = false;
  refreshSpend();
  if (chat?.id === id) drawTurn(n);
  return attempt;
}

// Asks before Set as default saves the store's method and candidates, as they will be after
// any unsent edits are re-run.
function askDefault(n, store) {
  const ui = panelOf(chat.id);
  const body = panelChanges(n);
  if (!body) return;
  // Only what the chat will hold is saved: a field it leaves to inherit stays so.
  const shown = shownFields(ui).stores[store];
  const what = [
    shown.method ? methodLabel(shown.method, cat) : "the inherited method",
    shown.candidates ? plural(Number(shown.candidates), "candidate") : "the inherited candidates",
  ];
  let text = `Save ${what[0]} and ${what[1]} as ${store}'s chat settings?`;
  if (anyChange(body)) text += " This also re-runs this message.";
  if (anyChange(body) && rewritesAgain(n, body)) text += ` ${REWRITE_AGAIN}`;
  ui.confirm = { store, text };
  drawTurn(n);
}

// Saves the chat's values for the store as the store's own. The server saves what the chat
// holds, so unsent edits are re-run first.
async function setDefault(n, store) {
  const id = chat.id;
  const ui = panelOf(id);
  ui.confirm = null;
  const body = panelChanges(n);
  if (!body) return;
  if (anyChange(body) && !(await sendRerun(n, body))) return;
  ui.busy = true;
  if (chat?.id === id) drawTurn(n);
  try {
    await api(`/chats/${encodeURIComponent(id)}/stores/${encodeURIComponent(store)}/default`, { method: "POST" });
    ui.notes[store] = { text: `Saved as ${store}'s chat settings.` };
    if (ui.edit) delete ui.edit.stores[store]; // saved: nothing of it is unsent
    // The chat holds the store's values no more, and the store holds new ones.
    await Promise.all([loadSettings(id).catch(said), loadDetails([store])]);
    if (ui.edit) ui.edit = editOf(shownFields(ui)); // against what the chat now holds
  } catch (e) {
    ui.notes[store] = { text: e.message, bad: true };
  }
  ui.busy = false;
  if (chat?.id === id) drawTurn(n);
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
    } else if (openId === id) {
      arrive(id, (c) => withAttempt(c, n, attempt));
    }
  } catch (e) {
    ui.error = e.message;
  }
  ui.busy = false;
  refreshSpend();
  if (chat?.id === id) drawTurn(n);
}

// -- adding documents --------------------------------------------------------------------------

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
function follow(file, jobId) {
  return followJob(jobId, (job) => {
    $("adding").textContent =
      job.status === "queued"
        ? `Adding ${file}: waiting for the store's other job…`
        : `Adding ${file}… ${Math.round((job.progress ?? 0) * 100)}%`;
  });
}

// One file into the store for good, as its own source: upload, estimate, confirm if it spends.
async function addFile(store, file) {
  const name = encodeURIComponent(store);
  $("adding").textContent = `Adding ${file.name}…`;
  const up = await upload(`/collections/${name}/uploads?kind=items&accept=text&filename=${encodeURIComponent(file.name)}`, file);
  const free = up.estimate?.est_usd === 0; // unpriced (null) is not free
  if (!free && !(await askToAdd(file.name, up.estimate ?? { est_usd: null }))) return null;
  $("adding").textContent = `Adding ${file.name}…`;
  const body = { ...manyBody(up, true), calls: up.estimate?.calls, est_usd: up.estimate?.est_usd };
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
  const addLine = (role, content) => {
    addNotes(id, { after: chat?.id === id ? lastN(chat) : start, role, content });
    if (chat?.id === id) drawLog();
  };
  const msg = (text, cls) => {
    $("add-doc-msg").textContent = text;
    $("add-doc-msg").className = cls;
  };
  msg("", "muted");
  adding = true;
  setBusy();
  let added = false;
  for (const file of files) {
    try {
      if (!/\.(txt|md)$/i.test(file.name)) throw new Error("only .txt and .md files can be added here");
      const said = await addFile(store, file);
      if (said) {
        addLine("note", said);
        msg(said, "muted");
      }
      added ||= Boolean(said);
    } catch (e) {
      addLine("error", `${file.name}: ${e.message}`);
      msg(`${file.name}: ${e.message}`, "err");
    }
    refreshSpend();
  }
  $("adding").textContent = "";
  adding = false;
  setBusy();
  if (added) await loadDetails([store]); // its chip shows the items it now holds
}

// -- start -------------------------------------------------------------------------------------

// Reads the answer models. A failed read shows the Connect AI note and is read again when the
// reader comes back to the page (focus) and before the next send, until one succeeds.
async function loadModels() {
  if (reading) return;
  reading = true;
  let ai = null;
  try {
    ai = await api("/ai");
  } catch {
    // no models known: the note below says so
  }
  reading = false;
  aiFailed = ai == null;
  $("model").innerHTML = ai ? modelOptions(ai.models, ai.roles.answer ?? ai.roles.default) : "";
  hasModel = $("model").options.length > 0;
  $("model-pick").hidden = !hasModel;
  $("note").hidden = hasModel;
  $("note").innerHTML = hasModel ? "" : 'Connect AI in <a href="/settings">Settings</a> to get written answers';
  setBusy();
}

const retryModels = () => {
  if (aiFailed) loadModels();
};
window.addEventListener("focus", retryModels);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) retryModels();
});

const said = (e) => say($("status"), e.message);

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
  if (!at("details.panel")) return;
  const panel = panelOf(chat.id);
  if (at("[data-rerun]")) rerun(n);
  else if (at("[data-rerun-yes]")) {
    const body = panelChanges(n);
    if (body) sendRerun(n, body).catch(said);
  } else if (at("[data-rerun-no]") || at("[data-unsave]")) {
    panel.confirm = null;
    drawTurn(n);
  } else if (at("fieldset[data-edit]")) {
    const store = at("fieldset[data-edit]").dataset.edit;
    if (at("[data-default]")) askDefault(n, store);
    else if (at("[data-save]")) setDefault(n, store).catch(said);
  }
});
// An edit is kept as typed, so a redraw keeps it; nothing is sent until Re-run or Set as default.
// An edit that changes nothing is no edit. One made while a confirm shows cancels it: its
// text was worked out before the edit.
const edited = (event) => {
  const el = event.target.closest("details.panel");
  if (!el || !chat) return;
  const ui = panelOf(chat.id);
  ui.edit = editOf(readPanel(el));
  if (ui.confirm) {
    ui.confirm = null;
    for (const box of el.querySelectorAll(".confirm")) box.remove();
  }
  refreshDefaults(el);
};
$("log").addEventListener("input", edited);
$("log").addEventListener("change", edited);
// What is open is kept, so a redraw keeps it. Toggle does not bubble: caught on its way down.
$("log").addEventListener(
  "toggle",
  (event) => {
    const el = event.target;
    const turn = el.closest?.(".turn");
    if (!turn || !chat) return;
    const ui = turnOf(chat.id, Number(turn.dataset.n));
    if (el.matches("details.retrieval")) ui.open = el.open;
    else if (el.matches("details.panel")) panelOf(chat.id).open = el.open;
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

const ready = loadModels()
  .then(async () => (stores = await api("/collections")))
  .then(async () => (cat = await catalog().catch((e) => (said(e), null)))); // labels: not needed to chat
ready.catch(() => {}).then(() => ($("new").disabled = false)); // New chat lists the stores read
ready
  .then(async () => (defaults = await api("/search-defaults").catch((e) => (said(e), null)))) // not needed to chat
  .then(loadList)
  .then(() => openChat(hashId()))
  .catch(said);
