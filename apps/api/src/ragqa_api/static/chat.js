import { $, api, catalog, esc, head, methodLabel, modelOptions, refreshSpend, shell, usd } from "/static/ui.js";

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

// -- the conversation, kept per store in sessionStorage; the page works without it ------------

const key = () => `chat:${$("store").value}`;

function load() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(key()));
    return Array.isArray(saved) ? saved : [];
  } catch {
    return [];
  }
}

function save() {
  try {
    sessionStorage.setItem(key(), JSON.stringify(turns));
  } catch {
    // no storage: the chat lives until the page closes
  }
}

function forget() {
  try {
    sessionStorage.removeItem(key());
  } catch {
    // nothing kept, nothing to clear
  }
}

// -- drawing ---------------------------------------------------------------------------------

// The reply's text, escaped, with each [n] that has a source linked to it.
function textHtml(text, cited, at) {
  const have = new Set(cited.map((c) => c.n));
  return esc(text).replace(/\[(\d+)\]/g, (mark, n) =>
    have.has(+n) ? `<a class="cite" href="#src-${at}-${n}">${mark}</a>` : mark,
  );
}

function sourceHtml(c, at) {
  const long = c.text.length > PASSAGE;
  const body = long
    ? `<details><summary>${esc(c.text.slice(0, PASSAGE))}…</summary><div class="full">${esc(c.text)}</div></details>`
    : `<div>${esc(c.text)}</div>`;
  const from = c.source ? ` · ${esc(c.source)}` : "";
  return `<li id="src-${at}-${c.n}"><span class="what">[${c.n}] ${esc(c.id)}${from}</span>${body}</li>`;
}

function replyHtml(t, at) {
  const tags = t.query_tags?.length ? ` · question tags ${esc(t.query_tags.join(", "))}` : "";
  const sources = t.citations.length
    ? `<div class="what muted">Sources</div><ul class="sources">${t.citations.map((c) => sourceHtml(c, at)).join("")}</ul>`
    : "";
  return `<div class="msg reply"><div>${textHtml(t.answer, t.citations, at)}</div>${sources}
    <div class="meta muted">${esc(methodLabel(t.method))}${tags} · ${usd(t.usd)}</div></div>`;
}

function turnHtml(t, at) {
  if (t.role === "user") return `<div class="msg user">${esc(t.content)}</div>`;
  if (t.role === "error") return `<div class="msg err" role="alert">${esc(t.content)}${t.usd > 0 ? ` <span class="muted">(${usd(t.usd)})</span>` : ""}</div>`;
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
  $("status").textContent = busy ? "thinking…" : "";
}

// -- sending ---------------------------------------------------------------------------------

async function ask() {
  const content = $("box").value.trim();
  if (!content || answering || !hasModel || !$("store").value) return;
  const store = $("store").value;
  turns.push({ role: "user", content });
  // The conversation the model sees: the user's turns and its own answers, never an error.
  const messages = turns
    .filter((t) => t.role !== "error")
    .map((t) => (t.role === "user" ? { role: "user", content: t.content } : { role: "assistant", content: t.answer }))
    .slice(-(TURNS + 1));
  $("box").value = "";
  save();
  draw();
  setBusy(true);
  try {
    const r = await api("/chat", {
      method: "POST",
      body: { collection: store, messages, method: $("method").value, model: $("model").value || null },
    });
    if (store !== $("store").value) return; // the store moved on while this was thinking
    turns.push(r.error || r.answer === null ? { role: "error", content: r.error ?? "no answer", usd: r.usd } : { role: "reply", ...r });
  } catch (e) {
    if (store !== $("store").value) return;
    turns.push({ role: "error", content: e.message });
  } finally {
    setBusy(false);
    refreshSpend();
  }
  save();
  draw();
}

// -- the pickers -----------------------------------------------------------------------------

let picking = 0; // bumped by each store pick, so only the newest one draws

async function pickStore() {
  const mine = ++picking;
  const name = $("store").value;
  turns = load();
  draw();
  const detail = await api(`/collections/${encodeURIComponent(name)}`);
  if (mine !== picking) return;
  const methods = detail.methods_available;
  const first = detail.vocabulary_version && methods.includes("keyword_rag") ? "keyword_rag" : "hybrid";
  $("method").innerHTML = methods
    .map((m) => `<option value="${esc(m)}"${m === first ? " selected" : ""}>${esc(methodLabel(m))}</option>`)
    .join("");
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

$("store").addEventListener("change", () => pickStore().catch((e) => ($("status").textContent = e.message)));
$("new").addEventListener("click", () => {
  turns = [];
  forget();
  draw();
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
