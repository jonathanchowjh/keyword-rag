import { $, UNPRICED, api, count, esc, head, modelOptions, pct, plural, refreshSpend, shell, upload, usd } from "/static/ui.js";

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
let check = null; // the routing check shown: {id, stores, busy, result, error}, for one chat and its stores
let checks = 0; // bumped by each check, so only the newest one draws

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

// The store's name, marked when it has since been deleted.
const storeText = (store) => (chat?.deleted_stores.includes(store) ? `${store} · deleted` : store);

// One cited passage, listed under the answer.
function sourceHtml(c, at, ui) {
  const part = c.parts > 1 ? ` · part ${esc(c.part)} of ${esc(c.parts)}` : "";
  const from = (c.source ? ` · ${esc(c.source)}` : "") + part + (c.store ? ` · ${esc(storeText(c.store))}` : "");
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
  // Said whenever the answer had no passages: no store was kept, or the kept ones found none.
  const why = a.decisions.some((d) => d.kept)
    ? "The stores kept found no passages, so this answer used none."
    : "No store was kept, so this answer used no store passages.";
  const none = a.evidence.length ? "" : `<div class="meta warn none-kept">${why}</div>`;
  const note = a.error ? `<div class="meta warn">${esc(a.error)}</div>` : ""; // e.g. a failed rewrite
  return `<div class="msg reply"><div class="md" data-text="${esc(a.answer)}">${textHtml(a.answer, cited, at)}</div>${sources}
    ${none}${note}<div class="meta muted">Cost ${usd(a.usd)}</div></div>`;
}

// -- the retrieval step ------------------------------------------------------------------------

const two = (x) => (x == null ? "–" : Number(x).toFixed(2));
const unset = (x, shown = x) => (x == null ? "unset" : shown);

// The method, and for keyword_rag how its query tags were picked and the tags it picked (an
// attempt saved before tags were kept has none).
function methodText(d) {
  if (!d.method) return "–";
  if (d.method !== "keyword_rag" || !d.search?.query_tags) return d.method;
  return `keyword_rag · query tags: ${d.search.query_tags}${d.tags?.length ? ` · ${d.tags.join(", ")}` : ""}`;
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

// One store's row; with `routed`, its route score and route bar too (a decision saved
// before routing has neither key: a dash).
function decisionHtml(d, routed) {
  const route = routed
    ? `<td class="num route">${esc(two(d.route))}</td><td class="num route-bar">${esc(two(d.route_bar))}</td>`
    : "";
  return `<tr data-store="${esc(d.store)}"${d.kept ? "" : ' class="dropped"'}><th scope="row">${esc(storeText(d.store))}</th>
    <td class="method">${esc(methodText(d))}</td>${route}<td class="num best">${esc(two(d.best))}</td>
    <td class="num bar">${esc(barText(d))}</td><td class="kept">${esc(keptText(d))}</td></tr>`;
}

// The Stores table shows route columns while the chat routes by profile, and always for an
// attempt that was routed so.
const routedBy = (a) => chat.settings.routing === "profiles" || a.decisions.some((d) => d.route != null || d.route_bar != null);

// One passage the answer was given. Its id, ev-<message n>-<attempt>-<passage n>, is where
// the answer's [n] links, so the attempts of one message never share an id.
function evidenceHtml(e, at, ui) {
  const key = `ev-${at}-${Number(e.n)}`;
  const what = [`[${Number(e.n)}] ${e.id}`, e.store && storeText(e.store), e.source, e.parts > 1 ? `part ${e.part} of ${e.parts}` : null]
    .concat(e.rank != null ? [`rank ${e.rank}`] : [], e.channels?.length ? [e.channels.join(", ")] : [])
    .filter((x) => x != null && x !== "")
    .join(" · ");
  return `<li id="${key}" data-store="${esc(e.store)}"><span class="what">${esc(what)}</span>${passageHtml(e.text, key, ui)}</li>`;
}

// The built-in method, used when none is saved: keyword_rag where it can run, else hybrid,
// as Chats._decide picks it.
const builtInMethod = (own) => (own?.methods_available?.includes("keyword_rag") ? "keyword_rag" : "hybrid");

// What the chat's next search uses for the store, as the Edit form's input values: the
// store's saved defaults with the chat's held override on top. The defaults come from the
// store's detail; while it is missing (its fetch failed), from `d`, a decision of this chat.
// In the override, an absent field is the saved value and a null one the built-in default:
// method "" (the built-in choice), k the chat's evidence size, and the eval set's bar.
function nextBase(store, d) {
  const own = details[store];
  const held = chat.overrides?.[store] ?? {};
  const size = chat.settings.evidence_size;
  const savedMethod = own ? (own.chat_method ?? builtInMethod(own)) : (d?.method ?? "hybrid");
  const savedK = own ? (own.chat_k ?? size) : (d?.k ?? size);
  const savedBar = own ? (own.chat_bar ?? null) : d?.bar_source === "manual" ? d.bar : null; // a held bar equal to it is the store's
  const base = (own ? own.search : d?.search) ?? {};
  const weights = { ...base.weights, ...held.search?.weights };
  const tags = held.search?.query_tags ?? base.query_tags ?? "similarity";
  const heldBar = "bar" in held && held.bar !== savedBar; // one equal to the saved bar is the store's
  const out = {
    method: "method" in held ? (held.method ?? "") : savedMethod,
    k: String("k" in held ? (held.k ?? size) : savedK),
    query_tags: tags === "oracle" ? "classifier" : tags, // a chat has no gold: as Chats._decide
    ownBar: !heldBar,
    setBar: heldBar && held.bar === null,
    bar: heldBar && held.bar !== null ? String(held.bar) : "",
    keep: Boolean(held.keep),
  };
  for (const c of CHANNELS) out[c] = String(weights[c] ?? 1);
  return out;
}

// The decision for the store in the attempt message n shows, if it has one.
function decisionOf(n, store) {
  const m = chat.messages.find((x) => x.n === n);
  return m ? shownAttempt(m, turnOf(chat.id, n))?.decisions.find((d) => d.store === store) : undefined;
}

// The store's override for what the form changes from the next search's settings, and the
// held fields to drop: [override, drop].
function changesOf(n, store) {
  const base = nextBase(store, decisionOf(n, store));
  return overrideOf(turnOf(chat.id, n).edits[store] ?? base, base);
}

const barFrom = (e) => (e.ownBar ? "store" : e.setBar ? "set" : "typed");

// The override for what `edit` changes from `base`, only the fields changed, and the held
// fields to drop: [override, drop]. A null clears the saved value: the built-in method, a
// blank k (the chat's evidence size), the eval set's bar. "Use the store's bar" over a held
// bar drops the held one, so the next search uses what is saved. Throws, with what to fix,
// on a value the route would refuse.
function overrideOf(edit, base) {
  const out = {};
  if (edit.method !== base.method) out.method = edit.method || null;
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
  const drop = [];
  const from = barFrom(edit);
  if (from === "store" && barFrom(base) !== "store") drop.push("bar");
  else if (from !== barFrom(base) || (from === "typed" && edit.bar !== base.bar)) {
    const bar = edit.bar.trim() === "" ? NaN : Number(edit.bar);
    if (from === "typed" && !(bar >= -1 && bar <= 1)) throw new Error("A bar is a number from -1 to 1.");
    out.bar = from === "set" ? null : bar;
  }
  if (edit.keep !== base.keep) out.keep = edit.keep;
  return [out, drop];
}

function readEdit(box) {
  const field = (name) => box.querySelector(`[name="${name}"]`);
  const out = { ownBar: field("ownBar").checked, setBar: field("setBar").checked, keep: field("keep").checked };
  for (const name of ["method", "k", "query_tags", "bar", ...CHANNELS]) out[name] = field(name).value;
  return out;
}

const option = (value, chosen, label = value) =>
  `<option value="${esc(value)}"${value === chosen ? " selected" : ""}>${esc(label)}</option>`;

// A store's Edit form: its unsent edit when there is one, else what the next search uses. `d`
// is the store's decision in the attempt shown, if any.
function editHtml(store, d, ui) {
  const base = nextBase(store, d);
  const e = ui.edits[store] ?? base;
  const own = details[store];
  const methods = [...new Set([...(own?.methods_available ?? []), base.method, e.method])].filter(Boolean);
  const methodLabel = (m) => (own?.chat_method === m ? `${m} (saved)` : m);
  const builtIn = option("", e.method, `built-in default (${builtInMethod(own)})`);
  const size = chat.settings.evidence_size;
  const saved = own
    ? `<p class="saved muted">Saved defaults: method ${esc(unset(own.chat_method))}, k ${esc(unset(own.chat_k))}, bar ${esc(unset(own.chat_bar, two(own.chat_bar)))}.
        The built-in method, a blank k (the chat's evidence size, ${Number(size)}) and the eval set's bar clear a saved value: Re-run retrieval shows the result, and Set as default saves it.</p>`
    : "";
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
      <label>Method <select name="method">${builtIn}${methods.map((m) => option(m, e.method, methodLabel(m))).join("")}</select></label>
      <label>k <input name="k" type="number" min="1" max="${MAX_EVIDENCE}" step="1" value="${esc(e.k)}" placeholder="${Number(size)}"></label>
    </div><div class="row kw-only"><span class="muted">keyword_rag only:</span>
      <label>Query tags <select name="query_tags">${QUERY_TAGS.map((t) => option(t, e.query_tags)).join("")}</select></label>
      ${weights}
    </div><div class="row">
      <label>Bar <input name="bar" type="number" min="-1" max="1" step="0.01" value="${esc(e.bar)}" placeholder="${esc(d && d.bar_source !== "override" ? two(d.bar) : "")}"${e.ownBar || e.setBar ? " disabled" : ""}></label>
      <label class="check"><input name="ownBar" type="checkbox"${e.ownBar ? " checked" : ""}> Use the store's bar</label>
      <label class="check"><input name="setBar" type="checkbox"${e.setBar ? " checked" : ""}> Use the eval set's bar</label>
      <label class="check"><input name="keep" type="checkbox"${e.keep ? " checked" : ""}> Search anyway</label>
    </div>${saved}<div class="row">
      <button type="button" data-rerun${off}>Re-run retrieval</button>
      <button type="button" class="secondary" data-default${off}>Set as default</button>
      <span class="edit-note${note?.bad ? " err" : " muted"}" role="status">${esc(note?.text ?? "")}</span>
    </div>${confirm}</fieldset>`;
}

// The attempt's Retrieval step, collapsed unless the reader opened it: the search text, each
// store's decision as it ran, the passages, and, on the latest attempt, an Edit form for each
// of the chat's stores still there, showing what the chat's next search uses.
function retrievalHtml(a, at, ui, latest) {
  const passages = [...a.evidence].sort((x, y) => x.n - y.n).map((e) => evidenceHtml(e, at, ui)).join("");
  const routed = routedBy(a);
  const routeHeads = routed ? '<th class="num">Route score</th><th class="num">Route bar</th>' : "";
  const stores = a.decisions.length
    ? `<div class="scroll"><table class="decisions"><thead><tr><th>Store</th><th>Method</th>${routeHeads}<th class="num">Best match</th>
        <th class="num">Bar</th><th>Kept</th></tr></thead><tbody>${a.decisions.map((d) => decisionHtml(d, routed)).join("")}</tbody></table></div>`
    : '<p class="muted">No store was searched.</p>';
  const decision = (store) => a.decisions.find((d) => d.store === store);
  const edits = latest ? live().map((store) => editHtml(store, decision(store), ui)).join("") : "";
  const elsewhere = latest ? "" : '<p class="muted">To edit and re-run, open the latest attempt.</p>';
  return `<details class="retrieval"${ui.open ? " open" : ""}><summary>Retrieval step</summary>
    <p><strong>Searched for:</strong> <span class="searched">${esc(a.search_text)}</span></p>
    <h3>Stores</h3>${stores}
    <h3>Evidence</h3>${passages ? `<ol class="evidence">${passages}</ol>` : '<p class="muted">No passages.</p>'}
    ${edits ? `<h3>Edit: settings for the next search in this chat</h3><div class="edits">${edits}</div>` : elsewhere}</details>`;
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
  $("routing").value = chat.settings.routing ?? "all";
  drawCheck();
  const keep = $("doc-store").value;
  $("doc-store").innerHTML = live()
    .map((s) => `<option value="${esc(s)}"${s === keep ? " selected" : ""}>${esc(s)}</option>`)
    .join("");
}

// -- the routing check ------------------------------------------------------------------------

// What the questions checked on are, by split.
const SPLIT_SAYS = {
  held_out: "held-out questions: those outside the dev slice the bars were set on",
  dev: "dev questions: these are the questions the bars were set on",
  full: "questions: the whole eval sets, the dev slice the bars were set on among them",
};
const MODES = { gate: "Search every store", profiles: "By routing profile" };

function modeHtml(key, m) {
  const searched = m.mean_searched == null ? "–" : Number(m.mean_searched).toFixed(1);
  return `<tr><th scope="row">${esc(MODES[key])}</th><td class="num recall">${esc(pct(m.recall))}</td>
    <td class="num precision">${esc(pct(m.precision))}</td><td class="num none">${esc(pct(m.none_rate))}</td>
    <td class="num searched">${esc(searched)}</td></tr>`;
}

function perStoreHtml(store, s) {
  return `<tr><th scope="row">${esc(store)}</th><td class="num questions">${esc(count(s.questions))}</td>
    <td class="num recall-gate">${esc(pct(s.recall_gate))}</td><td class="num recall-profiles">${esc(pct(s.recall_profiles))}</td></tr>`;
}

// Why a store adds no questions: it has no eval set, or (held out) its set is all dev slice.
function addsNoneHtml(store, s, split) {
  let why = `${store} adds no questions: its eval set ${s.eval_set} has none usable here.`;
  if (s.eval_set == null)
    why = `${store} adds no questions: it has no eval set. It still counts against the others when it is kept for their questions.`;
  else if (split === "held_out")
    why = `${store} adds no held-out questions: its eval set ${s.eval_set} has no more than 200 questions, all in the dev slice the bars were set on. Check on the dev slice to score it.`;
  return `<p class="muted adds-none">${esc(why)}</p>`;
}

function checkHtml(c) {
  if (c.busy) return '<p class="muted">Checking routing…</p>';
  const hide = '<div class="row"><button type="button" class="secondary" data-hide>Hide</button></div>';
  if (c.error) return `<p class="err" role="status">${esc(c.error)}</p>${hide}`;
  const r = c.result;
  const said = `<p>On ${esc(count(r.questions))} ${esc(SPLIT_SAYS[r.split] ?? r.split)}.</p>`;
  if (!r.questions) {
    const why = r.split === "held_out" ? ": every question is in its eval set's dev slice. Check on the dev slice instead." : ".";
    return `${said}<p class="muted">${esc(`No questions to check${why}`)}</p>${hide}`;
  }
  const modes = Object.keys(MODES).map((k) => modeHtml(k, r.modes[k])).join("");
  const perStore = Object.entries(r.per_store).map(([store, s]) => perStoreHtml(store, s)).join("");
  const addsNone = Object.entries(r.per_store)
    .filter(([, s]) => !s.questions)
    .map(([store, s]) => addsNoneHtml(store, s, r.split))
    .join("");
  return `${said}<div class="tables">
    <table class="modes decisions"><thead><tr><th scope="col">Mode</th><th scope="col">Recall</th><th scope="col">Precision</th>
      <th scope="col">None kept</th><th scope="col">Stores searched</th></tr></thead><tbody>${modes}</tbody></table>
    <table class="per-store decisions"><thead><tr><th scope="col">Store</th><th scope="col">Its questions</th>
      <th scope="col">Kept, every store</th><th scope="col">Kept, by profile</th></tr></thead><tbody>${perStore}</tbody></table></div>
    ${addsNone}<p class="muted">Recall: the share of questions whose own store was kept. Precision: the share of the kept stores that
      are the question's own. None kept: the share with no store kept. A store's own questions are its eval set's.
      Its questions: how many of the questions are the store's own. Kept, every store and Kept, by profile: the share of the
      store's own questions on which it was kept, when every store is searched and when routing by profile.</p>${hide}`;
}

// The check is shown while the page is on its chat and the chat has the stores it ran on.
function drawCheck() {
  const shown = check && chat && check.id === chat.id && check.stores.join("\n") === live().join("\n");
  $("routing-check").innerHTML = shown ? checkHtml(check) : "";
}

// Scores the gate and the profile router on the chat's stores' own questions. Free: it embeds.
async function checkRouting() {
  if (!chat || live().length < 2) return;
  const mine = ++checks;
  const body = { stores: live(), split: $("routing-split").value };
  check = { id: chat.id, stores: body.stores, busy: true };
  drawCheck();
  setBusy();
  let done;
  try {
    done = { result: await api("/routing/eval", { method: "POST", body }) };
  } catch (e) {
    done = { error: e.message };
  }
  if (mine !== checks) return;
  check = { ...check, busy: false, ...done };
  drawCheck();
  setBusy();
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

// Draws every message again in place, as drawLog does but without scrolling: for a change
// every Edit form shows, such as the chat's held overrides or a store's defaults.
function drawTurns() {
  for (const m of chat?.messages ?? []) drawTurn(m.n);
}

function setBusy() {
  const waiting = chat != null && pending[chat.id] != null;
  $("send").disabled = !chat || waiting || !hasModel;
  $("add-doc").disabled = !chat || adding || !live().length;
  $("doc-store").disabled = adding;
  $("check-routing").disabled = !chat || live().length < 2 || Boolean(check?.busy && check.id === chat.id);
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
  drawTurns(); // the Edit forms show the stores' defaults
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

// Saves a change to the open chat's title, stores, evidence size, rewrite or routing.
async function patch(body) {
  if (!chat) return;
  const id = chat.id;
  try {
    const saved = await api(`/chats/${encodeURIComponent(id)}`, { method: "PATCH", body });
    if (chat?.id === id) {
      // Only stores that exist can be added, so the deleted ones are those still kept.
      const deleted = chat.deleted_stores.filter((s) => saved.stores.includes(s));
      // The server drops a removed store's override; the PATCH reply does not hold them.
      const overrides = Object.fromEntries(Object.entries(chat.overrides ?? {}).filter(([s]) => saved.stores.includes(s)));
      chat = { ...chat, ...saved, deleted_stores: deleted, overrides };
      drawHead();
      setBusy();
      loadDetails(live().filter((s) => !(s in details))); // and redraws the Edit forms
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

// The chat's held overrides, read again after the server changed them, and every Edit form
// redrawn with them. Only they are taken from the read: a reply landing meanwhile may be
// newer than the read's messages.
async function loadOverrides(id) {
  const got = await api(`/chats/${encodeURIComponent(id)}`);
  if (chat?.id !== id) return;
  chat = { ...chat, overrides: got.overrides };
  drawTurns();
}

// Re-runs message n's retrieval with the store's changes as the chat's override, and shows
// the new attempt. A refusal is said beside the store and the edit stays. Returns the attempt,
// or null when there is none.
async function rerun(n, store) {
  const id = chat.id;
  const ui = turnOf(id, n);
  if (ui.busy) return null;
  let override, drop;
  try {
    [override, drop] = changesOf(n, store);
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
    const body = { overrides: Object.keys(override).length ? { [store]: override } : {} };
    if (drop.length) body.drop = { [store]: drop };
    attempt = await api(`/chats/${encodeURIComponent(id)}/messages/${Number(n)}/attempts`, { method: "POST", body });
    delete ui.edits[store]; // the new attempt ran with them
    ui.notes[store] = null;
    ui.tab = attempt.attempt;
    if (chat?.id === id) chat = withAttempt(chat, n, attempt);
    await loadOverrides(id).catch(said); // what every form's next search now uses
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

// Saves the store's settings, as its form shows them, as its default. The server saves the
// override the chat holds, so a form that differs from it is re-run first.
async function setDefault(n, store) {
  const id = chat.id;
  const ui = turnOf(id, n);
  ui.confirm = null;
  let edited = false;
  try {
    const [override, drop] = changesOf(n, store);
    edited = Object.keys(override).length > 0 || drop.length > 0;
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
    delete ui.edits[store];
    // The server dropped the store's override (Search anyway with it) and holds new defaults.
    await Promise.all([loadOverrides(id).catch(said), loadDetails([store])]);
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
$("routing").addEventListener("change", () => patch({ routing: $("routing").value }));
$("check-routing").addEventListener("click", () => checkRouting().catch(said));
$("routing-check").addEventListener("click", (event) => {
  if (!event.target.closest("[data-hide]")) return;
  check = null;
  drawCheck();
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
  // The store's bar and the eval set's are one choice: checking one unchecks the other.
  const other = { ownBar: "setBar", setBar: "ownBar" }[event.target.name];
  if (other && event.target.checked) box.querySelector(`[name="${other}"]`).checked = false;
  const edit = readEdit(box);
  turnOf(chat.id, Number(turn.dataset.n)).edits[box.dataset.edit] = edit;
  box.querySelector('[name="bar"]').disabled = edit.ownBar || edit.setBar;
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
