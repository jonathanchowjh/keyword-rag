import { $, UNPRICED, api, catalog, esc, head, methodLabel, modelOptions, plural, refreshSpend, shell, upload, usd } from "/static/ui.js";

const PASSAGE = 300; // characters of a source shown before it is expanded
const MAX_EVIDENCE = 20; // the most passages a chat may answer from, the route's rule
const MAX_CANDIDATES = 60; // the most candidates a store may add to the pool, the route's rule
const PASSAGES_ONLY = "Passages only: no answer model";
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
const local = {}; // chat id -> lines the server never kept: refusals and Add document's notes
const turns = {}; // "chat id:message n" (and "chat id:panel") -> what the reader has open or edited there, kept across redraws

// The chat's stores that still exist.
const live = () => (chat ? chat.stores.filter((s) => !chat.deleted_stores.includes(s)) : []);
const lastN = (c) => c.messages.at(-1)?.n ?? 0;
// The newest message with an attempt: the settings panel sits under it.
const latestN = () => chat?.messages.filter((m) => m.attempts.length).at(-1)?.n ?? null;

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

// One passage the attempt found. Its id, ev-<message n>-<attempt>-<passage n>, is where the
// answer's [n] links, so the attempts of one message never share an id. Its rank is in its
// own store's list; its rerank, the reranker's relevance, is there when the pool was reranked.
function evidenceHtml(e, at, ui) {
  const key = `ev-${at}-${Number(e.n)}`;
  const what = [
    `[${Number(e.n)}] ${e.id}`,
    e.store && storeText(e.store),
    e.source,
    e.parts > 1 ? `part ${e.part} of ${e.parts}` : null,
    e.rank != null ? `rank ${e.rank}${e.store ? ` in ${e.store}` : ""}` : null,
    e.relevance != null ? `rerank ${Number(e.relevance).toFixed(2)}` : null,
  ]
    .filter((x) => x != null && x !== "")
    .join(" · ");
  return `<li id="${key}" data-store="${esc(e.store)}"><span class="what">${esc(what)}</span>${passageHtml(e.text, key, ui)}</li>`;
}

function evidenceListHtml(a, at, ui) {
  const passages = [...a.evidence].sort((x, y) => x.n - y.n).map((e) => evidenceHtml(e, at, ui)).join("");
  return passages ? `<ol class="evidence">${passages}</ol>` : '<p class="muted">No passages.</p>';
}

// With no answer model an unanswered attempt is its passages, listed open under the message.
const passagesOnly = (a) => a.answer == null && !hasModel;

// An attempt's answer with its sources and cost. With no answer: without an answer model,
// its passages; with one, its error and cost (or that it is not answered yet) and Answer from
// this attempt. `at` is "<message n>-<attempt>".
function attemptHtml(a, at, ui) {
  if (passagesOnly(a)) {
    const note = a.error ? `<div class="meta warn">${esc(a.error)}</div>` : ""; // e.g. a failed rewrite
    return `<div class="msg passages"><p class="said">${esc(PASSAGES_ONLY)}${costHtml(a.usd)}</p>${note}${evidenceListHtml(a, at, ui)}</div>`;
  }
  if (a.answer == null) {
    const said = a.error
      ? `<div class="msg err">${esc(a.error)}${costHtml(a.usd)}</div>`
      : `<div class="msg unanswered muted">Not answered yet.${costHtml(a.usd)}</div>`;
    const error = ui.error ? `<span class="err" role="status">${esc(ui.error)}</span>` : "";
    const off = ui.busy ? " disabled" : "";
    return `${said}<div class="row"><button type="button" data-answer="${Number(a.attempt)}"${off}>Answer from this attempt</button>${error}</div>`;
  }
  const passages = new Set(a.evidence.map((e) => e.n));
  const cited = a.citations.filter((c) => passages.has(c.n)); // linked only where its passage is drawn
  const sources = a.citations.length
    ? `<div class="what muted">Sources</div><ul class="sources">${a.citations.map((c) => sourceHtml(c, at, ui)).join("")}</ul>`
    : "";
  // Said whenever the answer had no passages.
  const none = a.evidence.length ? "" : '<div class="meta warn no-passages">The stores found no passages, so this answer used none.</div>';
  const note = a.error ? `<div class="meta warn">${esc(a.error)}</div>` : ""; // e.g. a failed rewrite
  return `<div class="msg reply"><div class="md" data-text="${esc(a.answer)}">${textHtml(a.answer, cited, at)}</div>${sources}
    ${none}${note}<div class="meta muted">Cost ${usd(a.usd)}</div></div>`;
}

// -- the retrieval step ------------------------------------------------------------------------

// The method, and for keyword_rag how its query tags were picked and the tags it picked (an
// attempt saved before tags were kept has none). `label` names the method.
function methodText(d, label = (m) => m) {
  if (!d.method) return "–";
  if (d.method !== "keyword_rag" || !d.search?.query_tags) return label(d.method);
  return `${label("keyword_rag")} · query tags: ${d.search.query_tags}${d.tags?.length ? ` · ${d.tags.join(", ")}` : ""}`;
}

// Where an attempt's search text came from. A follow-up joined with no earlier message is as typed.
function fromText(p, n) {
  if (p.search_from === "rewrite") return "rewritten by AI";
  if (p.search_from === "legacy") return "kept from an earlier attempt";
  const earlier = chat.messages.filter((m) => m.n < n).length;
  const joined = p.search_from === "joined" ? Math.min(Number(p.follow_up_turns) || 0, earlier) : 0;
  if (!joined) return "as typed";
  return joined === 1 ? "with the previous message" : `with the ${joined} previous messages`;
}

// The reranker's short name: its model name after the last "/".
const shortName = (model) => String(model ?? "").split("/").at(-1);

// One store's line: what it found, what the evidence kept, the method it ran, and its note or
// error. A store deleted before the attempt was not searched.
function countHtml(d) {
  const gone = d.note === "deleted" && !d.method;
  const count = gone ? `${storeText(d.store)}: not searched` : `${storeText(d.store)}: ${plural(d.found ?? 0, "candidate")}, ${d.used ?? 0} kept`;
  const method = d.method ? ` · ${methodText(d, (m) => methodLabel(m, cat))}` : "";
  const note = d.error ? ` · error: ${d.error}` : d.note && !gone ? ` · ${d.note}` : "";
  return `<li data-store="${esc(d.store)}"${d.error || gone ? ' class="dropped"' : ""}><span class="count">${esc(count)}</span><span class="method muted">${esc(method)}</span><span class="note">${esc(note)}</span></li>`;
}

// An attempt's search as the pipeline ran it: the text and where it came from, what it was
// reranked against and by which reranker, each store's counts, and the pipeline's notes.
function pipelineHtml(a, p, n) {
  const searched = `<p><strong>Searched for:</strong> <span class="searched">${esc(a.search_text)}</span> <span class="from muted">${esc(fromText(p, n))}</span></p>`;
  let reranked = "";
  if (p.merged_by === "rerank") {
    const by = p.rerank_model ? `<span class="muted">by ${esc(shortName(p.rerank_model))}</span>` : "";
    reranked =
      p.rerank_text !== a.search_text
        ? `<p class="reranked"><strong>Reranked against:</strong> <span class="rerank-text">${esc(p.rerank_text)}</span> ${by}</p>`
        : `<p class="reranked"><strong>Reranked</strong> ${by}</p>`;
  } else if (!p.rerank) {
    reranked = '<p class="reranked muted">Merged by rank: rerank is off.</p>';
  }
  // How many of the stores' candidates were pooled for the rerank (or the merge by rank):
  // Rerank up to caps it, so more candidates can change nothing. Old attempts lack the pool.
  const found = a.decisions.reduce((sum, d) => sum + (d.found ?? 0), 0);
  const cap = p.rerank_pool == null ? "" : ` (Rerank up to ${p.rerank_pool})`;
  const pooled = p.pool == null ? "" : `<p class="pooled muted">${esc(`Pooled ${p.pool} of ${found}${cap}`)}</p>`;
  const counts = a.decisions.length ? `<ul class="counts">${a.decisions.map(countHtml).join("")}</ul>` : '<p class="muted">No store was searched.</p>';
  const notes = p.notes?.length ? `<ul class="notes">${p.notes.map((x) => `<li class="warn">${esc(x)}</li>`).join("")}</ul>` : "";
  return `${searched}${reranked}${pooled}<h3>Stores</h3>${counts}${notes}`;
}

// An old decision's outcome as it was recorded: kept or dropped, with its reason. One saved
// before stores were kept or dropped (it has no `kept`) was kept unless it failed.
function keptText(d) {
  if (d.reason === "deleted" || d.note === "deleted") return "deleted";
  if (d.error) return `dropped · error: ${d.error}`;
  const why = d.reason ?? d.note;
  return `${d.kept === false ? "dropped" : "kept"}${why ? ` · ${why}` : ""}`;
}

// An attempt saved before the pipeline: its search text and each store's decision, read only.
function recordedHtml(a) {
  const rows = a.decisions
    .map((d) => {
      const dropped = d.kept === false || d.error || d.reason === "deleted" || d.note === "deleted";
      return `<tr data-store="${esc(d.store)}"${dropped ? ' class="dropped"' : ""}><th scope="row">${esc(storeText(d.store))}</th>
        <td class="method">${esc(methodText(d))}</td><td class="kept">${esc(keptText(d))}</td></tr>`;
    })
    .join("");
  const table = a.decisions.length
    ? `<div class="scroll"><table class="decisions"><thead><tr><th>Store</th><th>Method</th><th>Result</th></tr></thead><tbody>${rows}</tbody></table></div>`
    : '<p class="muted">No store was searched.</p>';
  return `<p><strong>Searched for:</strong> <span class="searched">${esc(a.search_text)}</span></p><h3>Stores</h3>${table}`;
}

// The attempt's Retrieval step, collapsed unless the reader opened it.
function retrievalHtml(a, at, ui, n) {
  const body = a.pipeline ? pipelineHtml(a, a.pipeline, n) : recordedHtml(a);
  const evidence = passagesOnly(a) ? '<p class="muted">The passages are listed above.</p>' : evidenceListHtml(a, at, ui);
  return `<details class="retrieval"${ui.open ? " open" : ""}><summary>Retrieval step</summary>${body}<h3>Evidence</h3>${evidence}</details>`;
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

// A number field as the route takes it: blank is null (inherit). Throws, with what to fix, on
// a value the route would refuse.
function whole(value, max, name) {
  if (value.trim() === "") return null;
  const x = Number(value);
  if (!(Number.isInteger(x) && x >= 1 && x <= max)) throw new Error(`${name} is a whole number from 1 to ${max}.`);
  return x;
}

// The re-run's body: every field the panel changes from the chat's own values, null for one
// set back to inherit.
function changesOf(ui) {
  const saved = savedFields();
  const shown = shownFields(ui);
  const settings = {};
  const changed = {};
  if (shown.evidence_size !== saved.evidence_size) settings.evidence_size = whole(shown.evidence_size, MAX_EVIDENCE, "Evidence size");
  for (const f of ["rerank", "rewrite"]) if (shown[f] !== saved[f]) settings[f] = shown[f] === "" ? null : shown[f] === "on";
  for (const store of live()) {
    const now = shown.stores[store];
    const was = saved.stores[store];
    const out = {};
    if (now.method !== was.method) out.method = now.method || null;
    if (now.candidates !== was.candidates) out.candidates = whole(now.candidates, MAX_CANDIDATES, "Candidates");
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
  const listed = methods.map((m) => {
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
      <label>Candidates <input name="candidates" aria-label="Candidates" type="number" min="1" max="${MAX_CANDIDATES}" step="1" value="${esc(f.candidates)}" placeholder="${esc(candidates)}"></label>
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
      <label>Evidence size <input name="evidence_size" aria-label="Evidence size" type="number" min="1" max="${MAX_EVIDENCE}" step="1" value="${esc(shown.evidence_size)}" placeholder="${esc(size)}"></label>
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
    body = tabs + attemptHtml(shown, at, ui) + retrievalHtml(shown, at, ui, n);
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
  if (a && passagesOnly(a) && !a.error) return PASSAGES_ONLY;
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
  // The curl lines for this chat, as text: an app sends its next message, or starts a copy.
  $("curl-send").textContent = `curl -s '${location.origin}/chats/${chat.id}/messages?brief=true'` +
    ` -H 'content-type: application/json' -d '{"content":"…"}'`;
  $("curl-copy").textContent = `curl -s ${location.origin}/chats -H 'content-type: application/json'` +
    ` -d '{"from":"${chat.id}"}'`;
  drawChips();
  const others = stores.filter((s) => !chat.stores.includes(s.name));
  $("add-store").innerHTML = ['<option value="">Add a store…</option>']
    .concat(others.map((s) => `<option value="${esc(s.name)}">${esc(s.name)} · ${esc(s.qa_type)}</option>`))
    .join("");
  $("add-store").disabled = !others.length;
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
  let late = [];
  try {
    const got = await api(`/chats/${encodeURIComponent(id)}`);
    if (mine !== opening) return;
    late = arrived[id] ?? [];
    delete arrived[id];
    chat = late.reduce((c, merge) => merge(c), got);
  } catch (e) {
    if (mine === opening) $("status").textContent = e.message;
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

loadModels()
  .then(async () => (stores = await api("/collections")))
  .then(async () => (cat = await catalog().catch((e) => (said(e), null)))) // labels: not needed to chat
  .then(async () => (defaults = await api("/search-defaults").catch((e) => (said(e), null)))) // not needed to chat
  .then(loadList)
  .then(() => openChat(hashId()))
  .catch(said);
