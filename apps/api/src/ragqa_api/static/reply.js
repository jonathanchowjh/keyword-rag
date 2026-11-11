// A chat reply as the page draws it: the answer as safe markdown with its citations linked,
// its sources and cost, and the Retrieval step with its evidence. Bare functions; what
// they read of the open chat comes in as one context object, built per attempt:
// ctx = { at, ui, n, chat, hasModel, cat }. `at` is "<message n>-<attempt>", `ui` the attempt's
// view state, `n` the message number.
import { NONE, esc, methodLabel, passageHtml, plural, sourceLabel, textHtml, usd, whereHtml } from "/static/ui.js";

export const PASSAGES_ONLY = "Passages only: no answer model";

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

// The store's name, marked when it has since been deleted.
const storeText = (store, chat) => (chat?.deleted_stores.includes(store) ? `${store} · deleted` : store);

// One cited passage, listed under the answer: its file and part, as its evidence is.
function sourceHtml(c, ctx) {
  const { at, ui, chat } = ctx;
  const from = (c.source ? ` · ${esc(sourceLabel(c.source))}` : "") + (c.store ? ` · ${esc(storeText(c.store, chat))}` : "");
  return `<li><span class="what">[${Number(c.n)}] ${whereHtml(c)}${from}</span>${passageHtml(c.text, { key: `src-${at}-${Number(c.n)}`, open: ui.expanded.has(`src-${at}-${Number(c.n)}`) })}</li>`;
}

const costHtml = (x) => (x > 0 ? ` <span class="muted">(${usd(x)})</span>` : "");

// One passage the attempt found. Its id, ev-<message n>-<attempt>-<passage n>, is where the
// answer's [n] links, so the attempts of one message never share an id. Its rank is in its
// own store's list; its rerank, the reranker's relevance, is there when the pool was reranked.
function evidenceHtml(e, ctx) {
  const { at, ui, chat } = ctx;
  const key = `ev-${at}-${Number(e.n)}`;
  const what = [
    e.store && storeText(e.store, chat),
    e.source && sourceLabel(e.source),
    e.rank != null ? `rank ${e.rank}${e.store ? ` in ${e.store}` : ""}` : null,
    e.relevance != null ? `rerank ${Number(e.relevance).toFixed(2)}` : null,
  ]
    .filter((x) => x != null && x !== "")
    .join(" · ");
  return `<li id="${key}" data-store="${esc(e.store)}"><span class="what">[${Number(e.n)}] ${whereHtml(e)}${what ? ` · ${esc(what)}` : ""}</span>${passageHtml(e.text, { key, open: ui.expanded.has(key) })}</li>`;
}

// The passages as the server sends them, best match first; each keeps the n the answer cites.
function evidenceListHtml(a, ctx) {
  const passages = a.evidence.map((e) => evidenceHtml(e, ctx)).join("");
  if (!passages) return '<p class="muted">No passages.</p>';
  const order = a.evidence.length >= 2
    ? `<p class="muted evidence-order">${a.answer != null ? "Best match first. [n] is the number the answer cites." : "Best match first."}</p>`
    : "";
  return `${order}<ol class="evidence">${passages}</ol>`;
}

// With no answer model an unanswered attempt is its passages, listed open under the message.
export const passagesOnly = (a, hasModel) => a.answer == null && !hasModel;

// An attempt's answer with its sources and cost. With no answer: without an answer model,
// its passages; with one, its error and cost (or that it is not answered yet) and Answer from
// this attempt.
export function attemptHtml(a, ctx) {
  const { at, ui, hasModel } = ctx;
  if (passagesOnly(a, hasModel)) {
    const note = a.error ? `<div class="meta warn">${esc(a.error)}</div>` : ""; // e.g. a failed rewrite
    return `<div class="msg passages"><p class="said">${esc(PASSAGES_ONLY)}${costHtml(a.usd)}</p>${note}${evidenceListHtml(a, ctx)}</div>`;
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
    ? `<div class="what muted">Sources</div><ul class="sources">${a.citations.map((c) => sourceHtml(c, ctx)).join("")}</ul>`
    : "";
  // Said whenever the answer had no passages.
  const none = a.evidence.length ? "" : '<div class="meta warn no-passages">The stores found no passages, so this answer used none.</div>';
  const note = a.error ? `<div class="meta warn">${esc(a.error)}</div>` : ""; // e.g. a failed rewrite
  const cached = a.pipeline?.answer_cached ? " · answer cached" : ""; // the answer only: a paid rewrite is not
  return `<div class="msg reply"><div class="md" data-text="${esc(a.answer)}">${textHtml(a.answer, (s) => citeHtml(s, cited, at))}</div>${sources}
    ${none}${note}<div class="meta muted">Cost ${usd(a.usd)}${cached}</div></div>`;
}

// -- the retrieval step ------------------------------------------------------------------------

// The method, and for keyword_rag how its query tags were picked and the tags it picked (an
// attempt saved before tags were kept has none). `label` names the method.
function methodText(d, label = (m) => m) {
  if (!d.method) return NONE;
  if (d.method !== "keyword_rag" || !d.search?.query_tags) return label(d.method);
  return `${label("keyword_rag")} · query tags: ${d.search.query_tags}${d.tags?.length ? ` · ${d.tags.join(", ")}` : ""}`;
}

// Where an attempt's search text came from. A follow-up joined with no earlier message is as typed.
function fromText(p, ctx) {
  const { n, chat } = ctx;
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
function countHtml(d, ctx) {
  const { chat, cat } = ctx;
  const gone = d.note === "deleted" && !d.method;
  const count = gone ? `${storeText(d.store, chat)}: not searched` : `${storeText(d.store, chat)}: ${plural(d.found ?? 0, "candidate")}, ${d.used ?? 0} kept`;
  const method = d.method ? ` · ${methodText(d, (m) => methodLabel(m, cat))}` : "";
  const note = d.error ? ` · error: ${d.error}` : d.note && !gone ? ` · ${d.note}` : "";
  return `<li data-store="${esc(d.store)}"${d.error || gone ? ' class="dropped"' : ""}><span class="count">${esc(count)}</span><span class="method muted">${esc(method)}</span><span class="note">${esc(note)}</span></li>`;
}

// An attempt's search as the pipeline ran it: the text and where it came from, what it was
// reranked against and by which reranker, each store's counts, and the pipeline's notes.
function pipelineHtml(a, ctx) {
  const p = a.pipeline;
  const searched = `<p><strong>Searched for:</strong> <span class="searched">${esc(a.search_text)}</span> <span class="from muted">${esc(fromText(p, ctx))}</span></p>`;
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
  const counts = a.decisions.length ? `<ul class="counts">${a.decisions.map((d) => countHtml(d, ctx)).join("")}</ul>` : '<p class="muted">No store was searched.</p>';
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
function recordedHtml(a, ctx) {
  const { chat } = ctx;
  const rows = a.decisions
    .map((d) => {
      const dropped = d.kept === false || d.error || d.reason === "deleted" || d.note === "deleted";
      return `<tr data-store="${esc(d.store)}"${dropped ? ' class="dropped"' : ""}><th scope="row">${esc(storeText(d.store, chat))}</th>
        <td class="method">${esc(methodText(d))}</td><td class="kept">${esc(keptText(d))}</td></tr>`;
    })
    .join("");
  const table = a.decisions.length
    ? `<div class="scroll"><table class="decisions"><thead><tr><th>Store</th><th>Method</th><th>Result</th></tr></thead><tbody>${rows}</tbody></table></div>`
    : '<p class="muted">No store was searched.</p>';
  return `<p><strong>Searched for:</strong> <span class="searched">${esc(a.search_text)}</span></p><h3>Stores</h3>${table}`;
}

// The attempt's Retrieval step, collapsed unless the reader opened it.
export function retrievalHtml(a, ctx) {
  const { ui, hasModel } = ctx;
  const body = a.pipeline ? pipelineHtml(a, ctx) : recordedHtml(a, ctx);
  const evidence = passagesOnly(a, hasModel) ? '<p class="muted">The passages are listed above.</p>' : evidenceListHtml(a, ctx);
  return `<details class="retrieval"${ui.open ? " open" : ""}><summary>Retrieval step</summary>${body}<h3>Evidence</h3>${evidence}</details>`;
}
