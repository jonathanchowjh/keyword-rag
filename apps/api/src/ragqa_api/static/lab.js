import { $, api, catalog, costMark, esc, head, methodLabel, modelOptions, passageHtml, refreshSpend, say, shell, usd } from "/static/ui.js";

const PRESET = new Set(["hybrid", "kw_individual", "kw_rrf"]);

shell("lab");
$("head").innerHTML = head({
  title: "Lab",
  sub: "One question, several methods side by side. Highlighted chips are the keywords the question and the item share.",
});

// A method's plain label with its code id beside it.
const named = (id) => `${esc(methodLabel(id))}<code>${esc(id)}</code>`;

async function loadCollections() {
  const list = await api("/collections");
  $("collection").innerHTML = list.map((c) => `<option>${esc(c.name)}</option>`).join("");
  if (list.length) await loadMethods();
  else say($("status"), "No stores yet. Create one from the Stores page.");
}

let asking = 0; // bumped by each pick, so only the newest one draws the method list

async function loadMethods() {
  const mine = ++asking;
  const name = $("collection").value;
  const detail = await api(`/collections/${encodeURIComponent(name)}`);
  if (mine !== asking) return; // the choice moved on while this was loading
  $("methods").innerHTML = detail.methods_available
    .map(
      (m) => `<label><input type="checkbox" value="${esc(m)}" ${PRESET.has(m) ? "checked" : ""}> ${named(m)}${costMark(m, detail)}</label>`,
    )
    .join("");
}

function chipsHtml(words, on = new Set()) {
  return words.map((w) => `<span class="chip ${on.has(w) ? "on" : ""}">${esc(w)}</span>`).join("");
}

// A hit's keywords (lists) or tags (a vocabulary), lit where they match the question, and,
// for Keyword-RAG, the channels that found it.
function hitCard(hit) {
  const matched = new Set(hit.matched || []);
  const words = Object.entries(hit.keywords || {})
    .filter(([facet]) => facet !== "all")
    .flatMap(([, list]) => list);
  const chips = chipsHtml([...words, ...(hit.tags || [])], matched);
  const found = hit.channels?.length
    ? `<div class="muted" data-channels>found by ${esc(hit.channels.join(", "))}</div>` : "";
  const gold = hit.answer ? `<span class="muted">example answer ${esc(hit.answer)}</span>` : "";
  return `<div class="hit"><div class="id"><span>${esc(hit.id)}</span><span>score ${hit.score.toFixed(2)}</span></div>
    ${passageHtml(hit.text)}
    <div>${chips}</div>${found}${gold}</div>`;
}

// Keyword-RAG's tags for the question: what its keywords channel searched with.
function queryTagsHtml(v) {
  if (!("query_tags" in v) || v.error) return "";
  const tags = v.query_tags || [];
  const shown = tags.length ? chipsHtml(tags, new Set(tags))
    : '<span class="muted">none, so the keywords channel found nothing</span>';
  return `<div data-query-tags><span class="muted">Question tags</span> ${shown}</div>`;
}

function column(method, v) {
  return `<section class="col">
    <h2><span>${named(method)}</span><span class="muted">${usd(v.usd)}</span></h2>
    ${v.error ? `<div class="err">${esc(v.error)}</div>` : ""}
    ${queryTagsHtml(v)}
    ${v.hits.map(hitCard).join("") || (v.error ? "" : '<div class="muted">no retrieval</div>')}
    ${v.answer !== null ? `<div class="answer">Answer: ${esc(v.answer)}</div>` : ""}
  </section>`;
}

$("collection").addEventListener("change", loadMethods);

$("form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const methods = [...document.querySelectorAll("#methods input:checked")].map((i) => i.value);
  const choices = $("choices").value.split("\n").map((s) => s.trim()).filter(Boolean);
  const body = {
    collection: $("collection").value,
    question: $("question").value,
    methods,
    choices: choices.length ? choices : null,
    k: +$("k").value,
    rrf_k: +$("rrfk").value,
    answer: $("answer").checked,
    model: $("model").value || null,
  };
  $("run").disabled = true;
  say($("status"), "running…");
  try {
    const r = await api("/query", { method: "POST", body });
    $("results").innerHTML = Object.entries(r.results).map(([m, v]) => column(m, v)).join("");
    say($("status"), "");
    refreshSpend();
  } catch (e) {
    $("status").innerHTML = `<span class="err">${esc(e.message)}</span>`;
  } finally {
    $("run").disabled = false;
  }
});

// The answer model: enabled, priced models from Settings, the answer job's choice first.
async function loadModels() {
  const ai = await api("/ai");
  $("model").innerHTML = modelOptions(ai.models, ai.roles.answer ?? ai.roles.default);
}

// Labels come from the catalog, so wait for it before drawing method names.
catalog()
  .catch(() => {})
  .then(loadModels)
  .then(loadCollections)
  .catch((e) => say($("status"), e.message));
