import { $, api, askDelete, catalog, embedderLabel, esc, head, mayCost, methodLabel, modelOptions, refreshSpend, say, shell, usd } from "/static/ui.js";

shell("settings");
$("head").innerHTML = head({
  title: "Settings",
  sub: "AI writes chat answers, rewrites follow-ups, and builds keyword tags. Search and eval runs on retrieval work without it.",
});

const ENV = "set by environment";

let ai = null; // the last GET /ai
const notes = {}; // connection id -> what its last button said, kept across redraws

// A model id may contain "/"; each segment is encoded on its own.
const modelPath = (m) => `/ai/models/${encodeURIComponent(m.connection_id)}/${m.model_id.split("/").map(encodeURIComponent).join("/")}`;
const connPath = (id) => `/ai/connections/${encodeURIComponent(id)}`;

// Runs one change, then redraws from the server; a refusal shows its reason. The clicked button
// stays off until the redraw, so a second click can't send (and pay for) the request twice.
async function act(change, button = null) {
  say($("msg"), "");
  if (button) button.disabled = true;
  try {
    try {
      await change();
    } catch (e) {
      say($("msg"), e.message);
    }
    await load();
  } finally {
    if (button) button.disabled = false;
  }
}

// -- connections ------------------------------------------------------------------

function connectionHtml(c) {
  // Answered by hand: nothing is sent, so there is no key, nothing to test and no model list.
  const manual = c.kind === "manual";
  const locked = c.from_env.includes("api_key");
  const key = locked ? ENV : c.key ?? "no key";
  const where = c.base_url ? `<code>${esc(c.base_url)}</code>` : "";
  const keyBox = manual ? "" : `<div class="keyline"><label>New key <input type="password" name="key" autocomplete="off"
        ${locked ? `disabled placeholder="${ENV}"` : ""}></label>
        <button type="button" class="secondary" data-act="key" disabled>Save key</button></div>`;
  const status = manual
    ? '<span class="muted">prompts wait for replies: <code>python -m ragqa_core.replies</code> (see Answered by hand in REFERENCE.md)</span>'
    : `<span class="muted">key ${esc(key)}</span>`;
  return `<div class="card conn" data-connection="${esc(c.id)}">
    <div class="top"><strong>${esc(c.label)}</strong><code>${esc(c.kind)}</code>${where}
      ${status}</div>
    ${keyBox}
    <div class="actions flush">
      ${manual ? "" : `<button type="button" class="secondary" data-act="test">Test</button>
      <button type="button" class="secondary" data-act="fetch">Fetch models</button>`}
      ${c.from_env.length ? "" : '<button type="button" class="secondary" data-act="remove">Delete connection…</button>'}
      <span class="result" role="status">${notes[c.id] ?? ""}</span></div>
    <div class="ask"></div>
  </div>`;
}

function drawConnections() {
  $("connections").innerHTML = ai.connections.map(connectionHtml).join("") ||
    '<p class="muted">No connections yet. Add one below.</p>';
}

const testNote = (r) =>
  r.ok ? `<span class="ok">Worked with ${esc(r.model)} (${esc(usd(r.usd))})</span>` : `<span class="err">${esc(r.error)}</span>`;

const ACTIONS = {
  async test(id) {
    notes[id] = "testing…";
    $("connections").querySelector(`[data-connection="${CSS.escape(id)}"] .result`).textContent = notes[id];
    notes[id] = testNote(await api(connPath(id) + "/test", { method: "POST" }));
    refreshSpend();
  },
  async fetch(id) {
    const { added } = await api(connPath(id) + "/models/fetch", { method: "POST" });
    notes[id] = `<span class="muted">${added.length} new model${added.length === 1 ? "" : "s"}</span>`;
  },
  async key(id, box) {
    const input = box.querySelector("[name=key]");
    if (!input.value) return;
    await api(connPath(id), { method: "PATCH", body: { api_key: input.value } });
    input.value = "";
    notes[id] = '<span class="muted">key saved</span>';
  },
};

$("connections").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-act]");
  if (!button) return;
  const box = button.closest("[data-connection]");
  const c = ai.connections.find((x) => x.id === box.dataset.connection);
  if (button.dataset.act === "remove") {
    askDelete(box.querySelector(".ask"), {
      question: `Delete connection ${c.label} (${c.id}), with its key and models? This cannot be undone.`,
      yes: `Yes, delete ${c.label}`,
      path: connPath(c.id),
      gone: "/settings",
    });
    return;
  }
  act(() => ACTIONS[button.dataset.act](box.dataset.connection, box), button);
});

// Save key stays off until something is typed, so an empty box can never clear a key.
$("connections").addEventListener("input", (event) => {
  const input = event.target.closest("[name=key]");
  if (input) input.closest(".keyline").querySelector("button").disabled = !input.value;
});

// -- add form ---------------------------------------------------------------------

function drawPresets() {
  $("add-preset").innerHTML = ai.presets.map((p) => `<option value="${esc(p.id)}">${esc(p.label)}</option>`).join("");
  fillFromPreset();
}

function fillFromPreset() {
  const p = ai.presets.find((x) => x.id === $("add-preset").value);
  if (!p) return;
  $("add-label").value = p.label;
  $("add-url").value = p.base_url ?? "";
  $("add-key").required = false;
  $("add-key").placeholder = "";
}

$("add-preset").addEventListener("change", fillFromPreset);

$("add").addEventListener("submit", (event) => {
  event.preventDefault();
  const body = { preset: $("add-preset").value, id: $("add-id").value.trim(), label: $("add-label").value.trim() || undefined,
    base_url: $("add-url").value.trim() || undefined, api_key: $("add-key").value || undefined };
  act(async () => {
    $("add-go").disabled = true; // until the POST settles, so a second click adds nothing twice
    try {
      await api("/ai/connections", { method: "POST", body });
    } finally {
      $("add-go").disabled = false;
    }
    $("add-id").value = "";
    $("add-key").value = "";
  });
});

// -- models -----------------------------------------------------------------------

const priceBox = (name, value) =>
  `<label>${name === "price_in" ? "Input" : "Output"} <input class="price" name="${name}" type="number" min="0" step="any"
    value="${value ?? ""}"></label>`;

// The price choices, in USD per million tokens: $0, the list prices, none, or typed by hand.
const HAND = "hand";
const NO_PRICE = "none";
const priceKey = (i, o) => `${i}/${o}`;
function priceOptions() {
  const listed = (ai.prices || []).map((p) => ({ value: priceKey(p.price_in, p.price_out),
    label: `${p.label} rates: $${p.price_in} in, $${p.price_out} out` }));
  return [{ value: priceKey(0, 0), label: "$0 (free)" }, ...listed,
    { value: NO_PRICE, label: "No price (can't be used)" }, { value: HAND, label: "Enter by hand…" }];
}

function priceChoice(m) {
  if (m.price_in == null || m.price_out == null) return NO_PRICE;
  const key = priceKey(m.price_in, m.price_out);
  return priceOptions().some((o) => o.value === key) ? key : HAND;
}

function modelHtml(m) {
  const unpriced = m.price_in == null || m.price_out == null;
  const byHand = ai.connections.find((c) => c.id === m.connection_id)?.kind === "manual";
  const choice = priceChoice(m);
  const options = priceOptions()
    .map((o) => `<option value="${esc(o.value)}" ${o.value === choice ? "selected" : ""}>${esc(o.label)}</option>`)
    .join("");
  const note = unpriced ? '<span class="warn">needs a price before use</span>'
    : m.price_in === 0 && m.price_out === 0 && !byHand
      ? '<span class="warn" data-free>$0: the budget does not limit it</span>' : "";
  return `<div class="model" data-model="${esc(m.ref)}" data-connection="${esc(m.connection_id)}" data-id="${esc(m.model_id)}">
    <span class="name">${esc(m.label)}<code>${esc(m.model_id)}</code></span>
    <label><input type="checkbox" name="enabled" ${m.enabled ? "checked" : ""}> Enabled</label>
    <label title="USD per million tokens">Price <select name="price">${options}</select></label>
    <span data-hand ${choice === HAND ? "" : "hidden"}>${priceBox("price_in", m.price_in)}${priceBox("price_out", m.price_out)}</span>
    <button type="button" class="secondary" data-save>Save prices</button>
    ${note}</div>`;
}

function drawModels() {
  $("models").innerHTML = ai.connections
    .map((c) => {
      const rows = ai.models.filter((m) => m.connection_id === c.id).map(modelHtml).join("");
      return `<div class="card" data-connection="${esc(c.id)}"><strong>${esc(c.label)}</strong>
        ${rows || '<p class="muted">No models. Fetch the list, or add one by id.</p>'}
        <div class="row add-model"><label>Add a model by id <input name="new-id"></label>
          <button type="button" class="secondary" data-add>Add model</button></div></div>`;
    })
    .join("");
}

// A row saves whole, on its Save button: both prices or neither. Only that row and the job lists
// redraw, so text typed elsewhere on the page stays.
async function saveRow(row) {
  const number = (name) => {
    const v = row.querySelector(`[name=${name}]`).value;
    return v === "" ? null : Number(v);
  };
  const choice = row.querySelector("[name=price]").value;
  const [chosenIn, chosenOut] = choice === NO_PRICE ? [null, null] : choice.split("/").map(Number);
  const typed = choice === HAND;
  const body = { enabled: row.querySelector("[name=enabled]").checked,
    price_in: typed ? number("price_in") : chosenIn, price_out: typed ? number("price_out") : chosenOut };
  say($("msg"), "");
  if (typed && (body.price_in === null || body.price_out === null)) {
    say($("msg"), "Give both prices, or choose one from the list.");
    return;
  }
  const button = row.querySelector("[data-save]");
  button.disabled = true; // until the PUT answers, so a second click saves nothing twice
  try {
    await api(modelPath({ connection_id: row.dataset.connection, model_id: row.dataset.id }), { method: "PUT", body });
    ai = await api("/ai");
    row.outerHTML = modelHtml(ai.models.find((m) => m.ref === row.dataset.model));
    drawRoles();
  } catch (e) {
    say($("msg"), e.message);
    button.disabled = false;
  }
}

// "Enter by hand…" shows the two price boxes; any other choice hides them.
$("models").addEventListener("change", (event) => {
  if (event.target.name !== "price") return;
  event.target.closest(".model").querySelector("[data-hand]").hidden = event.target.value !== HAND;
});

$("models").addEventListener("click", (event) => {
  const save = event.target.closest("[data-save]");
  if (save) return saveRow(save.closest(".model"));
  if (!event.target.closest("[data-add]")) return;
  const card = event.target.closest("[data-connection]");
  const id = card.querySelector("[name=new-id]").value.trim();
  if (!id) return;
  act(() => api(connPath(card.dataset.connection) + "/models", { method: "POST", body: { model_id: id } }), event.target.closest("button"));
});

// -- jobs -------------------------------------------------------------------------

const JOB_NOTE = {
  default: "Used by any job without its own choice",
  answer: "Writes chat answers, answers in runs and the Lab, judges a Documents run's answers, and the eval-set page's Get the AI's answer",
  summary: "Summarises items, and questions for keyword methods, in stores built with summaries",
  keywords: "Proposes keyword lists and vocabularies, and tags items when a store is built",
  query: "Picks a question's keywords when a store searches with model tags, and the iterative method's hops",
  rewrite: "Rewrites a chat follow-up as one standalone question",
};

// A job's model that can no longer be used still shows, marked and selected, so the
// choice is visible and picking anything else (even "same as default") changes it.
function unusableOption(ref) {
  if (!ref) return "";
  const m = ai.models.find((x) => x.ref === ref);
  const why = !m
    ? "not set up"
    : !m.enabled
      ? "disabled"
      : !m.usable
        ? "no key"
        : m.price_in == null || m.price_out == null
          ? "needs a price"
          : "";
  return why ? `<option value="${esc(ref)}" selected>${esc(m?.label ?? ref)} (${why})</option>` : "";
}

function drawRoles() {
  $("roles").innerHTML = ai.known_roles
    .map((role) => {
      const first = role === "default" ? "none chosen" : "same as default";
      const locked = role === "default" && ai.default_from_env;
      const note = [JOB_NOTE[role], locked ? ENV : ""].filter(Boolean).join("; ");
      return `<label>${esc(cat?.roles?.[role] ?? role)} <select data-role="${esc(role)}"${locked ? " disabled" : ""}><option value="">${first}</option>
        ${unusableOption(ai.roles[role])}${modelOptions(ai.models, ai.roles[role])}</select>
        ${note ? `<span class="muted">${esc(note)}</span>` : ""}</label>`;
    })
    .join("");
}

$("roles").addEventListener("change", (event) => {
  const select = event.target.closest("select[data-role]");
  act(() => api(`/ai/roles/${encodeURIComponent(select.dataset.role)}`, { method: "PUT", body: { model: select.value || null } }));
});

// -- budget -----------------------------------------------------------------------

function drawBudget() {
  const b = ai.budget;
  $("cap").value = b.cap_usd ?? "";
  $("cap").disabled = b.from_env;
  $("cap").placeholder = b.from_env ? ENV : "";
  $("cap-go").disabled = b.from_env;
  const cap = b.cap_usd == null ? "no cap" : usd(b.cap_usd);
  $("spent").textContent = `Spent ${usd(b.spent_usd)} of ${cap} this month${b.from_env ? ` (${ENV})` : ""}`;
}

$("budget").addEventListener("submit", (event) => {
  event.preventDefault();
  act(async () => {
    await api("/ai/budget", { method: "PUT", body: { cap_usd: Number($("cap").value) } });
    refreshSpend();
  });
});

// -- search defaults ----------------------------------------------------------------------

const JINA_TINY = "jinaai/jina-reranker-v1-tiny-en";
const FIELDS = ["dense_model", "method", "candidates", "evidence_size", "rerank", "rerank_model", "rerank_pool", "rewrite", "follow_up_turns", "rrf_k"];

let cat = null;
let defaults = null; // the last GET /search-defaults

const onOff = (v) => (v ? "on" : "off");

function methodOption(m, chosen) {
  return `<option value="${esc(m)}"${m === chosen ? " selected" : ""}>${esc(mayCost(m, cat))}</option>`;
}

function rerankerOption(name, size, chosen, recommended) {
  const notes = [size == null ? "" : `${size} GB`, name === recommended ? "(recommended)" : "",
    name === JINA_TINY ? "about 1.5 s for 20 passages on an M4 Pro" : "",
    name.startsWith("BAAI/bge-reranker-") ? "slower; for a faster machine" : ""].filter(Boolean);
  return `<option value="${esc(name)}"${name === chosen ? " selected" : ""}>${esc([name, ...notes].join(" · "))}</option>`;
}

function searchHtml() {
  const d = defaults;
  const f = d.in_force;
  const rec = d.recommended;
  const range = (field) => `min="${esc(d.limits[field][0])}" max="${esc(d.limits[field][1])}" step="1"`;
  const recommended = (text) => `<span class="muted" data-recommended>Recommended: ${esc(text)}</span>`;
  const number = (field, label, extra = "") =>
    `<label>${esc(label)} <input type="number" name="${field}" ${range(field)} value="${esc(f[field])}">
      ${extra}${recommended(rec[field])}</label>`;
  const check = (field, label, extra = "") =>
    `<label class="check"><input type="checkbox" name="${field}"${f[field] ? " checked" : ""}> ${esc(label)} ${extra}
      ${recommended(onOff(rec[field]))}</label>`;
  const names = Object.fromEntries(d.rerankers.map((r) => [r.name, r.size_gb]));
  // the service's own reranker may not be fastembed's to list; the select still shows it
  for (const name of [rec.rerank_model, f.rerank_model]) if (name && !(name in names)) names[name] = null;
  const models = Object.entries(names).map(([n, size]) => rerankerOption(n, size, f.rerank_model, rec.rerank_model)).join("");
  const embedders = d.embedders
    .map((e) => `<option value="${esc(e.name)}"${e.name === f.dense_model ? " selected" : ""}>${esc(embedderLabel(e, d.embed_machine, rec.dense_model))}</option>`)
    .join("");
  const why = d.notes.rerank ? `<span class="muted" data-rerank-note>${esc(d.notes.rerank)}</span>` : "";
  // The in-force reranker's warm-up, as the service reports it; ready or idle says nothing.
  const loading = d.rerank_status === "loading"
    ? '<span class="muted" data-downloading>Downloading in the background; chat merges by rank until it is ready.</span>'
    : d.rerank_status === "failed"
      ? `<span class="err" data-rerank-error>Could not load it: ${esc(d.rerank_error ?? "unknown error")}. Chat merges by rank; Save tries again.</span>`
      : "";
  return `<label>Embedding model for new stores <select name="dense_model">${embedders}</select>
      ${recommended(rec.dense_model)}
      <span class="muted" data-embedder-note>Existing stores keep theirs; rebuild one from its page.</span></label>
    <label>Default search method <select name="method">${d.methods.map((m) => methodOption(m, f.method)).join("")}</select>
      ${recommended(methodLabel(rec.method, cat))}</label>
    ${number("candidates", "Candidates per store")}
    ${number("evidence_size", "Evidence size")}
    ${check("rerank", "Rerank", why)}
    <label>Reranker model <select name="rerank_model">${models}</select>${loading}
      ${recommended(rec.rerank_model ?? "the service's reranker")}</label>
    <label>Rerank up to <input type="number" name="rerank_pool" ${range("rerank_pool")} value="${esc(f.rerank_pool)}">
      <span class="muted">passages per message</span>${recommended(rec.rerank_pool)}</label>
    ${check("rewrite", "Rewrite follow-ups", `<span class="muted" data-rewrite-note${ai?.resolved?.rewrite ? " hidden" : ""}>(needs a rewrite model)</span>`)}
    ${number("follow_up_turns", "Previous messages joined, without rewrite")}
    ${number("rrf_k", "RRF k", '<span class="muted">(chat and /query; rerank off, and inside hybrid)</span>')}
    <div class="row"><button type="submit" id="search-go">Save search defaults</button>
      <button type="button" id="search-reset" class="secondary">Reset to recommended</button></div>`;
}

function drawSearch() {
  $("search").innerHTML = searchHtml();
}

// The fields whose value differs from what is in force; an emptied number goes back to recommended.
function searchChanges() {
  const form = $("search");
  const changes = {};
  for (const field of FIELDS) {
    const box = form.elements[field];
    const value = box.type === "checkbox" ? box.checked : box.type === "number" ? (box.value === "" ? null : Number(box.value)) : box.value;
    if (value !== defaults.in_force[field] && !(value === null && defaults.saved[field] === null)) {
      // The recommended reranker is saved as null, so it keeps following the recommendation.
      changes[field] = field === "rerank_model" && value === defaults.recommended.rerank_model ? null : value;
    }
    // A reranker that failed to load is sent again as it is, so Save retries it now.
    else if (field === "rerank_model" && defaults.rerank_status === "failed") changes[field] = defaults.saved[field];
  }
  return changes;
}

async function putSearch(body, done) {
  $("search-msg").className = "";
  $("search-msg").textContent = "";
  try {
    defaults = await api("/search-defaults", { method: "PUT", body });
    drawSearch();
    $("search-msg").className = "ok";
    $("search-msg").textContent = done;
  } catch (e) {
    $("search-msg").className = "err";
    $("search-msg").textContent = e.message;
  }
}

$("search").addEventListener("submit", (event) => {
  event.preventDefault();
  const changes = searchChanges();
  if (Object.keys(changes).length) putSearch(changes, "Saved");
});

$("search").addEventListener("click", (event) => {
  if (!event.target.closest("#search-reset")) return;
  putSearch(Object.fromEntries(FIELDS.map((f) => [f, null])), "Back to recommended");
});

async function loadSearch() {
  [cat, defaults] = await Promise.all([catalog(), api("/search-defaults")]);
  drawSearch();
}

// -- page -------------------------------------------------------------------------

async function load() {
  const first = ai === null;
  const [aiRead, catRead] = await Promise.allSettled([api("/ai"), catalog()]);
  if (aiRead.status === "fulfilled") ai = aiRead.value;
  if (catRead.status === "fulfilled") cat = catRead.value;
  const failed = [aiRead, catRead].find((r) => r.status === "rejected");
  if (failed) say($("msg"), failed.reason.message, "err");
  if (!ai) return; // nothing to draw the sections from
  document.querySelector("[data-rewrite-note]")?.toggleAttribute("hidden", Boolean(ai?.resolved?.rewrite));
  drawConnections();
  drawModels();
  drawRoles();
  drawBudget();
  if (first) drawPresets();
}

load().catch((e) => say($("msg"), e.message, "err"));
loadSearch().catch((e) => ($("search-msg").textContent = e.message));
