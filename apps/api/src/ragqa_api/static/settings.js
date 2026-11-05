import { $, api, esc, head, modelOptions, refreshSpend, shell, usd } from "/static/ui.js";

shell("settings");
$("head").innerHTML = head({
  title: "Settings",
  sub: "Where the AI comes from, what each model costs, and which model does which job.",
});

const ENV = "set by environment";

let ai = null; // the last GET /ai
const notes = {}; // connection id -> what its last button said, kept across redraws

// A model id may contain "/"; each segment is encoded on its own.
const modelPath = (m) => `/ai/models/${encodeURIComponent(m.connection_id)}/${m.model_id.split("/").map(encodeURIComponent).join("/")}`;
const connPath = (id) => `/ai/connections/${encodeURIComponent(id)}`;

const say = (text) => ($("msg").textContent = text);

// Runs one change, then redraws from the server; a refusal shows its reason.
async function act(change) {
  say("");
  try {
    await change();
  } catch (e) {
    say(e.message);
  }
  await load();
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
    ? '<span class="muted">prompts wait for replies: <code>python -m ragqa_core.replies</code></span>'
    : `<span class="muted">key ${esc(key)}</span>`;
  return `<div class="card conn" data-connection="${esc(c.id)}">
    <div class="top"><strong>${esc(c.label)}</strong><code>${esc(c.kind)}</code>${where}
      ${status}</div>
    ${keyBox}
    <div class="actions" style="margin:0">
      ${manual ? "" : `<button type="button" class="secondary" data-act="test">Test</button>
      <button type="button" class="secondary" data-act="fetch">Fetch models</button>`}
      ${c.from_env.length ? "" : '<button type="button" class="danger" data-act="remove">Remove</button>'}
      <span class="result" role="status">${notes[c.id] ?? ""}</span></div>
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
  async remove(id) {
    await api(connPath(id), { method: "DELETE" });
    delete notes[id];
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
  act(() => ACTIONS[button.dataset.act](box.dataset.connection, box));
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
    await api("/ai/connections", { method: "POST", body });
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
const NONE = "none";
const priceKey = (i, o) => `${i}/${o}`;
function priceOptions() {
  const listed = (ai.prices || []).map((p) => ({ value: priceKey(p.price_in, p.price_out),
    label: `${p.label} rates: $${p.price_in} in, $${p.price_out} out` }));
  return [{ value: priceKey(0, 0), label: "$0 (free)" }, ...listed,
    { value: NONE, label: "No price (can't be used)" }, { value: HAND, label: "Enter by hand…" }];
}

function priceChoice(m) {
  if (m.price_in == null || m.price_out == null) return NONE;
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
    <button type="button" class="secondary" data-save>Save</button>
    ${note}</div>`;
}

function drawModels() {
  $("models").innerHTML = ai.connections
    .map((c) => {
      const rows = ai.models.filter((m) => m.connection_id === c.id).map(modelHtml).join("");
      return `<div class="card" data-connection="${esc(c.id)}"><strong>${esc(c.label)}</strong>
        ${rows || '<p class="muted">No models. Fetch the list, or add one by id.</p>'}
        <div class="row" style="margin-top:8px"><label>Add a model by id <input name="new-id"></label>
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
  const [chosenIn, chosenOut] = choice === NONE ? [null, null] : choice.split("/").map(Number);
  const typed = choice === HAND;
  const body = { enabled: row.querySelector("[name=enabled]").checked,
    price_in: typed ? number("price_in") : chosenIn, price_out: typed ? number("price_out") : chosenOut };
  say("");
  if (typed && (body.price_in === null || body.price_out === null)) {
    say("Give both prices, or choose one from the list.");
    return;
  }
  try {
    await api(modelPath({ connection_id: row.dataset.connection, model_id: row.dataset.id }), { method: "PUT", body });
    ai = await api("/ai");
    row.outerHTML = modelHtml(ai.models.find((m) => m.ref === row.dataset.model));
    drawRoles();
  } catch (e) {
    say(e.message);
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
  act(() => api(connPath(card.dataset.connection) + "/models", { method: "POST", body: { model_id: id } }));
});

// -- jobs -------------------------------------------------------------------------

const JOB_NOTE = { default: "Used by any job without its own choice" };

// A job's model that can no longer be used still shows, marked and selected, so the
// choice is visible and picking anything else (even "same as default") changes it.
function unusableOption(ref) {
  if (!ref) return "";
  const m = ai.models.find((x) => x.ref === ref);
  const why = !m ? "not set up" : !m.enabled ? "disabled" : m.price_in == null || m.price_out == null ? "needs a price" : "";
  return why ? `<option value="${esc(ref)}" selected>${esc(m?.label ?? ref)} (${why})</option>` : "";
}

function drawRoles() {
  $("roles").innerHTML = ai.known_roles
    .map((role) => {
      const first = role === "default" ? "none chosen" : "same as default";
      const locked = role === "default" && ai.default_from_env;
      const note = [JOB_NOTE[role], locked ? ENV : ""].filter(Boolean).join("; ");
      return `<label>${esc(role)} <select data-role="${esc(role)}"${locked ? " disabled" : ""}><option value="">${first}</option>
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

// -- page -------------------------------------------------------------------------

async function load() {
  const first = ai === null;
  ai = await api("/ai");
  drawConnections();
  drawModels();
  drawRoles();
  drawBudget();
  if (first) drawPresets();
}

load().catch((e) => say(e.message));
