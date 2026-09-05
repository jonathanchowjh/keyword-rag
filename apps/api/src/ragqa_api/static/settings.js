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
  const locked = c.from_env.includes("api_key");
  const key = locked ? ENV : c.key ?? "no key";
  const where = c.base_url ? `<code>${esc(c.base_url)}</code>` : "";
  const keyBox = c.local
    ? ""
    : `<div class="keyline"><label>New key <input type="password" name="key" autocomplete="off"
        ${locked ? `disabled placeholder="${ENV}"` : ""}></label>
        <button type="button" class="secondary" data-act="key" ${locked ? "disabled" : ""}>Save key</button></div>`;
  return `<div class="card conn" data-connection="${esc(c.id)}">
    <div class="top"><strong>${esc(c.label)}</strong><code>${esc(c.kind)}</code>${where}
      <span class="muted">${c.local ? "local" : `key ${esc(key)}`}</span></div>
    ${keyBox}
    <div class="actions" style="margin:0">
      <button type="button" class="secondary" data-act="test">Test</button>
      <button type="button" class="secondary" data-act="fetch">Fetch models</button>
      <button type="button" class="danger" data-act="remove">Remove</button>
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
  $("add-key").placeholder = p.local ? "optional for a local server" : "";
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

function modelHtml(m) {
  const unpriced = m.price_in == null || m.price_out == null;
  return `<div class="model" data-model="${esc(m.ref)}" data-connection="${esc(m.connection_id)}" data-id="${esc(m.model_id)}">
    <span class="name">${esc(m.label)}<code>${esc(m.model_id)}</code></span>
    <label><input type="checkbox" name="enabled" ${m.enabled ? "checked" : ""}> Enabled</label>
    ${priceBox("price_in", m.price_in)}${priceBox("price_out", m.price_out)}
    ${unpriced ? '<span class="warn">needs a price before use</span>' : ""}</div>`;
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

$("models").addEventListener("change", (event) => {
  const row = event.target.closest(".model");
  if (!row) return;
  const number = (name) => {
    const v = row.querySelector(`[name=${name}]`).value;
    return v === "" ? null : Number(v);
  };
  const m = { connection_id: row.dataset.connection, model_id: row.dataset.id };
  const body = { enabled: row.querySelector("[name=enabled]").checked, price_in: number("price_in"), price_out: number("price_out") };
  act(() => api(modelPath(m), { method: "PUT", body }));
});

$("models").addEventListener("click", (event) => {
  if (!event.target.closest("[data-add]")) return;
  const card = event.target.closest("[data-connection]");
  const id = card.querySelector("[name=new-id]").value.trim();
  if (!id) return;
  act(() => api(connPath(card.dataset.connection) + "/models", { method: "POST", body: { model_id: id } }));
});

// -- jobs -------------------------------------------------------------------------

const JOB_NOTE = { default: "Used by any job without its own choice" };

function drawRoles() {
  $("roles").innerHTML = ai.known_roles
    .map((role) => {
      const first = role === "default" ? "none chosen" : "same as default";
      return `<label>${esc(role)} <select data-role="${esc(role)}"><option value="">${first}</option>
        ${modelOptions(ai.models, ai.roles[role])}</select>
        ${JOB_NOTE[role] ? `<span class="muted">${esc(JOB_NOTE[role])}</span>` : ""}</label>`;
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
