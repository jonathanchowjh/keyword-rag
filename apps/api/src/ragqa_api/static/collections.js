import { $, LEGEND, api, catalog, clock, confirmSpend, esc, head, methodLabel, paint, pct, pill, plural,
  refreshSpend, shell, short, verdictHtml } from "/static/ui.js";

const REFRESH_MS = 15000;
const BENCHMARKS = [
  { name: "sqa", qa_type: "mcq", summarize: false, dataset: "scienceqa" },
  { name: "mq", qa_type: "multihop", summarize: false, dataset: "musique" },
  { name: "tq", qa_type: "table", summarize: false, dataset: "tatqa" },
  { name: "sb", qa_type: "science", summarize: false, dataset: "scibench" },
];

let cat = null;
let lastList = "";

shell("collections");
$("head").innerHTML = head({
  title: "Stores",
  sub: "What questions are answered from, and whether keyword retrieval helps on it.",
});

// -- the list ------------------------------------------------------------------

function rowHtml(c) {
  const type = cat.qa_types[c.qa_type];
  const noun = type?.item_noun ?? "item";
  const counts = `${plural(c.count, noun)}, ${plural(c.questions, "question")}`;
  const a = c.answer;
  const pair = cat.answer_pair.map((m) => methodLabel(m, cat)).join(" and ");
  const answer = a
    ? `<div title="${esc(LEGEND)}">${verdictHtml(a, c.qa_type, cat)}</div><div class="small">run ${esc(short(a.run_id))} · ${esc(
        clock(a.at))}${a.answered ? "" : " · retrieval only"}</div>`
    : `<span class="muted">No answer yet</span><div class="small">${esc(
        c.runs ? `${plural(c.runs, "run")}, none finished with ${pair}` : `no runs yet: run ${pair} to get one`)}</div>`;
  const progress = c.state !== "ready" && c.progress != null ? ` ${pct(c.progress)}` : "";
  return `<a class="row-link" href="/c/${encodeURIComponent(c.name)}">
    <div><strong>${esc(c.name)}</strong> · ${esc(type?.label ?? c.qa_type)}
      <div class="small">${esc(counts)}</div></div>
    <div>${answer}</div>
    <div>${pill(c.state)}${esc(progress)}</div></a>`;
}

async function refresh() {
  try {
    const [specs] = await Promise.all([api("/collections"), catalog().then((c) => (cat = c))]);
    const full = await Promise.all(specs.map((s) => api(`/collections/${encodeURIComponent(s.name)}`)));
    drawDefaults(new Set(specs.map((s) => s.name)));
    const html = full.length
      ? `<div class="table"><div class="th"><div>Store</div><div>Do keywords help?</div>
          <div>Ready?</div></div>${full.map(rowHtml).join("")}</div>`
      : '<p class="muted">No stores yet. Create one, or set up the four benchmarks.</p>';
    if (html !== lastList) {
      $("list").innerHTML = html;
      lastList = html;
    }
  } catch (e) {
    $("list").innerHTML = `<p class="err">${esc(e.message)}</p>`;
    lastList = "";
  }
}

const capital = (text) => text.charAt(0).toUpperCase() + text.slice(1);

// -- new store ------------------------------------------------------------

async function showForm() {
  cat = await catalog();
  const types = Object.entries(cat.qa_types)
    .map(([id, t]) => `<option value="${esc(id)}">${esc(t.label)}: retrieves ${esc(t.retrieves)}</option>`)
    .join("");
  const recipes = Object.entries(cat.recipes)
    .sort(([a], [b]) => (b === "paper_2024") - (a === "paper_2024"))
    .map(([id, r]) => `<option value="${esc(id)}">${esc(capital(r.label))}: ${esc(r.says)}</option>`)
    .join("");
  $("form").hidden = false;
  $("form").innerHTML = `<div class="card form">
    <label>Name<input id="f-name" autocomplete="off"></label>
    <label>Type<select id="f-type">${types}</select></label>
    <label>Kind<select id="f-recipe">${recipes}</select></label>
    <label class="check"><input type="checkbox" id="f-sum"> Summarise each item with AI <span class="cost muted">costs money</span></label>
    <div class="row"><button type="button" id="f-create">Create</button>
      <button type="button" id="f-cancel" class="secondary">Cancel</button>
      <span id="f-msg" class="err" role="status"></span></div></div>`;
  $("f-cancel").addEventListener("click", () => ($("form").hidden = true));
  $("f-create").addEventListener("click", create);
}

async function create() {
  const name = $("f-name").value.trim();
  try {
    await api("/collections", {
      method: "POST",
      body: { name, qa_type: $("f-type").value, summarize: $("f-sum").checked, recipe: $("f-recipe").value },
    });
    $("form").hidden = true;
    await refresh();
  } catch (e) {
    $("f-msg").textContent = e.message;
  }
}

// -- a default store ----------------------------------------------------------------------------

// A dataset whose store exists links to it: adding it again would load it again.
function drawDefaults(stores) {
  paint("defaults", Object.entries(cat.datasets)
    .map(([id, type]) => `<div class="row">${stores.has(id)
      ? `<a href="/c/${encodeURIComponent(id)}">Open ${esc(id)}</a>`
      : `<button type="button" class="secondary" data-dataset="${esc(id)}">Add ${esc(id)}</button>`}
      <span class="muted">${esc(cat.qa_types[type]?.label ?? type)}</span></div>`)
    .join(""));
}

// Creates the store named after the dataset, then asks to load it: the estimate first, and a
// free load (a Hub build applies) queues at once.
async function addDefault(dataset) {
  const type = cat.datasets[dataset];
  $("defaults-msg").textContent = "";
  $("defaults-confirm").textContent = "";
  try {
    const existing = (await api("/collections")).some((c) => c.name === dataset);
    if (!existing) await api("/collections", { method: "POST", body: { name: dataset, qa_type: type, summarize: false } });
    await refresh();
    const path = `/collections/${encodeURIComponent(dataset)}/load`;
    const first = await api(path, { method: "POST", body: { dataset, confirm: false } });
    if (!first.estimate) {
      $("defaults-confirm").textContent = `Queued (job ${short(first.job_id)})`;
      return;
    }
    const slot = document.createElement("span");
    const free = first.estimate.est_usd === 0 && first.estimate.hub?.entry;
    $("defaults-confirm").replaceChildren(free ? "Free: built from the Hub. " : "", slot);
    confirmSpend(slot, first.estimate, async () => {
      try {
        const job = await api(path, { method: "POST", body: { dataset, confirm: true } });
        $("defaults-confirm").textContent = `Queued (job ${short(job.job_id)})`;
        refreshSpend();
        await refresh();
      } catch (e) {
        $("defaults-msg").textContent = e.message;
      }
    });
  } catch (e) {
    $("defaults-msg").textContent = e.message;
  }
}

$("defaults").addEventListener("click", (e) => {
  const button = e.target.closest("button[data-dataset]");
  if (button) addDefault(button.dataset.dataset);
});

// -- the four benchmarks ------------------------------------------------------

function step(text, cls = "") {
  const li = document.createElement("li");
  li.className = cls;
  li.textContent = text;
  $("steps").querySelector("ol").append(li);
  return li;
}

async function setup() {
  $("steps").innerHTML = '<ol class="steps"></ol>';
  $("setup").disabled = true;
  const paying = []; // loads whose first answer was an estimate: {b, estimate}
  try {
    const existing = new Set((await api("/collections")).map((c) => c.name));
    for (const b of BENCHMARKS) {
      if (existing.has(b.name)) {
        step(`${b.name} exists, skipped`, "muted");
        continue;
      }
      try {
        await api("/collections", {
          method: "POST",
          body: { name: b.name, qa_type: b.qa_type, summarize: b.summarize },
        });
        step(`${b.name} created`, "ok");
        const first = await api(`/collections/${b.name}/load`, { method: "POST", body: { dataset: b.dataset } });
        if (first.estimate) paying.push({ b, estimate: first.estimate });
        else step(`${b.name}: loading ${b.dataset} (queued, job ${short(first.job_id)})`, "ok");
      } catch (e) {
        step(`${b.name}: ${e.message}`, "err");
      }
    }
    if (paying.length) confirmLoads(paying);
  } finally {
    $("setup").disabled = false;
    await refresh();
  }
}

// One summed estimate and one Start for every load that would spend; loads that were
// queued free are already running and are not posted again.
function confirmLoads(paying) {
  const sum = (key) => paying.reduce((total, p) => total + Number(p.estimate[key] ?? 0), 0);
  const names = paying.map((p) => p.b.name).join(", ");
  const li = step(`Loading ${names} costs `);
  const slot = document.createElement("span");
  li.append(slot);
  confirmSpend(slot, { est_usd: sum("est_usd"), calls: sum("calls") }, async () => {
    slot.textContent = "starting…";
    for (const { b } of paying) {
      try {
        const job = await api(`/collections/${b.name}/load`, {
          method: "POST",
          body: { dataset: b.dataset, confirm: true },
        });
        step(`${b.name}: loading ${b.dataset} (queued, job ${short(job.job_id)})`, "ok");
      } catch (e) {
        step(`${b.name}: ${e.message}`, "err");
      }
    }
    slot.textContent = "confirmed";
    refreshSpend();
    refresh();
  });
}

$("new").addEventListener("click", showForm);
$("setup").addEventListener("click", setup);
refresh();
setInterval(refresh, REFRESH_MS);
