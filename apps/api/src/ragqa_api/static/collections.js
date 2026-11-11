import { $, LEGEND, api, catalog, clock, confirmSpend, dropStaleStart, esc, head, methodLabel, nameField, paint, pct, pill, plural,
  noScoreLine, poll, refreshSpend, say, scoreLine, shell, short, verdictHtml, watchName } from "/static/ui.js";

const REFRESH_MS = 15000;
const BENCHMARKS = [
  { name: "sqa", qa_type: "mcq", summarize: false, dataset: "scienceqa" },
  { name: "mq", qa_type: "multihop", summarize: false, dataset: "musique" },
  { name: "tq", qa_type: "table", summarize: false, dataset: "tatqa" },
  { name: "sb", qa_type: "science", summarize: false, dataset: "scibench" },
];

let cat = null;
let lastList = "";
let startHidden = false;
let startDraw = 0; // the newest draw's number: an older, slower draw never paints
const startDone = new Set(); // steps found done: kept for the page's life, not fetched again

shell("collections");
$("head").innerHTML = head({
  title: "Stores",
  sub: "Each store holds the documents your chats search and your runs test.",
});

// -- the list ------------------------------------------------------------------

function rowHtml(c) {
  const type = cat.qa_types[c.qa_type];
  const noun = type?.item_noun ?? "item";
  const counts = `${plural(c.count, noun)}, ${plural(c.questions, "question")}`;
  const a = c.answer;
  const pair = cat.answer_pair.map((m) => methodLabel(m, cat)).join(" and ");
  const docs = c.qa_type === "documents";
  const said = () => (docs ? esc(scoreLine(a, cat)) : `Do keywords help? ${verdictHtml(a, c.qa_type, cat)}`);
  const answer = a
    ? `<div title="${esc(LEGEND)}">${said()}</div><div class="small">run ${esc(short(a.run_id))} · ${esc(
        clock(a.at))}${a.answered ? "" : " · retrieval only"}</div>`
    : docs
    ? `<span class="muted">${esc(noScoreLine(c, cat))}</span>`
    : `<span class="muted">No answer yet</span><div class="small">${esc(
        c.runs ? `${plural(c.runs, "run")}, none finished with ${pair}` : `no runs yet: run ${pair} to get one`)}</div>`;
  const progress = c.state !== "ready" && c.progress != null ? ` ${pct(c.progress)}` : "";
  return `<a class="row-link" href="/c/${encodeURIComponent(c.name)}">
    <div><strong>${esc(c.name)}</strong> · ${esc(type?.label ?? c.qa_type)}
      <div class="small">${esc(counts)}</div>${c.load_error
        ? `<div class="small err">${esc("Last load stopped: " + c.load_error)}</div>` : ""}</div>
    <div>${answer}</div>
    <div>${pill(c.state)}${esc(progress)}</div></a>`;
}

async function refresh() {
  try {
    const [specs] = await Promise.all([api("/collections"), catalog().then((c) => (cat = c))]);
    const full = await Promise.all(specs.map((s) => api(`/collections/${encodeURIComponent(s.name)}`)));
    drawDefaults(new Set(specs.map((s) => s.name)));
    const html = full.length
      ? `<div class="table"><div class="th"><div>Store</div><div>Result</div>
          <div>Ready?</div></div>${full.map(rowHtml).join("")}</div>`
      : '<p class="muted">No stores yet. Make one from your documents below.</p>';
    if (html !== lastList) {
      $("list").innerHTML = html;
      lastList = html;
    }
    drawStart(full);
  } catch (e) {
    $("list").innerHTML = `<p class="err">${esc(e.message)}</p>`;
    lastList = "";
  }
}

// -- start here -----------------------------------------------------------------
// Four steps worked out from what exists; nothing about them is stored.

const attempt = async (path, usable) => {
  try {
    return usable(await api(path));
  } catch {
    return false; // a failed request leaves its step not done for this draw
  }
};

function aiUsable(ai) {
  return ai.models.some((m) => m.ready);
}

async function startSteps(full) {
  const steps = [
    { key: "ai", label: "Connect AI", href: "/settings", optional: true,
      check: () => attempt("/ai", aiUsable) },
    { key: "store", label: "Pick or make a store", href: "#new", optional: false,
      check: async () => full.some((c) => c.count > 0) },
    { key: "chat", label: "Chat", href: "/chat", optional: false,
      check: () => attempt("/chats", (rows) => rows.some((r) => r.messages > 0)) },
    { key: "run", label: "Test with an eval set", href: "/eval-sets", optional: false,
      check: () => attempt("/jobs?kind=eval&status=done&limit=1", (rows) => rows.length > 0) },
  ];
  const done = await Promise.all(
    steps.map((s) => (startDone.has(s.key) ? true : s.check())));
  return steps.map(({ check, ...s }, i) => {
    if (done[i]) startDone.add(s.key);
    return { ...s, done: done[i] };
  });
}

function startHtml(steps) {
  if (steps.every((s) => s.optional || s.done)) return "";
  const items = steps.map((s) => `<li data-step="${esc(s.key)}" data-done="${s.done}">
    <a href="${esc(s.href)}">${esc(s.label)}</a> <span class="muted">${s.done ? "done" : "not done"}</span>${s.key === "ai"
      ? ' <span class="muted">Optional for Documents: without it, chat gives passages only.</span>' : ""}</li>`);
  return `<section class="card" id="start-here"><h2>Start here</h2><ol class="start-steps">${items.join("")}</ol></section>`;
}

async function drawStart(full) {
  if (startHidden) return;
  const mine = ++startDraw;
  try {
    const html = startHtml(await startSteps(full));
    if (mine !== startDraw || startHidden) return;
    paint("start", html);
    startHidden = html === "";
  } catch {
    // the strip is a guide: it never breaks the list
  }
}

$("start").addEventListener("click", (e) => {
  if (e.target.closest('a[href="#new"]')) {
    e.preventDefault();
    showForm();
  }
});

const capital = (text) => text.charAt(0).toUpperCase() + text.slice(1);

// -- new store ------------------------------------------------------------

async function showForm() {
  let embedded = "";
  try {
    const d = await api("/search-defaults");
    embedded = `Embedded with ${d.in_force.dense_model} (Settings → Search defaults)`;
  } catch {
    embedded = "";
  }
  cat = await catalog();
  const types = Object.entries(cat.qa_types)
    .map(([id, t]) => `<option value="${esc(id)}">${esc(t.label)}: retrieves ${esc(t.retrieves)}</option>`)
    .join("");
  $("form").hidden = false;
  $("form").innerHTML = `<div class="card form">
    ${nameField()}
    <label>Type<select id="f-type">${types}</select></label>
    <label>Kind<select id="f-recipe"></select></label>
    <p class="muted" data-embedded-with>${esc(embedded)}</p>
    <label class="check" id="f-sum-row"><input type="checkbox" id="f-sum"> Summarise each item with AI <span class="cost muted">costs money</span></label>
    <div class="row"><button type="button" id="f-create">Create</button>
      <button type="button" id="f-cancel" class="secondary">Cancel</button>
      <span id="f-msg" class="err" role="status"></span></div></div>`;
  watchName();
  showType();
  $("f-type").addEventListener("change", showType);
  $("f-cancel").addEventListener("click", () => ($("form").hidden = true));
  $("f-create").addEventListener("click", create);
  $("f-name").focus(); // scrolls the form, below the list, into view
}

// The selected type's own recipes, paper 2024 first where it is one. documents_tagged is
// reached by adding keyword tags to a Documents store, never chosen here. Summaries feed the
// keyword lists, which a Documents store has none of: the switch is hidden and off there.
function showType() {
  const type = $("f-type").value;
  $("f-sum-row").hidden = type === "documents";
  if (type === "documents") $("f-sum").checked = false;
  const own = cat.type_recipes[type] ?? [];
  $("f-recipe").innerHTML = own
    .filter((id) => id !== "documents_tagged")
    .sort((a, b) => (b === "paper_2024") - (a === "paper_2024"))
    .map((id) => `<option value="${esc(id)}">${esc(capital(cat.recipes[id].label))}: ${esc(cat.recipes[id].says)}</option>`)
    .join("");
}

// Create stays off while its POST is out, so a second click makes nothing twice.
async function create() {
  const name = $("f-name").value.trim();
  $("f-create").disabled = true;
  try {
    await api("/collections", {
      method: "POST",
      body: { name, qa_type: $("f-type").value, summarize: $("f-sum").checked, recipe: $("f-recipe").value },
    });
    location.href = "/c/" + encodeURIComponent(name); // leaves the page: Create stays off
  } catch (e) {
    $("f-create").disabled = false;
    say($("f-msg"), e.message);
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
        // What was shown: a price that changed since is refused, to be checked again.
        const shown = { calls: first.estimate.calls, est_usd: first.estimate.est_usd };
        const job = await api(path, { method: "POST", body: { dataset, ...shown, confirm: true } });
        $("defaults-confirm").textContent = `Queued (job ${short(job.job_id)})`;
        refreshSpend();
        await refresh();
      } catch (e) {
        dropStaleStart(slot, e);
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
  // One load that can't be priced leaves the total unpriced, never a low sum.
  const unpriced = paying.some((p) => p.estimate.est_usd == null && Number(p.estimate.calls ?? 0) > 0);
  const names = paying.map((p) => p.b.name).join(", ");
  const li = step(`Loading ${names} costs `);
  const slot = document.createElement("span");
  li.append(slot);
  const total = { est_usd: unpriced ? null : sum("est_usd"), calls: sum("calls") };
  confirmSpend(slot, total, async () => {
    slot.textContent = "starting…";
    for (const { b, estimate } of paying) {
      try {
        // Each load's own shown price: one that changed since is refused, to be checked again.
        const shown = { calls: estimate.calls, est_usd: estimate.est_usd };
        const job = await api(`/collections/${b.name}/load`, {
          method: "POST",
          body: { dataset: b.dataset, ...shown, confirm: true },
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

// Back from the page Create opened: the browser may restore this one with Create still off.
window.addEventListener("pageshow", (e) => {
  if (e.persisted && $("f-create")) $("f-create").disabled = false;
});
$("new").addEventListener("click", showForm);
$("setup").addEventListener("click", setup);
refresh();
poll(refresh, REFRESH_MS);
