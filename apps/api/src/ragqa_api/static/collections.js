import { $, LEGEND, api, catalog, clock, confirmSpend, esc, head, methodLabel, pct, pill, plural,
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
  title: "Collections",
  sub: "Each question bank, and whether keyword retrieval helps on it.",
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
    const html = full.length
      ? `<div class="table"><div class="th"><div>Collection</div><div>Do keywords help?</div>
          <div>Ready?</div></div>${full.map(rowHtml).join("")}</div>`
      : '<p class="muted">No collections yet. Create one, or set up the four benchmarks.</p>';
    if (html !== lastList) {
      $("list").innerHTML = html;
      lastList = html;
    }
  } catch (e) {
    $("list").innerHTML = `<p class="err">${esc(e.message)}</p>`;
    lastList = "";
  }
}

// -- new collection ------------------------------------------------------------

async function showForm() {
  cat = await catalog();
  const types = Object.entries(cat.qa_types)
    .map(([id, t]) => `<option value="${esc(id)}">${esc(t.label)}: retrieves ${esc(t.retrieves)}</option>`)
    .join("");
  $("form").hidden = false;
  $("form").innerHTML = `<div class="card form">
    <label>Name<input id="f-name" autocomplete="off"></label>
    <label>Type<select id="f-type">${types}</select></label>
    <label class="check"><input type="checkbox" id="f-sum"> Summarise each item with Claude <span class="credit muted">uses credit</span></label>
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
      body: { name, qa_type: $("f-type").value, summarize: $("f-sum").checked },
    });
    $("form").hidden = true;
    await refresh();
  } catch (e) {
    $("f-msg").textContent = e.message;
  }
}

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
