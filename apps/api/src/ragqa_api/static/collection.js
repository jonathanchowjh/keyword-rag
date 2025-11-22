import { api, catalog, clock, confirmSpend, esc, gapBar, head, methodLabel, pct, pill, pts,
  refreshSpend, shell, verdictText } from "/static/ui.js";

const SLOW_MS = 15000;
const FAST_MS = 3000;
const BUSY = new Set(["loading", "rekeying", "evaluating", "queued"]);
const DATASETS = { mcq: "scienceqa", multihop: "musique", table: "tatqa" };
const GROUPS = [
  ["baseline", "Baseline"],
  ["control", "Controls"],
  ["plain", "Plain retrieval"],
  ["modern", "Modern"],
  ["keywords", "Keywords"],
];
const CONTROLS = ["random_example", "fixed_example"];
const MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5"];

const name = decodeURIComponent(location.pathname.split("/")[2] ?? "");
const enc = encodeURIComponent(name);
const $ = (id) => document.getElementById(id);

let cat = null;
let col = null; // the latest GET /collections/{name}
let runs = [];
let answerSetup = null; // the setup the answer's run recorded, fetched once per run
let answerSetupFor = null;
const painted = {}; // section id -> html last drawn, so a refresh never rebuilds what has not changed
let listsVersion = undefined; // the keyword-lists version the textareas were filled from
let listsDirty = false;

shell(`c/${name}`);

function paint(id, html) {
  if (painted[id] === html) return false;
  painted[id] = html;
  $(id).innerHTML = html;
  return true;
}

const plural = (n, noun) => `${n} ${noun}${n === 1 ? "" : "s"}`;

// -- head and answer -------------------------------------------------------------

// Why the answer's run no longer matches the collection, in words.
function staleNotes() {
  const a = col.answer;
  if (!a || !answerSetup) return [];
  const after = `after run ${a.run_id.slice(0, 4)}`;
  const notes = [];
  if (answerSetup.keyword_version !== col.keyword_version) notes.push(`keyword lists changed ${after}`);
  if (answerSetup.dense_model !== col.dense_model) notes.push(`dense model changed ${after}`);
  return notes;
}

function headHtml() {
  const type = cat.qa_types[col.qa_type];
  const noun = type?.item_noun ?? "item";
  const lists = col.keyword_lists ? `keyword lists ${col.keyword_version}` : "no keyword lists yet";
  const status = [plural(col.count, noun), plural(col.questions, "question"), lists, col.dense_model]
    .map(esc)
    .join(" · ");
  const progress = col.progress != null ? ` ${pct(col.progress)}` : "";
  const stale = staleNotes().map((n) => ` · <span class="warn">${esc(n)}</span>`).join("");
  return head({
    title: name,
    crumb: [{ label: "Collections", href: "/" }],
    sub: `${type?.label ?? col.qa_type}: retrieves ${type?.retrieves ?? "items"}`,
    aside: `${status} · ${pill(col.state)}${esc(progress)}${stale}`,
  });
}

function answerHtml() {
  const a = col.answer;
  if (!a) return '<p class="muted">No answer yet. Run the keyword methods against Hybrid to get one.</p>';
  const span = Math.max(10, Math.ceil(Math.max(Math.abs(a.lo), Math.abs(a.hi), Math.abs(a.diff)) / 5) * 5);
  const label = `${methodLabel(a.candidate, cat)} against ${methodLabel(a.reference, cat)}`;
  return `<p class="verdict">${esc(verdictText(a, cat))}</p>
    <p>${gapBar(a.lo, a.diff, a.hi, span)} <span class="muted">${esc(label)}, ${esc(pts(a.diff, a.lo, a.hi))}</span></p>
    <p class="muted">From <a href="/c/${encodeURIComponent(name)}/runs/${encodeURIComponent(a.run_id)}">run ${esc(
      a.run_id.slice(0, 4))}</a> at ${esc(clock(a.at))}${a.answered ? "" : ", retrieval only"} ·
      run ${esc(a.runs)} on this collection</p>`;
}

// -- runs --------------------------------------------------------------------------

function runsHtml() {
  if (!runs.length) return '<p class="muted">No runs yet.</p>';
  const rows = runs
    .map((r) => {
      const methods = (r.methods ?? []).map((m) => methodLabel(m, cat)).join(", ");
      const progress = r.status === "running" ? ` ${pct(r.progress)}` : "";
      return `<tr><td><a href="/c/${encodeURIComponent(name)}/runs/${encodeURIComponent(r.id)}">${esc(
        clock(r.created_at))}</a></td><td>${esc(methods)}</td><td class="num">${esc(r.question_count)}</td>
        <td>${r.answer ? "Answered" : "Retrieval only"}</td>
        <td class="num">$${Number(r.usd_total ?? 0).toFixed(2)}</td>
        <td>${pill(r.status)}${esc(progress)}</td></tr>`;
    })
    .join("");
  return `<table><thead><tr><th>Time</th><th>Methods</th><th>Questions</th><th>Answers</th>
    <th>Cost</th><th>Status</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function runFormHtml() {
  const groups = GROUPS.map(([id, title]) => {
    const boxes = Object.entries(cat.methods)
      .filter(([, m]) => m.group === id)
      .map(([mid, m]) => {
        const on = cat.answer_pair.includes(mid) || CONTROLS.includes(mid);
        const ok = col.methods_available.includes(mid);
        return `<label class="check"><input type="checkbox" name="method" value="${esc(mid)}"
          ${on && ok ? "checked" : ""} ${ok ? "" : "disabled"}> ${esc(m.label)}<code>${esc(mid)}</code></label>`;
      })
      .join("");
    return boxes ? `<fieldset><legend>${esc(title)}</legend><div class="checks">${boxes}</div></fieldset>` : "";
  }).join("");
  const models = MODELS.map((m) => `<option>${esc(m)}</option>`).join("");
  return `<div class="card form">${groups}
    <label class="check"><input type="checkbox" id="rf-answer" checked> Have Claude answer the questions</label>
    <div class="row"><label>Model<select id="rf-model">${models}</select></label>
      <label>Questions<select id="rf-split">
        <option value="dev">Dev slice (${esc(col.dev_questions)})</option>
        <option value="full">All (${esc(col.questions)})</option></select></label></div>
    <div class="row"><button type="button" id="rf-estimate">Estimate</button>
      <span id="rf-confirm"></span><span id="rf-msg" class="err"></span></div></div>`;
}

function bindRunForm() {
  const form = $("run-form");
  const body = () => ({
    collection: name,
    methods: [...form.querySelectorAll("input[name=method]:checked")].map((i) => i.value),
    answer: $("rf-answer").checked,
    model: $("rf-model").value,
    split: $("rf-split").value,
    k: 5,
    rrf_k: 60,
  });
  const forget = () => ($("rf-confirm").innerHTML = "");
  form.addEventListener("change", forget);
  $("rf-estimate").addEventListener("click", async () => {
    $("rf-msg").textContent = "";
    forget();
    const request = body();
    if (!request.methods.length) {
      $("rf-msg").textContent = "Pick at least one method.";
      return;
    }
    try {
      const { estimate } = await api("/eval", { method: "POST", body: { ...request, confirm: false } });
      confirmSpend($("rf-confirm"), estimate, async () => {
        try {
          await api("/eval", { method: "POST", body: { ...request, confirm: true } });
          form.hidden = true;
          forget();
          refreshSpend();
          await refresh();
        } catch (e) {
          $("rf-msg").textContent = e.message;
        }
      });
    } catch (e) {
      $("rf-msg").textContent = e.message;
    }
  });
}

function toggleRunForm() {
  const form = $("run-form");
  if (!form.hidden) {
    form.hidden = true;
    return;
  }
  form.innerHTML = runFormHtml();
  form.hidden = false;
  bindRunForm();
}

// -- items and questions --------------------------------------------------------------

function itemsHtml() {
  const type = cat.qa_types[col.qa_type];
  const noun = type?.item_noun ?? "item";
  const count = `<p>${esc(plural(col.count, noun))}.</p>`;
  if (col.count > 0) {
    return `${count}<p class="muted">Items are read-only here for now; adding, editing and sources come later.</p>`;
  }
  const dataset = DATASETS[col.qa_type];
  return `${count}<div class="card form">
    <div>Load <code>${esc(dataset)}</code> into this collection.</div>
    <div class="row"><label>Limit (blank for all)<input id="ld-limit" type="number" min="1"></label>
      <button type="button" id="ld-go">Load ${esc(dataset)}</button>
      <span id="ld-confirm"></span><span id="ld-msg" class="err"></span></div></div>`;
}

function bindLoad() {
  const go = $("ld-go");
  if (!go) return;
  const dataset = DATASETS[col.qa_type];
  go.addEventListener("click", async () => {
    $("ld-msg").textContent = "";
    const limit = parseInt($("ld-limit").value, 10);
    const base = { dataset, ...(limit > 0 ? { limit } : {}) };
    try {
      const first = await api("/collections/" + enc + "/load", { method: "POST", body: { ...base, confirm: false } });
      if (!first.estimate) {
        $("ld-confirm").textContent = `Queued (job ${first.job_id.slice(0, 4)})`;
        await refresh();
        return;
      }
      confirmSpend($("ld-confirm"), { ...first.estimate, calls: first.estimate.items }, async () => {
        try {
          const job = await api("/collections/" + enc + "/load", { method: "POST", body: { ...base, confirm: true } });
          $("ld-confirm").textContent = `Queued (job ${job.job_id.slice(0, 4)})`;
          refreshSpend();
          await refresh();
        } catch (e) {
          $("ld-msg").textContent = e.message;
        }
      });
    } catch (e) {
      $("ld-msg").textContent = e.message;
    }
  });
}

function askedHtml() {
  return `<p>${esc(plural(col.questions, "question"))}, of which ${esc(col.dev_questions)} in the dev
    slice that runs use by default.</p>`;
}

// -- keyword lists ----------------------------------------------------------------------

const LIST_KEYS = [
  ["topic", "Topic"],
  ["method", "Method"],
  ["common", "Common"],
];

function buildLists() {
  const areas = LIST_KEYS.map(
    ([k, title]) => `<label>${title}<textarea id="kw-${k}" rows="8" spellcheck="false"></textarea></label>`
  ).join("");
  $("lists").innerHTML = `<p class="muted">One keyword per line. Each item is described by the
    best few from each list.</p><div class="lists">${areas}</div>
    <div class="actions"><button type="button" id="kw-save">Save lists</button>
      <button type="button" id="kw-propose" class="secondary">Propose with Claude</button></div>
    <p id="kw-hint" class="muted"></p><p id="kw-msg" role="status"></p>`;
  for (const [k] of LIST_KEYS) $(`kw-${k}`).addEventListener("input", () => (listsDirty = true));
  $("kw-save").addEventListener("click", saveLists);
  $("kw-propose").addEventListener("click", proposeLists);
}

function fillLists(lists) {
  for (const [k] of LIST_KEYS) $(`kw-${k}`).value = (lists?.[k] ?? []).join("\n");
}

function readLists() {
  const out = {};
  for (const [k] of LIST_KEYS) {
    out[k] = $(`kw-${k}`).value.split("\n").map((s) => s.trim()).filter(Boolean);
  }
  return out;
}

function note(html) {
  $("kw-msg").innerHTML = html;
}

async function saveLists() {
  note("");
  try {
    const saved = await api("/collections/" + enc + "/keyword-lists", { method: "PUT", body: readLists() });
    listsDirty = false;
    listsVersion = col.keyword_version ?? null; // keep the edits until the rekey switches version
    note(`<span class="ok">Saved as version ${esc(saved.version)}: re-keys ${esc(
      plural(col.count, "item"))}.</span> Job ${esc(
      saved.job_id.slice(0, 4))} ${pill("queued")}`);
    await refresh();
  } catch (e) {
    note(`<span class="err">${esc(e.message)}</span>`);
  }
}

async function proposeLists() {
  note("");
  const path = "/collections/" + enc + "/keyword-lists/propose";
  try {
    const { estimate } = await api(path, { method: "POST", body: { confirm: false } });
    const slot = document.createElement("span");
    $("kw-msg").replaceChildren("Proposing costs ", slot);
    confirmSpend(slot, estimate, async () => {
      try {
        fillLists(await api(path, { method: "POST", body: { confirm: true } }));
        listsDirty = true;
        note("Proposed by Claude. Review them, then Save lists.");
        refreshSpend();
      } catch (e) {
        note(`<span class="err">${esc(e.message)}</span>`);
      }
    });
  } catch (e) {
    note(`<span class="err">${esc(e.message)}</span>`);
  }
}

function syncLists() {
  $("kw-hint").textContent = `Saving re-keys ${plural(col.count, "item")}, free.`;
  const version = col.keyword_version ?? null;
  if (version !== listsVersion && !listsDirty) {
    fillLists(col.keyword_lists);
    listsVersion = version;
  }
}

// -- delete ---------------------------------------------------------------------------

function showDelete(asking = false, message = "") {
  $("delete").innerHTML = asking
    ? `<p>Delete ${esc(name)}, with its items, questions, lists and runs? This cannot be undone.</p>
       <div class="actions"><button type="button" id="del-yes" class="danger">Yes, delete ${esc(name)}</button>
       <button type="button" id="del-no" class="secondary">Keep it</button></div>`
    : `<div class="actions"><button type="button" id="del-ask" class="secondary">Delete…</button></div>
       <p class="err">${esc(message)}</p>`;
  if (asking) {
    $("del-no").addEventListener("click", () => showDelete());
    $("del-yes").addEventListener("click", async () => {
      try {
        await api("/collections/" + enc, { method: "DELETE" });
        location.href = "/";
      } catch (e) {
        showDelete(false, e.message);
      }
    });
  } else {
    $("del-ask").addEventListener("click", () => showDelete(true));
  }
}

// -- refresh ----------------------------------------------------------------------------

let timer = null;
async function refresh() {
  clearTimeout(timer);
  try {
    cat = await catalog();
    [col, runs] = await Promise.all([api("/collections/" + enc), api("/eval?collection=" + enc)]);
    if (col.answer && answerSetupFor !== col.answer.run_id) {
      answerSetup = (await api("/eval/" + encodeURIComponent(col.answer.run_id))).setup;
      answerSetupFor = col.answer.run_id;
    }
    $("head").innerHTML = headHtml();
    paint("answer", answerHtml());
    paint("runs", runsHtml());
    paint("asked", askedHtml());
    if (paint("items", itemsHtml())) bindLoad();
    syncLists();
  } catch (e) {
    $("head").innerHTML = head({ title: name, crumb: [{ label: "Collections", href: "/" }] }) +
      `<p class="err">${esc(e.message)}</p>`;
  }
  timer = setTimeout(refresh, col && BUSY.has(col.state) ? FAST_MS : SLOW_MS);
}

$("new-run").addEventListener("click", toggleRunForm);
buildLists();
showDelete();
refresh();
