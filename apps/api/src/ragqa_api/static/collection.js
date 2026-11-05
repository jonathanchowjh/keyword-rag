import { $, BUSY, api, askDelete, axisHtml, catalog, clock, confirmSpend, count, costMark, esc, gapBar, gapSpan,
  head, legendHtml, methodLabel, metricLabel, modelOptions, paint, pct, pill, plural, pts, refreshSpend, shell, short, UNPRICED, usd, verdictHtml,
  importBox, verdictNotes, wordHtml } from "/static/ui.js";

const SLOW_MS = 15000;
const FAST_MS = 3000;
const BUSY_STATES = new Set(["loading", "rekeying", "retagging", "copying", "evaluating", "queued", "importing", "labelling"]); // the collection's own
const DATASETS = { science: "scibench", mcq: "scienceqa", multihop: "musique", table: "tatqa" };
const GROUPS = [
  ["baseline", "Baseline"],
  ["control", "Controls"],
  ["plain", "Plain retrieval"],
  ["modern", "Modern"],
  ["keywords", "Keywords"],
];
const CONTROLS = ["random_example", "fixed_example"];

const name = decodeURIComponent(location.pathname.split("/")[2] ?? "");
const enc = encodeURIComponent(name);

let cat = null;
let col = null; // the latest GET /collections/{name}
let runs = [];
let versions = []; // GET /collections/{name}/keyword-lists, newest first
let loadJob = null; // the newest load job, or null
let listsVersion = undefined; // the keyword-lists version the textareas were filled from
let listsDirty = false;

shell(`c/${name}`);

// -- head and answer -------------------------------------------------------------

// Why the answer's run no longer matches the collection, in words.
function staleNotes() {
  const a = col.answer;
  // A run that recorded no versions (migrated) has nothing to compare against: no note.
  return a ? a.since.changes.map((c) => `${c} changed after run ${short(a.run_id)}`) : [];
}

function headHtml() {
  const type = cat.qa_types[col.qa_type];
  const noun = type?.item_noun ?? "item";
  const lists = col.keyword_lists ? `keyword lists ${short(col.keyword_version)}` : "no keyword lists yet";
  const status = [plural(col.count, noun), plural(col.questions, "question"), lists, col.dense_model]
    .map(esc)
    .join(" · ");
  const progress = col.progress != null ? ` ${pct(col.progress)}` : "";
  const stale = staleNotes().map((n) => ` · <span class="warn">${esc(n)}</span>`).join("");
  return head({
    title: name,
    crumb: [{ label: "Stores", href: "/" }],
    sub: `${type?.label ?? col.qa_type}: retrieves ${type?.retrieves ?? "items"}`,
    aside: `${status} · ${pill(col.state)}${esc(progress)}${stale}`,
  });
}

function answerHtml() {
  const a = col.answer;
  if (!a) return '<p class="muted">No answer yet. Run the keyword methods against Hybrid to get one.</p>';
  const span = gapSpan([a.lo, a.hi, a.diff]);
  return `<p class="verdict">${verdictHtml(a, col.qa_type, cat)}</p>
    <div>${gapBar(a.lo, a.diff, a.hi, span)}${axisHtml(span)}</div>
    ${legendHtml()}
    <p class="muted">From <a href="/c/${enc}/runs/${encodeURIComponent(a.run_id)}">run ${esc(
      short(a.run_id))}</a> at ${esc(clock(a.at))}${a.answered ? "" : ", retrieval only"} ·
      run ${esc(a.runs)} on this store</p>`;
}

// -- runs --------------------------------------------------------------------------

// What a finished run said about the declared pair: "Yes +6 pts (2–10)", or "–" without it.
function saidHtml(r) {
  const a = r.answer_summary;
  if (!a || BUSY.has(r.status)) return "–";
  const notes = verdictNotes(a).map((n) => ` · ${n}`).join("");
  return `${wordHtml(a)} ${esc(pts(a.diff, a.lo, a.hi))}${notes ? `<span class="muted">${esc(notes)}</span>` : ""}`;
}

function runsHtml() {
  if (!runs.length) return '<p class="muted">No runs yet.</p>';
  const rows = runs
    .map((r) => {
      const methods = (r.methods ?? []).map((m) => methodLabel(m, cat)).join(" · ");
      const progress = r.status === "running" ? ` ${pct(r.progress)}` : "";
      return `<tr data-run="${esc(r.id)}"><td><a href="/c/${enc}/runs/${encodeURIComponent(r.id)}">${esc(
        clock(r.created_at))}</a></td><td>${esc(methods)}</td><td class="num">${esc(r.question_count)}</td>
        <td>${r.answer ? "Answered" : "Retrieval only"}</td><td data-said>${saidHtml(r)}</td>
        <td class="num">${esc(usd(r.usd_total))}</td>
        <td>${pill(r.status)}${esc(progress)}</td></tr>`;
    })
    .join("");
  return `<table><thead><tr><th>Time</th><th>Methods</th><th>Questions</th><th>Answers</th>
    <th title="${esc(`The verdict on ${cat.answer_pair.map((m) => methodLabel(m, cat)).join(" against ")}`)}">Said</th><th>Cost</th><th>Status</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

// What a run on this collection can score, and what it cannot, in the catalog's labels.
function measureHtml(counts) {
  if (counts !== col) return evidenceHtml(counts);
  const label = (id) => metricLabel(id, col.qa_type, cat);
  const rows = Object.entries(col.measurable ?? {});
  const yes = rows.filter(([, m]) => m.measurable).map(([id, m]) => label(id) + (m.note ? ` (${m.note})` : ""));
  const no = rows.filter(([, m]) => !m.measurable).map(([id, m]) => label(id) + (m.note ? ` (${m.note})` : ""));
  return `<p>This run will measure: ${esc(yes.join(", ") || "nothing")}.</p>` +
    (no.length ? `<p class="muted">Not measurable here: ${esc(no.join("; "))}.</p>` : "");
}

// For an eval set other than the store's default, only the estimate knows what it holds here.
function evidenceHtml(counts) {
  const held = counts.evidence_named > 0
    ? `<p>Evidence ids: ${esc(count(counts.evidence_held))} of ${esc(count(counts.evidence_named))} named are in this store.</p>`
    : "";
  return `<p>${esc(plural(counts.questions, "question"))} in this eval set.</p>${held}`;
}

function setOptions(sets) {
  const blank = col.eval_set ? "" : '<option value="">(choose one)</option>';
  const options = sets
    .map((e) => `<option value="${esc(e.name)}"${e.name === col.eval_set ? " selected" : ""}>${esc(e.name)}</option>`)
    .join("");
  return blank + options;
}

function runFormHtml(ai, sets) {
  const groups = GROUPS.map(([id, title]) => {
    const boxes = Object.entries(cat.methods)
      .filter(([, m]) => m.group === id)
      .map(([mid, m]) => {
        const on = cat.answer_pair.includes(mid) || CONTROLS.includes(mid);
        const ok = col.methods_available.includes(mid);
        return `<label class="check"><input type="checkbox" name="method" value="${esc(mid)}"
          ${on && ok ? "checked" : ""} ${ok ? "" : "disabled"}> ${esc(m.label)}<code>${esc(mid)}</code>${costMark(mid, col, cat)}</label>`;
      })
      .join("");
    return boxes ? `<fieldset><legend>${esc(title)}</legend><div class="checks">${boxes}</div></fieldset>` : "";
  }).join("");
  const models = modelOptions(ai.models, ai.roles.answer ?? ai.roles.default);
  return `<div class="card form">${groups}
    <label class="check"><input type="checkbox" id="rf-answer" checked> Have the AI answer the questions <span class="cost muted">costs money</span></label>
    <div class="row"><label>Model<select id="rf-model">${models}</select></label>
      <label>Eval set<select id="rf-set">${setOptions(sets)}</select></label>
      <label>Questions<select id="rf-split">
        <option value="dev">Dev slice (${esc(count(col.dev_questions))})</option>
        <option value="full">All (${esc(count(col.questions))})</option></select></label></div>
    <div id="rf-measure">${measureHtml(col)}</div>
    <div class="row"><button type="button" id="rf-estimate">Estimate</button>
      <span id="rf-confirm"></span><span id="rf-msg" class="err" role="status"></span></div></div>`;
}

function bindRunForm() {
  const form = $("run-form");
  const body = () => ({
    collection: name,
    methods: [...form.querySelectorAll("input[name=method]:checked")].map((i) => i.value),
    answer: $("rf-answer").checked,
    model: $("rf-model").value || null,
    split: $("rf-split").value,
    eval_set: $("rf-set").value || null,
    k: 5,
    rrf_k: 60,
  });
  const forget = () => ($("rf-confirm").innerHTML = "");
  form.addEventListener("change", forget);
  // The counts follow the chosen eval set: re-read from the estimate, which is free.
  $("rf-set").addEventListener("change", async () => {
    $("rf-msg").textContent = "";
    const request = body();
    if (!request.methods.length) request.methods = ["hybrid"];
    try {
      const { estimate } = await api("/eval", { method: "POST", body: { ...request, confirm: false } });
      const chosen = $("rf-set").value;
      $("rf-split").querySelector("[value=dev]").textContent = `Dev slice (${count(estimate.dev_questions)})`;
      $("rf-split").querySelector("[value=full]").textContent = `All (${count(estimate.questions)})`;
      $("rf-measure").innerHTML = measureHtml(chosen === col.eval_set ? col : estimate);
    } catch (e) {
      $("rf-msg").textContent = e.message;
    }
  });
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

async function toggleRunForm() {
  const form = $("run-form");
  if (!form.hidden) {
    form.hidden = true;
    return;
  }
  try {
    const [ai, sets] = await Promise.all([api("/ai"), api("/eval-sets")]); // the models are whatever Settings holds now
    form.innerHTML = runFormHtml(ai, sets.filter((e) => e.qa_type === col.qa_type));
  } catch (e) {
    form.innerHTML = `<p class="err">${esc(e.message)}</p>`;
  }
  form.hidden = false;
  bindRunForm();
}

// -- items and questions --------------------------------------------------------------

// The built-in benchmark is offered while nothing is loaded, or the newest load did not finish;
// a collection that merely has no questions is not offered it again.
function needsLoad() {
  if (BUSY_STATES.has(col.state)) return false;
  return col.count === 0 || (loadJob != null && loadJob.status !== "done");
}

function itemsHtml() {
  const type = cat.qa_types[col.qa_type];
  return `<p>${esc(plural(col.count, type?.item_noun ?? "item"))}.</p>`;
}

function benchmarkHtml() {
  if (!needsLoad()) return "";
  const dataset = DATASETS[col.qa_type];
  const unfinished = col.count > 0 || loadJob != null;
  const label = unfinished ? "Finish loading" : `Load ${dataset}`;
  return `<div class="card form">
    <div>Or load the built-in benchmark, ${esc(dataset)}.</div>
    <div class="row"><label>Limit (blank for all)<input id="ld-limit" type="number" min="1"></label>
      <button type="button" id="ld-go">${esc(label)}</button>
      <span id="ld-confirm"></span><span id="ld-msg" class="err" role="status"></span></div></div>`;
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
        $("ld-confirm").textContent = `Queued (job ${short(first.job_id)})`;
        await refresh();
        return;
      }
      confirmSpend($("ld-confirm"), first.estimate, async () => {
        try {
          const job = await api("/collections/" + enc + "/load", { method: "POST", body: { ...base, confirm: true } });
          $("ld-confirm").textContent = `Queued (job ${short(job.job_id)})`;
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

// -- sources ----------------------------------------------------------------------------

function sourcesHtml() {
  const rows = col.sources ?? [];
  if (!rows.length) return "";
  const locked = BUSY_STATES.has(col.state) || col.unlabelled > 0;
  const body = rows
    .map(
      (r) => `<tr><td>${esc(r.source)}</td><td class="num">${esc(count(r.items))}</td>
        <td><input type="checkbox" data-source="${esc(r.source)}" aria-label="Use ${esc(r.source)} in retrieval"
          ${r.enabled ? "checked" : ""} ${locked ? "disabled" : ""}></td>
        <td><button type="button" class="secondary" data-delete="${esc(r.source)}">Delete</button></td></tr>`
    )
    .join("");
  const waiting = col.unlabelled > 0
    ? `<p class="muted">${esc(plural(col.unlabelled, "item"))} ${col.unlabelled === 1 ? "is" : "are"} still being
      labelled with their source; switches open when that finishes.</p>`
    : "";
  return `<table><thead><tr><th>Source</th><th>Items</th><th>Used by retrieval</th><th>Delete</th></tr></thead>
    <tbody>${body}</tbody></table>${waiting}`;
}

function bindSources() {
  const box = $("sources");
  box.addEventListener("change", async (e) => {
    const input = e.target.closest("input[data-source]");
    if (!input) return;
    $("sources-msg").textContent = "";
    try {
      await api(`/collections/${enc}/sources/${encodeURIComponent(input.dataset.source)}`, {
        method: "PATCH",
        body: { enabled: input.checked },
      });
      await refresh();
    } catch (err) {
      input.checked = !input.checked;
      $("sources-msg").textContent = err.message;
    }
  });
  box.addEventListener("click", (e) => {
    const button = e.target.closest("button[data-delete]");
    if (!button) return;
    const source = button.dataset.delete;
    const row = (col.sources ?? []).find((r) => r.source === source);
    askDelete($("sources-confirm"), {
      question: `Delete source ${source}: its ${count(row?.items ?? 0)} items?`,
      yes: `Yes, delete ${source}`,
      path: `/collections/${enc}/sources/${encodeURIComponent(source)}`,
      gone: `/c/${enc}`,
    });
  });
}

// -- its eval set -----------------------------------------------------------------------------------

function askedHtml() {
  if (!col.eval_set) {
    return '<p>No eval set yet. <a href="/eval-sets">Create one</a> to ask this store questions.</p>';
  }
  return `<p><a href="/e/${encodeURIComponent(col.eval_set)}">${esc(col.eval_set)}</a>:
    ${esc(plural(col.questions, "question"))}, of which ${esc(count(col.dev_questions))} in the dev slice
    that runs use by default.</p>`;
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
      <button type="button" id="kw-propose" class="secondary">Propose with AI</button></div>
    <p id="kw-hint" class="muted"></p><p id="kw-msg" role="status"></p>
    <h3>Earlier versions</h3><ul id="lists-history" class="history"></ul>`;
  for (const [k] of LIST_KEYS) $(`kw-${k}`).addEventListener("input", () => (listsDirty = true));
  $("kw-save").addEventListener("click", saveLists);
  $("kw-propose").addEventListener("click", proposeLists);
  $("lists-history").addEventListener("click", (e) => {
    const button = e.target.closest("button[data-version]");
    if (button) loadVersion(button.dataset.version);
  });
}

function historyHtml() {
  if (!versions.length) return '<li class="muted">No versions yet.</li>';
  return versions
    .map((v) => {
      const counts = LIST_KEYS.map(([k]) => `${count(v.counts[k] ?? 0)} ${k}`).join(", ");
      const active = v.active ? ' · <span class="ok">active</span>' : "";
      return `<li><code>${esc(short(v.version))}</code> ${esc(clock(v.created_at))} · ${esc(counts)}${active}
        <button type="button" class="secondary" data-version="${esc(v.version)}">Load into editor</button></li>`;
    })
    .join("");
}

// Fills the editor with an old version. The editor is then dirty, so a refresh keeps it; saving
// goes through the usual save, which makes a new version and re-keys.
async function loadVersion(version) {
  note("");
  try {
    const v = await api("/collections/" + enc + "/keyword-lists/" + encodeURIComponent(version));
    fillLists(v.lists);
    listsDirty = true;
    note(`Loaded version ${esc(short(version))}. Save lists to use it.`);
  } catch (e) {
    note(`<span class="err">${esc(e.message)}</span>`);
  }
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
    note(`<span class="ok">Saved as version ${esc(short(saved.version))}: re-keys ${esc(
      plural(col.count, "item"))}.</span> Job ${esc(short(saved.job_id))} ${pill("queued")}`);
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
        note("Proposed by the AI. Review them, then Save lists.");
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
  paint("lists-history", historyHtml());
  $("kw-hint").textContent = `Saving re-keys ${plural(col.count, "item")}, free.`;
  const version = col.keyword_version ?? null;
  if (version !== listsVersion && !listsDirty) {
    fillLists(col.keyword_lists);
    listsVersion = version;
  }
}

// -- delete ---------------------------------------------------------------------------

function buildDelete() {
  $("delete").innerHTML = `<div class="actions"><button type="button" id="del-ask" class="secondary">Delete…</button></div>
    <div id="del-confirm"></div>`;
  $("del-ask").addEventListener("click", () =>
    askDelete($("del-confirm"), {
      question: `Delete ${name}, with its items, lists and runs? Its eval sets stay. This cannot be undone.`,
      yes: `Yes, delete ${name}`,
      path: `/collections/${enc}`,
      gone: "/",
    })
  );
}

// -- refresh ----------------------------------------------------------------------------

let timer = null;
async function refresh() {
  clearTimeout(timer);
  try {
    cat = await catalog();
    let loads;
    [col, runs, loads, versions] = await Promise.all([
      api("/collections/" + enc),
      api("/eval?collection=" + enc),
      api("/jobs?collection=" + enc + "&kind=load&limit=1"),
      api("/collections/" + enc + "/keyword-lists"),
    ]);
    loadJob = loads[0] ?? null;
    $("head").innerHTML = headHtml();
    paint("answer", answerHtml());
    paint("runs", runsHtml());
    paint("asked", askedHtml());
    paint("items", itemsHtml());
    paint("sources", sourcesHtml());
    if (paint("benchmark", benchmarkHtml())) bindLoad();
    syncLists();
  } catch (e) {
    $("head").innerHTML = head({ title: name, crumb: [{ label: "Stores", href: "/" }] }) +
      `<p class="err">${esc(e.message)}</p>`;
  }
  timer = setTimeout(refresh, col && BUSY_STATES.has(col.state) ? FAST_MS : SLOW_MS);
}

$("new-run").addEventListener("click", toggleRunForm);
buildLists();
buildDelete();
bindSources();
importBox($("imp-items"), {
  uploadPath: `/collections/${enc}/uploads`,
  importPath: `/collections/${enc}/imports`,
  kind: "items",
  qaType: null,
  done: () => refresh(),
});
refresh();
