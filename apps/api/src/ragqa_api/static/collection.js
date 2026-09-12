import { $, BUSY, api, askDelete, axisHtml, catalog, clock, confirmSpend, count, costMark, esc, gapBar, gapSpan,
  head, legendHtml, methodLabel, metricLabel, modelOptions, paint, pct, pill, plural, pts, refreshSpend, shell, short, UNPRICED, usd, verdictHtml,
  upload, verdictNotes, wordHtml } from "/static/ui.js";

const SLOW_MS = 15000;
const FAST_MS = 3000;
const BUSY_STATES = new Set(["loading", "rekeying", "evaluating", "queued", "importing", "labelling"]); // the collection's own
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
const uploads = { items: null, questions: null }; // per kind: the upload being previewed, or null
const previewing = { items: 0, questions: 0 }; // per kind: the newest preview request, so an older answer is ignored

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
    crumb: [{ label: "Collections", href: "/" }],
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
      run ${esc(a.runs)} on this collection</p>`;
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
function measureHtml() {
  const label = (id) => metricLabel(id, col.qa_type, cat);
  const rows = Object.entries(col.measurable ?? {});
  const yes = rows.filter(([, m]) => m.measurable).map(([id, m]) => label(id) + (m.note ? ` (${m.note})` : ""));
  const no = rows.filter(([, m]) => !m.measurable).map(([id, m]) => label(id) + (m.note ? ` (${m.note})` : ""));
  return `<p>This run will measure: ${esc(yes.join(", ") || "nothing")}.</p>` +
    (no.length ? `<p class="muted">Not measurable here: ${esc(no.join("; "))}.</p>` : "");
}

function runFormHtml(ai) {
  const groups = GROUPS.map(([id, title]) => {
    const boxes = Object.entries(cat.methods)
      .filter(([, m]) => m.group === id)
      .map(([mid, m]) => {
        const on = cat.answer_pair.includes(mid) || CONTROLS.includes(mid);
        const ok = col.methods_available.includes(mid);
        return `<label class="check"><input type="checkbox" name="method" value="${esc(mid)}"
          ${on && ok ? "checked" : ""} ${ok ? "" : "disabled"}> ${esc(m.label)}<code>${esc(mid)}</code>${costMark(mid, col.summarize, cat)}</label>`;
      })
      .join("");
    return boxes ? `<fieldset><legend>${esc(title)}</legend><div class="checks">${boxes}</div></fieldset>` : "";
  }).join("");
  const models = modelOptions(ai.models, ai.roles.answer ?? ai.roles.default);
  return `<div class="card form">${groups}
    <label class="check"><input type="checkbox" id="rf-answer" checked> Have the AI answer the questions <span class="cost muted">costs money</span></label>
    <div class="row"><label>Model<select id="rf-model">${models}</select></label>
      <label>Questions<select id="rf-split">
        <option value="dev">Dev slice (${esc(count(col.dev_questions))})</option>
        <option value="full">All (${esc(count(col.questions))})</option></select></label></div>
    ${measureHtml()}
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

async function toggleRunForm() {
  const form = $("run-form");
  if (!form.hidden) {
    form.hidden = true;
    return;
  }
  try {
    form.innerHTML = runFormHtml(await api("/ai")); // the model list is whatever Settings holds now
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
      question: `Delete source ${source}: its ${count(row?.items ?? 0)} items and the questions imported with it?`,
      yes: `Yes, delete ${source}`,
      path: `/collections/${enc}/sources/${encodeURIComponent(source)}`,
      gone: `/c/${enc}`,
    });
  });
}

// -- importing a file ---------------------------------------------------------------------

// The file's name as a source name: other characters become "-", at most 64 long.
function sourceFor(file) {
  return file.name.replace(/\.[^.]*$/, "").replace(/[^A-Za-z0-9 _.-]/g, "-").slice(0, 64) || "upload";
}

function buildImport(kind) {
  const accept = ".jsonl,.json,.csv,.tsv,.txt,.md,.zip";
  $(`imp-${kind}`).innerHTML = `<div class="card form">
    <label>File<input type="file" id="imp-${kind}-file" accept="${accept}"></label>
    <label>Source name<input id="imp-${kind}-source" maxlength="64"></label>
    <div class="row"><button type="button" id="imp-${kind}-go">Upload</button>
      <span id="imp-${kind}-msg" class="err" role="status"></span></div>
    <div id="imp-${kind}-preview"></div></div>`;
  $(`imp-${kind}-file`).addEventListener("change", (e) => {
    const file = e.target.files[0];
    uploads[kind] = null; // the old preview belongs to the old file
    previewing[kind]++; // and no answer still on its way may paint it back
    $(`imp-${kind}-preview`).innerHTML = "";
    $(`imp-${kind}-msg`).textContent = "";
    if (file) $(`imp-${kind}-source`).value = sourceFor(file);
  });
  $(`imp-${kind}-go`).addEventListener("click", () => uploadFile(kind));
  // The estimate is priced under the source name, so a new name needs a new estimate.
  $(`imp-${kind}-source`).addEventListener("input", () => {
    if (uploads[kind]) previewAgain(kind);
  });
  $(`imp-${kind}-preview`).addEventListener("change", (e) => {
    if (e.target.closest("select[data-field]")) previewAgain(kind);
  });
}

async function uploadFile(kind) {
  const file = $(`imp-${kind}-file`).files[0];
  const msg = $(`imp-${kind}-msg`);
  msg.textContent = "";
  if (!file) {
    msg.textContent = "Pick a file first.";
    return;
  }
  try {
    const path = `/collections/${enc}/uploads?filename=${encodeURIComponent(file.name)}&kind=${kind}`;
    const got = await upload(path, file);
    uploads[kind] = got.upload_id;
    showPreview(kind, got.preview, got.estimate);
  } catch (e) {
    uploads[kind] = null;
    $(`imp-${kind}-preview`).innerHTML = "";
    msg.textContent = e.message;
  }
}

function mappingOf(kind) {
  const out = {};
  for (const select of $(`imp-${kind}-preview`).querySelectorAll("select[data-field]")) {
    out[select.dataset.field] = select.value || null;
  }
  return out;
}

function importBody(kind, confirm) {
  return {
    upload_id: uploads[kind],
    kind,
    mapping: mappingOf(kind),
    source_name: $(`imp-${kind}-source`).value,
    confirm,
  };
}

async function previewAgain(kind) {
  const msg = $(`imp-${kind}-msg`);
  const ticket = ++previewing[kind];
  msg.textContent = "";
  $(`imp-${kind}-import`)?.setAttribute("disabled", ""); // until the fresh estimate is in
  try {
    const got = await api(`/collections/${enc}/imports`, { method: "POST", body: importBody(kind, false) });
    if (ticket === previewing[kind]) showPreview(kind, got.preview, got.estimate);
  } catch (e) {
    if (ticket === previewing[kind]) msg.textContent = e.message;
  }
}

function previewHtml(kind, p, estimate) {
  const problems = p.problems.length
    ? `<ul class="warn">${p.problems.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`
    : "";
  const sample = p.sample.length
    ? `<div class="scroll"><table><thead><tr>${p.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
        <tbody>${p.sample.map((row) => `<tr>${p.columns.map((c) => `<td>${esc(row[c] ?? "")}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`
    : "";
  const selects = Object.keys(p.mapping)
    .map((field) => {
      const options = ['<option value="">(none)</option>']
        .concat(p.columns.map((c) => `<option value="${esc(c)}"${p.mapping[field] === c ? " selected" : ""}>${esc(c)}</option>`))
        .join("");
      const star = p.required.includes(field) ? " (required)" : "";
      return `<label>${esc(field)}${star}<select data-field="${esc(field)}">${options}</select></label>`;
    })
    .join("");
  const cost = estimate
    ? `<span class="muted">${esc(estimate.est_usd == null ? UNPRICED : usd(estimate.est_usd))} · ${esc(plural(estimate.calls ?? 0, "AI call"))}</span>`
    : "";
  const missing = p.missing.length ? `<p class="err">Map these fields first: ${esc(p.missing.join(", "))}.</p>` : "";
  return `<p>${esc(p.format)} · ${esc(plural(p.rows, "row"))} · ${esc(count(p.usable))} usable</p>${problems}${sample}
    <div class="row">${selects}</div>${missing}
    <div class="row"><button type="button" id="imp-${kind}-import" ${p.ready ? "" : "disabled"}>Import</button>
      ${cost}<span id="imp-${kind}-confirm"></span></div>`;
}

function showPreview(kind, preview, estimate) {
  $(`imp-${kind}-preview`).innerHTML = previewHtml(kind, preview, estimate);
  $(`imp-${kind}-import`).addEventListener("click", () => startImport(kind, estimate));
}

async function startImport(kind, estimate) {
  const msg = $(`imp-${kind}-msg`);
  msg.textContent = "";
  const queue = async () => {
    try {
      const job = await api(`/collections/${enc}/imports`, { method: "POST", body: importBody(kind, true) });
      $(`imp-${kind}-preview`).innerHTML = `<p class="ok">Queued (job ${esc(short(job.job_id))})</p>`;
      uploads[kind] = null;
      $(`imp-${kind}-file`).value = "";
      refreshSpend();
      await refresh();
    } catch (e) {
      msg.textContent = e.message;
    }
  };
  if (estimate && estimate.est_usd > 0) confirmSpend($(`imp-${kind}-confirm`), estimate, queue);
  else await queue();
}

// -- what it gets asked ----------------------------------------------------------------------

// "n of total", or a dash when there are no questions.
function ofTotal(n, total) {
  return total > 0 ? `${count(n)} of ${count(total)}` : "—";
}

function fieldsHtml() {
  const q = col.fields.questions;
  const evidence = q.gold_ids > 0 ? `${count(q.gold_ids)} of ${count(q.total)}` : "—";
  return `<p class="muted">Worked solutions: ${esc(ofTotal(q.solution, q.total))} · Skills: ${esc(
    ofTotal(q.skill, q.total))} · Evidence ids: ${esc(evidence)}</p>`;
}

function askedHtml() {
  const n = col.leaked;
  const leaked = n > 0
    ? `<p class="warn">${esc(n === 1 ? "1 question appears" : `${count(n)} questions appear`)} among the items
      (leaked) and ${n === 1 ? "is" : "are"} left out of runs.
      <label class="check"><input type="checkbox" id="leaked-include" ${col.include_leaked ? "checked" : ""}>
      Include them anyway</label></p><p id="leaked-msg" class="err" role="status"></p>`
    : "";
  return `<p>${esc(plural(col.questions, "question"))}, of which ${esc(count(col.dev_questions))} in the dev
    slice that runs use by default.</p>${leaked}${fieldsHtml()}`;
}

function bindAsked() {
  $("asked").addEventListener("change", async (e) => {
    const input = e.target.closest("#leaked-include");
    if (!input) return;
    try {
      await api(`/collections/${enc}/questions/leaked`, { method: "PATCH", body: { include: input.checked } });
      await refresh();
    } catch (err) {
      input.checked = !input.checked;
      $("leaked-msg").textContent = err.message;
    }
  });
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
      question: `Delete ${name}, with its items, questions, lists and runs? This cannot be undone.`,
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
    $("head").innerHTML = head({ title: name, crumb: [{ label: "Collections", href: "/" }] }) +
      `<p class="err">${esc(e.message)}</p>`;
  }
  timer = setTimeout(refresh, col && BUSY_STATES.has(col.state) ? FAST_MS : SLOW_MS);
}

$("new-run").addEventListener("click", toggleRunForm);
buildLists();
buildDelete();
bindSources();
bindAsked();
buildImport("items");
buildImport("questions");
refresh();
