import { ImportBox } from "/static/import_box.js";
import { $, NONE, api, askDelete, catalog, clock, count, esc, head, notFound, plural, say, shell, short } from "/static/ui.js";
import { Labelling } from "/static/labelling.js";

const PAGE_SIZE = 50;
const VERSIONS_SHOWN = 20; // the newest; Show all versions lists the rest

const name = decodeURIComponent(location.pathname.split("/")[2] ?? "");
const enc = encodeURIComponent(name);

let cat = null;
let set = null; // the latest GET /eval-sets/{name}
let rows = []; // the working copy of its questions; nothing here is written until Save changes
let page = 0;
let dirty = false;
let added = 0;
let allVersions = false; // Show all versions was clicked
let viewing = null; // the old version shown read-only, { version, questions }; null for none

shell(`e/${name}`);

const byId = (a, b) => String(a.id).localeCompare(String(b.id), undefined, { numeric: true });
const lines = (text) => text.split("\n").map((s) => s.trim()).filter(Boolean);
const commas = (text) => text.split(",").map((s) => s.trim()).filter(Boolean);

// -- head and counts ----------------------------------------------------------------------------

function headHtml() {
  const type = cat.qa_types[set.qa_type];
  return head({
    title: name,
    crumb: [{ label: "Eval sets", href: "/eval-sets" }],
    sub: `${type?.label ?? set.qa_type}: ${plural(set.questions.length, "question")}`,
    aside: set.version ? `version ${esc(short(set.version))}` : "",
  });
}

// "n of total", or the missing-value mark when there are no questions.
function ofTotal(n, total) {
  return total > 0 ? `${count(n)} of ${count(total)}` : NONE;
}

function fieldsHtml() {
  const total = rows.length;
  const have = (test) => rows.filter(test).length;
  const evidence = have((q) => q.gold_ids.length);
  return `Worked solutions: ${ofTotal(have((q) => q.meta?.solution), total)} · Skills: ${ofTotal(
    have((q) => q.meta?.skill), total)} · Evidence ids: ${evidence > 0 ? ofTotal(evidence, total) : NONE}`;
}

// -- the questions table ---------------------------------------------------------------------------

function viewRow(q, i) {
  return `<tr data-row="${i}"><td>${esc(q.id)}</td><td>${esc(q.text)}</td>
    <td>${esc((q.choices ?? []).join(" | "))}</td><td>${esc(q.gold_answers.join(" | "))}</td>
    <td>${esc(q.gold_ids.join(", "))}</td><td>${esc(q.meta?.source ?? "")}</td>
    <td><button type="button" class="secondary" data-edit="${i}">Edit</button>
      <button type="button" class="secondary" data-remove="${i}">Delete</button></td></tr>`;
}

function editRow(q, i) {
  const id = q._new
    ? `<input data-field="id" aria-label="id" value="${esc(q.id)}">`
    : esc(q.id);
  return `<tr data-row="${i}"><td>${id}</td>
    <td><textarea data-field="text" aria-label="text" rows="3">${esc(q.text)}</textarea></td>
    <td><textarea data-field="choices" aria-label="choices, one per line" rows="3">${esc((q.choices ?? []).join("\n"))}</textarea></td>
    <td><textarea data-field="gold_answers" aria-label="answer, one per line" rows="2">${esc(q.gold_answers.join("\n"))}</textarea></td>
    <td><input data-field="gold_ids" aria-label="evidence ids, comma separated" value="${esc(q.gold_ids.join(", "))}"></td>
    <td>${esc(q.meta?.source ?? "")}</td>
    <td><button type="button" data-done="${i}">Done</button>
      <button type="button" class="secondary" data-remove="${i}">Delete</button></td></tr>`;
}

function questionsHtml() {
  if (!rows.length) return '<p class="muted">No questions yet. Add one, or import a file.</p>';
  const from = page * PAGE_SIZE;
  const body = rows
    .slice(from, from + PAGE_SIZE)
    .map((q, k) => (q._open ? editRow(q, from + k) : viewRow(q, from + k)))
    .join("");
  return `<table><thead><tr><th>Id</th><th>Question</th><th>Choices</th><th>Answer</th>
    <th>Evidence ids</th><th>Source</th><th></th></tr></thead><tbody>${body}</tbody></table>`;
}

function pagerHtml() {
  const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  if (pages === 1) return "";
  return `<button type="button" class="secondary" data-page="-1" ${page === 0 ? "disabled" : ""}>Previous</button>
    <span>Page ${page + 1} of ${pages}</span>
    <button type="button" class="secondary" data-page="1" ${page >= pages - 1 ? "disabled" : ""}>Next</button>`;
}

function draw() {
  page = Math.min(page, Math.max(0, Math.ceil(rows.length / PAGE_SIZE) - 1));
  $("questions").innerHTML = questionsHtml();
  $("pager").innerHTML = pagerHtml();
  $("fields").textContent = fieldsHtml();
  showDirty();
}

// An import reloads the set, so it waits until the edits are saved or discarded.
function showDirty() {
  $("dirty").textContent = dirty ? "Unsaved changes" : "";
  $("imp-wrap").disabled = dirty;
  $("imp-note").textContent = dirty ? "Save or discard your changes before importing" : "";
  showRestore();
}

// Reads the open rows' fields back into the working copy.
function commit() {
  for (const tr of $("questions").querySelectorAll("tr[data-row]")) {
    const q = rows[Number(tr.dataset.row)];
    const field = (f) => tr.querySelector(`[data-field=${f}]`)?.value;
    if (!q._open || field("text") === undefined) continue;
    if (q._new) q.id = field("id").trim();
    q.text = field("text");
    q.choices = lines(field("choices")).length ? lines(field("choices")) : null;
    q.gold_answers = lines(field("gold_answers"));
    q.gold_ids = commas(field("gold_ids"));
  }
}

function bindTable() {
  $("questions").addEventListener("input", () => {
    if (dirty) return;
    dirty = true; // typing in an open row is an edit, though the table is not redrawn for it
    showDirty();
  });
  $("questions").addEventListener("click", (e) => {
    const button = e.target.closest("button");
    if (!button) return;
    commit();
    const d = button.dataset;
    if (d.edit !== undefined) rows[Number(d.edit)]._open = true;
    if (d.done !== undefined) {
      rows[Number(d.done)]._open = false;
      dirty = true;
    }
    if (d.remove !== undefined) {
      rows.splice(Number(d.remove), 1);
      dirty = true;
    }
    draw();
  });
  $("pager").addEventListener("click", (e) => {
    const button = e.target.closest("button[data-page]");
    if (!button) return;
    commit();
    page += Number(button.dataset.page);
    draw();
  });
  $("add").addEventListener("click", () => {
    commit();
    const taken = new Set(rows.map((q) => q.id));
    while (taken.has(`new-${++added}`));
    rows.push({ id: `new-${added}`, text: "", choices: null, gold_answers: [], gold_ids: [], meta: { source: "hand" }, _open: true, _new: true });
    page = Math.floor((rows.length - 1) / PAGE_SIZE);
    dirty = true;
    draw();
  });
  $("save").addEventListener("click", save);
  $("save-msg").addEventListener("click", async (e) => {
    if (!e.target.closest("#reload")) return;
    $("save-msg").textContent = "";
    try {
      await load();
      $("save-msg").className = "";
    } catch (err) {
      $("save-msg").textContent = err.message;
    }
  });
}

// -- saving ------------------------------------------------------------------------------------------

async function save() {
  commit();
  $("save-msg").className = "";
  $("save-msg").textContent = "";
  const questions = rows.map(({ id, text, choices, gold_answers, gold_ids, meta, judged_ids }) => ({
    id, text, choices, gold_answers, gold_ids, meta, judged_ids, // meta and judged_ids go back as read
  }));
  try {
    const saved = await api(`/eval-sets/${enc}/questions`, {
      method: "PUT",
      body: { questions, base_version: set.version },
    });
    await load();
    $("save-msg").className = "ok";
    $("save-msg").textContent = `Saved as version ${short(saved.version)}.`;
  } catch (e) {
    $("save-msg").className = "err"; // the edits stay where they are
    $("save-msg").textContent = e.message;
    if (e.status === 409) { // stale or busy: reloading is the way on, and it is the user's call
      $("save-msg").insertAdjacentHTML("beforeend",
        ' <button type="button" id="reload" class="secondary">Reload (discard my edits)</button>');
    }
  }
}

// -- versions, leaked, delete ------------------------------------------------------------------------

function versionsHtml() {
  let shown = allVersions ? set.versions : set.versions.slice(0, VERSIONS_SHOWN);
  // A restored old version is current but keeps its old place in the list: it is always shown.
  const current = set.versions.find((v) => v.version === set.version);
  if (current && !shown.includes(current)) shown = [current, ...shown];
  return shown
    .map((v) => {
      const current = v.version === set.version;
      const tail = current
        ? ' · <span class="ok">current</span>'
        : `<button type="button" class="secondary" data-view="${esc(v.version)}">View</button>`;
      return `<li><code>${esc(short(v.version))}</code> ${esc(clock(v.created_at))} · ${esc(
        plural(v.questions, "question"))}${tail}</li>`;
    })
    .join("");
}

function drawVersions() {
  $("versions").innerHTML = versionsHtml();
  $("ver-more").innerHTML = !allVersions && set.versions.length > VERSIONS_SHOWN
    ? '<div class="actions"><button type="button" id="ver-all" class="secondary">Show all versions</button></div>'
    : "";
}

// An old version's questions, read only, in the shape of the page's own table.
function versionTableHtml(questions) {
  const sorted = [...questions].sort(byId);
  if (set.qa_type === "documents") {
    const body = sorted.map((q) => `<tr><td>${esc(q.text)}</td><td>${count(q.gold_ids.length)}</td>
      <td>${q.judged_ids ? count(q.judged_ids.length) : NONE}</td></tr>`).join("");
    return `<table><thead><tr><th>Question</th><th>Right passages</th><th>Judged</th></tr></thead>
      <tbody>${body}</tbody></table>`;
  }
  const body = sorted.map((q) => `<tr><td>${esc(q.id)}</td><td>${esc(q.text)}</td>
    <td>${esc((q.choices ?? []).join(" | "))}</td><td>${esc(q.gold_answers.join(" | "))}</td>
    <td>${esc(q.gold_ids.join(", "))}</td><td>${esc(q.meta?.source ?? "")}</td></tr>`).join("");
  return `<table><thead><tr><th>Id</th><th>Question</th><th>Choices</th><th>Answer</th>
    <th>Evidence ids</th><th>Source</th></tr></thead><tbody>${body}</tbody></table>`;
}

function drawView() {
  if (!viewing) {
    $("ver-view").innerHTML = "";
    return;
  }
  const { version, questions } = viewing;
  $("ver-view").innerHTML = `<h3>${esc(`Version ${short(version)}: ${plural(questions.length, "question")}`)}</h3>
    <div class="scroll">${versionTableHtml(questions)}</div>
    <div class="actions"><button type="button" id="ver-restore">Restore this version</button>
      <button type="button" id="ver-back" class="secondary">Back to the current version</button></div>`;
  showRestore();
}

// Restoring replaces every question, so it waits for edits and marks not yet saved.
function showRestore() {
  const button = $("ver-restore");
  if (button) button.disabled = dirty || labelling.unsaved || labelling.busy();
}

async function viewVersion(version) {
  say($("ver-msg"), "", "err");
  try {
    const old = await api(`/eval-sets/${enc}?version=${encodeURIComponent(version)}`);
    viewing = { version: old.questions_version, questions: old.questions };
    drawView();
  } catch (e) {
    say($("ver-msg"), e.message, "err");
  }
}

// The whole-set PUT of the old questions, checked against the version the page holds.
async function restore() {
  const { version, questions } = viewing;
  say($("ver-msg"), "", "err");
  await labelling.restore(async () => {
    try {
      await api(`/eval-sets/${enc}/questions`, { method: "PUT", body: { questions, base_version: set.version } });
    } catch (e) {
      say($("ver-msg"), e.message, "err");
      return false;
    }
    viewing = null;
    drawView();
    try {
      await load();
      say($("ver-msg"), `Restored version ${short(version)}.`, "ok");
    } catch (e) {
      say($("ver-msg"), e.message, "err");
    }
    return true;
  });
}

function bindVersions() {
  $("versions").addEventListener("click", (e) => {
    const button = e.target.closest("button[data-view]");
    if (button) viewVersion(button.dataset.view);
  });
  $("ver-more").addEventListener("click", (e) => {
    if (!e.target.closest("#ver-all")) return;
    allVersions = true;
    drawVersions();
  });
  $("ver-view").addEventListener("click", (e) => {
    if (e.target.closest("#ver-restore")) restore();
    if (e.target.closest("#ver-back")) {
      viewing = null;
      drawView();
      say($("ver-msg"), "", "err");
    }
  });
}

function bindLeaked() {
  $("leaked-include").addEventListener("change", async (e) => {
    $("leaked-msg").textContent = "";
    try {
      await api(`/eval-sets/${enc}`, { method: "PATCH", body: { include_leaked: e.target.checked } });
    } catch (err) {
      e.target.checked = !e.target.checked;
      $("leaked-msg").textContent = err.message;
    }
  });
}

function bindDelete() {
  $("del-ask").addEventListener("click", async () => {
    $("del-confirm").innerHTML = "";
    try {
      const { runs } = await api(`/eval-sets/${enc}`, { method: "DELETE" }); // without confirm it only counts
      askDelete($("del-confirm"), {
        question: `Delete eval set ${name} and its ${plural(runs, "run")}?`,
        yes: `Yes, delete ${name}`,
        path: `/eval-sets/${enc}?confirm=true`,
        gone: "/eval-sets",
      });
    } catch (e) {
      $("del-confirm").innerHTML = `<p class="err" role="status">${esc(e.message)}</p>`;
    }
  });
}

// -- a Documents set: labelling (labelling.js) ----------------------------------------------------

const labelling = new Labelling({ currentSet: () => set, currentRows: () => rows, name, enc, load, showRestore });

// -- loading -------------------------------------------------------------------------------------------

async function load() {
  set = await api(`/eval-sets/${enc}`);
  rows = set.questions.map((q) => ({ ...q, gold_answers: q.gold_answers ?? [], gold_ids: q.gold_ids ?? [] })).sort(byId);
  dirty = false;
  $("head").innerHTML = headHtml();
  $("run-store").innerHTML = set.store
    ? `<a href="/c/${encodeURIComponent(set.store)}?eval_set=${enc}">Run against its store</a>`
    : "";
  drawVersions();
  // A Documents set is labelled, one question at a time; the leak check is off for it.
  const documents = set.qa_type === "documents";
  $("bench").hidden = documents;
  $("leaked").hidden = documents;
  $("labelling").hidden = !documents;
  if (documents) {
    await labelling.drawLabelling();
    showDirty();
  } else {
    $("leaked-include").checked = set.include_leaked;
    draw();
  }
}

async function start() {
  try {
    cat = await catalog();
    await load();
    new ImportBox($("imp-questions"), {
      uploadPath: `/eval-sets/${enc}/uploads`,
      importPath: `/eval-sets/${enc}/imports`,
      kind: "questions",
      qaType: set.qa_type,
      typeLabel: cat.qa_types[set.qa_type]?.label,
      done: () => (dirty ? null : load()), // edits made while it ran are kept; Save then says it moved on
    });
  } catch (e) {
    if (e.status === 404) {
      notFound({ title: name, crumb: [{ label: "Eval sets", href: "/eval-sets" }], text: `No eval set named ${name}.`,
        links: [{ label: "Eval sets", href: "/eval-sets" }] });
      return;
    }
    $("head").innerHTML = head({ title: name, crumb: [{ label: "Eval sets", href: "/eval-sets" }] }) +
      `<p class="err">${esc(e.message)}</p>`;
  }
}

bindTable();
bindLeaked();
bindDelete();
labelling.bindLabelling();
bindVersions();
start();
