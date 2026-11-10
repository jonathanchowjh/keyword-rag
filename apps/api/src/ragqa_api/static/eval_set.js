import { $, api, askDelete, catalog, clock, confirmSpend, count, esc, head, importBox, passageHtml, plural, refreshSpend, shell,
  short, whereHtml } from "/static/ui.js";

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

// "n of total", or a dash when there are no questions.
function ofTotal(n, total) {
  return total > 0 ? `${count(n)} of ${count(total)}` : "—";
}

function fieldsHtml() {
  const total = rows.length;
  const have = (test) => rows.filter(test).length;
  const evidence = have((q) => q.gold_ids.length);
  return `Worked solutions: ${ofTotal(have((q) => q.meta?.solution), total)} · Skills: ${ofTotal(
    have((q) => q.meta?.skill), total)} · Evidence ids: ${evidence > 0 ? ofTotal(evidence, total) : "—"}`;
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
      <td>${q.judged_ids ? count(q.judged_ids.length) : "–"}</td></tr>`).join("");
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
  if (button) button.disabled = dirty || unsaved || busy();
}

async function viewVersion(version) {
  say("ver-msg", "");
  try {
    const old = await api(`/eval-sets/${enc}?version=${encodeURIComponent(version)}`);
    viewing = { version: old.questions_version, questions: old.questions };
    drawView();
  } catch (e) {
    say("ver-msg", e.message);
  }
}

// The whole-set save of the old questions, checked against the version the page holds. While
// it is under way no mark, Find, Save, Label or Delete can change what the panel holds.
async function restore() {
  const { version, questions } = viewing;
  say("ver-msg", "");
  restoring = true;
  canSave(); // set before the first await, so a second click finds it disabled
  try {
    await api(`/eval-sets/${enc}/questions`, { method: "PUT", body: { questions, base_version: set.version } });
  } catch (e) {
    restoring = false;
    say("ver-msg", e.message);
    canSave();
    return;
  }
  viewing = null;
  drawView();
  try {
    await load();
    say("ver-msg", `Restored version ${short(version)}.`, "ok");
  } catch (e) {
    say("ver-msg", e.message);
  }
  restoring = false;
  // The question it held may be gone. Typed in while the PUT was out, the panel is left as it
  // is: the next Save is checked against the question as read, and offers Reload.
  if (open && !saving && !unsaved) openQuestion(null);
  canSave();
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
      say("ver-msg", "");
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

// -- a Documents set: labelling -------------------------------------------------------------------
// Its questions are labelled against the linked store: Find passages, then y or n on each. Every
// save writes that one question; only Get the AI's answer spends, after its estimate is confirmed.

let open = null; // the question in the panel as read, its conflict check; null for a new one
let shown = []; // the passages listed: the pool's, then labelled ones it missed, then lost ids
let found = []; // passages from Search the store not listed above
let notes = []; // the pool's notes
const LABEL = { y: "Right", n: "Wrong" }; // a mark as the buttons say it
let marks = new Map(); // passage id -> "y" or "n"; unmarked is not judged
let finding = 0; // the latest Find passages, so an older reply is dropped
let saving = false; // a save is in flight: nothing may change which question the panel holds
let restoring = false; // Restore's PUT is in flight: the same holds, and no mark changes
let unsaved = false; // a mark, the text or the reference answer changed since the panel opened or saved
let discarding = null; // what Label or Delete will do once its unsaved marks are discarded
let asking = 0; // the latest Get the AI's answer: a price drawn for an older one is dropped
let answering = 0; // the question the panel holds: a paid answer for an older one is not shown
let paying = false; // the confirmed AI's answer is out: Start can't buy it again

const busy = () => saving || restoring;

const docStore = () => set.store;
const lostIds = () => [...new Set(Object.values(set.missing ?? {}).flat())];

// `stores`: the Documents stores, in name order, each with its default eval set.
function storeLineHtml(stores) {
  if (docStore()) {
    const store = docStore();
    return `<p>Labelled against <a href="/c/${encodeURIComponent(store)}">${esc(store)}</a>.</p>`;
  }
  // An older set is usually its store's default, named after it: that store comes first.
  const pick = stores.find((s) => s.eval_set === name) ?? stores.find((s) => s.name === name) ?? stores[0];
  const options = stores
    .map((s) => `<option value="${esc(s.name)}" ${s === pick ? "selected" : ""}>${esc(s.name)}</option>`)
    .join("");
  return `<div class="row"><label>Pick a store to label against: <select id="lab-pick">${options}</select></label>
    <button type="button" id="lab-link" ${stores.length ? "" : "disabled"}>Link</button></div>`;
}

function warningsHtml() {
  const out = [];
  const lost = lostIds().length;
  if (docStore() && lost) {
    out.push(lost === 1
      ? `1 labelled passage is no longer in ${docStore()}`
      : `${count(lost)} labelled passages are no longer in ${docStore()}`);
  }
  const unscored = rows.filter((q) => !q.gold_ids.length).length;
  if (unscored) {
    out.push(unscored === 1
      ? "1 question has no right passage; it is not scored"
      : `${count(unscored)} questions have no right passage; they are not scored`);
  }
  return out.map((line) => `<p class="warn">${esc(line)}</p>`).join("");
}

function labelRowHtml(q) {
  const lost = set.missing?.[q.id]?.length ?? 0;
  const gone = lost && docStore() ? ` <span class="muted">(${count(lost)} no longer in ${esc(docStore())})</span>` : "";
  const label = docStore()
    ? `<button type="button" class="secondary" data-label="${esc(q.id)}" ${busy() ? "disabled" : ""}>Label</button> `
    : "";
  return `<tr><td>${esc(q.text)}</td><td>${count(q.gold_ids.length)}${gone}</td>
    <td>${q.judged_ids ? count(q.judged_ids.length) : "–"}</td>
    <td>${label}<button type="button" class="secondary" data-del="${esc(q.id)}" ${busy() ? "disabled" : ""}>Delete question…</button></td></tr>`;
}

function labelTableHtml() {
  if (!rows.length) return '<p class="muted">No questions yet.</p>';
  return `<table><thead><tr><th>Question</th><th>Right passages</th><th>Judged</th><th></th></tr></thead>
    <tbody>${rows.map(labelRowHtml).join("")}</tbody></table>`;
}

async function drawLabelling() {
  const stores = docStore()
    ? []
    : (await api("/collections"))
      .filter((c) => c.qa_type === "documents")
      .sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  $("lab-store").innerHTML = storeLineHtml(stores);
  $("lab-warn").innerHTML = warningsHtml();
  $("lab-questions").innerHTML = labelTableHtml();
  $("lab-panel").hidden = !docStore();
  $("lab-ai").hidden = !set.answer_available; // no answer model: the typed field is all there is
  $("lab-ai-hint").hidden = !set.answer_available;
  canSave();
}

// -- the panel --

// Every passage id in display order, then any marked id not listed (a Find that failed).
function listedIds() {
  const ids = [...shown, ...found].map((p) => p.id);
  return [...ids, ...[...marks.keys()].filter((id) => !ids.includes(id))];
}

function markHtml(p) {
  const mark = marks.get(p.id);
  const button = (m) => `<button type="button" class="${mark === m ? "" : "secondary"}" data-mark="${m}"
    aria-pressed="${mark === m}">${LABEL[m]}</button>`;
  let body = `<code>${esc(p.id)}</code>`; // the store no longer holds it: its id is all there is
  if (p.text !== undefined) {
    body = passageHtml(p.text, { meta: whereHtml(p) });
  }
  return `<li data-passage="${esc(p.id)}">${body}<div class="actions">${button("y")}${button("n")}</div></li>`;
}

function drawPanel() {
  $("lab-notes").innerHTML = notes.map((n) => `<li>${esc(n)}</li>`).join("");
  $("lab-passages").innerHTML = shown.map(markHtml).join("");
  $("lab-found").innerHTML = found.length
    ? `<h3>From your search</h3><ul class="evidence">${found.map(markHtml).join("")}</ul>`
    : "";
  const ids = listedIds();
  const y = ids.filter((id) => marks.get(id) === "y").length;
  const n = ids.filter((id) => marks.get(id) === "n").length;
  $("lab-count").textContent = ids.length
    ? `${count(y)} right, ${count(n)} wrong, ${count(ids.length - y - n)} not judged`
    : "";
  canSave();
}

function canSave() {
  const blank = !$("lab-text").value.trim();
  for (const id of ["lab-find", "lab-save", "lab-next"]) $(id).disabled = blank || busy();
  for (const b of $("lab-questions").querySelectorAll("button[data-label], button[data-del]")) b.disabled = busy();
  for (const b of $("lab-panel").querySelectorAll("button[data-mark]")) b.disabled = restoring;
  const right = listedIds().some((id) => marks.get(id) === "y");
  $("lab-ai").disabled = blank || !right || busy();
  for (const b of $("lab-ai-answer").querySelectorAll("button[data-ai-mark]")) b.disabled = busy();
  for (const b of $("lab-ai-confirm").querySelectorAll("button")) b.disabled = busy() || paying;
  showRestore();
}

// A change to the panel. A Delete asked before it checked for unsaved marks, so it is withdrawn.
function markUnsaved() {
  unsaved = true;
  $("lab-del").innerHTML = "";
}

// Label and Delete would drop the panel's unsaved marks (Delete reloads the page), so while
// there are any they ask first; `then` runs on Discard them.
function unlessUnsaved(then) {
  if (!unsaved) {
    then();
    return;
  }
  discarding = then;
  $("lab-discard").innerHTML = `<p class="warn">${esc("This question has unsaved marks.")}</p>
    <div class="actions"><button type="button" class="danger" id="lab-discard-yes">Discard them</button>
      <button type="button" class="secondary" id="lab-discard-keep">Cancel</button></div>`;
  $("lab-discard").scrollIntoView({ block: "nearest" });
}

function closeDiscard() {
  discarding = null;
  $("lab-discard").innerHTML = "";
}

function say(id, text, cls = "err") {
  $(id).className = cls;
  $(id).textContent = text;
}

// Opens a question in the panel (null: a new, empty one) and, for a question, finds its passages.
function openQuestion(q) {
  closeDiscard();
  forgetAnswer();
  answering++; // a paid answer still under way is for what was here before
  $("lab-ai-answer").innerHTML = "";
  open = q;
  unsaved = false;
  $("lab-text").value = q?.text ?? "";
  $("lab-answer").value = q?.gold_answers[0] ?? "";
  $("lab-search").value = "";
  marks = new Map();
  for (const id of q?.judged_ids ?? []) marks.set(id, "n");
  for (const id of q?.gold_ids ?? []) marks.set(id, "y");
  shown = [];
  found = [];
  notes = [];
  finding++; // a Find still under way is for what was here before
  for (const id of ["lab-find-msg", "lab-search-msg", "lab-ai-msg"]) say(id, "");
  drawPanel();
  if (q) findPassages();
}

async function findPassages() {
  const question = $("lab-text").value.trim();
  if (!question) return;
  // The passages already marked come back even when the pool misses them.
  const stored = open ? [...open.gold_ids, ...(open.judged_ids ?? [])] : [];
  const labelled = [...new Set([...stored, ...marks.keys()])];
  const mine = ++finding;
  say("lab-find-msg", "");
  try {
    const pool = await api(`/eval-sets/${enc}/pool`, { method: "POST", body: { question, labelled } });
    if (mine !== finding) return;
    shown = [...pool.passages, ...pool.labelled, ...pool.missing.map((id) => ({ id }))];
    found = [];
    notes = pool.notes;
    drawPanel();
  } catch (e) {
    if (mine === finding) say("lab-find-msg", e.message);
  }
}

// Search the store: hybrid at 10, never Settings' method, which may be a paid one.
async function searchStore() {
  const question = $("lab-search").value.trim();
  if (!question) return;
  say("lab-search-msg", "");
  try {
    const reply = await api("/query", {
      method: "POST",
      body: { collection: docStore(), question, methods: ["hybrid"], k: 10 },
    });
    const result = reply.results.hybrid;
    if (result.error) say("lab-search-msg", result.error);
    const listed = new Set([...shown, ...found].map((p) => p.id));
    const before = found.length;
    for (const hit of result.hits) {
      if (listed.has(hit.id)) continue;
      const meta = hit.meta ?? {};
      found.push({ id: hit.id, text: hit.text, file: meta.file, part: meta.part, parts: meta.parts });
    }
    if (!result.error && found.length === before) {
      say("lab-search-msg", "Nothing new: every passage the search found is already listed.");
    }
    drawPanel();
  } catch (e) {
    say("lab-search-msg", e.message);
  }
}

function toggleMark(id, m) {
  if (restoring) return; // what the panel holds may be replaced when it lands
  forgetAnswer();
  if (marks.get(id) === m) marks.delete(id); // clicked again: back to not judged
  else marks.set(id, m);
  markUnsaved();
  drawPanel();
}

// -- the AI's answer --

// Any mark, text or question change drops a price not yet started: it priced what was here.
function forgetAnswer() {
  asking++;
  $("lab-ai-confirm").innerHTML = "";
}

// The AI's answer from the passages marked Right: its estimate first, one call only on Start.
async function askAnswer() {
  forgetAnswer();
  const mine = asking;
  const opened = answering;
  const passages = listedIds().filter((id) => marks.get(id) === "y");
  const body = { question: $("lab-text").value.trim(), passages, confirm: false };
  const path = `/eval-sets/${enc}/answer`;
  say("lab-ai-msg", "");
  try {
    const { estimate } = await api(path, { method: "POST", body });
    if (asking !== mine) return; // the panel changed while it was priced
    confirmSpend($("lab-ai-confirm"), estimate, async () => {
      // A save or Restore may reset the panel, dropping what was paid for; one call per confirm.
      if (busy() || paying) return;
      if (asking !== mine) {
        $("lab-ai-confirm").innerHTML = "";
        return;
      }
      paying = true;
      try {
        const shown = { confirm: true, calls: estimate.calls, est_usd: estimate.est_usd };
        const { answer } = await api(path, { method: "POST", body: { ...body, ...shown } });
        refreshSpend();
        if (asking === mine) $("lab-ai-confirm").innerHTML = "";
        if (answering === opened) showAnswer(answer); // a mark change keeps it; a question change doesn't
      } catch (e) {
        if (answering === opened) say("lab-ai-msg", e.message);
      } finally {
        paying = false;
        canSave();
      }
    });
    canSave(); // Start waits out a save or Restore under way
  } catch (e) {
    if (asking === mine) say("lab-ai-msg", e.message);
  }
}

function showAnswer(text) {
  const button = (m) => `<button type="button" class="secondary" data-ai-mark="${m}" aria-pressed="false">${LABEL[m]}</button>`;
  $("lab-ai-answer").innerHTML = `<p>Is this the right answer?</p><div class="full">${esc(text)}</div>
    <div class="actions">${button("y")}${button("n")}</div>`;
  canSave();
}

// y puts the answer in the reference answer field, n drops it; neither saves. An answer not
// yet taken is not unsaved: asking again is free from the cache.
function takeAnswer(m) {
  if (busy()) return; // a save or Restore is deciding what the panel holds
  const box = $("lab-ai-answer");
  if (m === "y") {
    $("lab-answer").value = box.querySelector(".full").textContent;
    markUnsaved();
    showRestore();
    say("lab-ai-msg", "Added as the reference answer. Save question to keep it.", "ok");
  }
  box.innerHTML = "";
}

async function saveQuestion(next) {
  if (busy()) return;
  closeDiscard(); // what it would discard is being saved
  saving = true;
  canSave(); // set before the first await, so a second click finds the buttons disabled
  try {
    await writeQuestion(next);
  } finally {
    saving = false;
    canSave();
  }
}

async function writeQuestion(next) {
  say("lab-msg", "");
  const ids = listedIds();
  const gold_ids = ids.filter((id) => marks.get(id) === "y");
  const judged_ids = ids.filter((id) => marks.has(id));
  const typed = $("lab-answer").value.trim();
  const rest = (open?.gold_answers ?? []).slice(1); // the other accepted answers, as they were
  const gold_answers = typed ? [typed, ...rest] : rest;
  const text = $("lab-text").value.trim();
  const before = open; // what is saved; the panel is rebound only while it still holds it
  unsaved = false; // what is in the panel now is being saved; a change from here on is not
  try {
    let id = open?.id;
    let version;
    if (open) {
      const question = { ...open, text, gold_answers, gold_ids, judged_ids };
      ({ version } = await api(`/eval-sets/${enc}/questions/${encodeURIComponent(id)}`, {
        method: "PUT",
        body: { question, base: open },
      }));
    } else {
      ({ id, version } = await api(`/eval-sets/${enc}/questions`, {
        method: "POST",
        body: { question: { text, gold_answers, gold_ids, judged_ids } },
      }));
    }
    await load();
    if (open === before) {
      if (next) {
        openQuestion(null);
        $("lab-text").focus();
      } else {
        open = set.questions.find((q) => q.id === id) ?? null; // the next Save is checked against it
      }
    }
    say("lab-msg", `Saved as version ${short(version)}.`, "ok");
    say("lab-ai-msg", ""); // "Save question to keep it" reads wrong once it is kept
  } catch (e) {
    say("lab-msg", e.message); // every mark stays where it is
    markUnsaved();
    if (e.status === 404 && open === before && before) {
      // Deleted meanwhile: its row goes, and the next Save writes these marks as a new question.
      try {
        await load();
      } catch {
        // the detail already shown is the one that matters
      }
      if (open === before) open = null;
    }
    if (e.status === 409 && open) {
      $("lab-msg").insertAdjacentHTML("beforeend",
        ' <button type="button" id="lab-reload" class="secondary">Reload it (discard my marks)</button>');
    }
  }
}

async function reloadQuestion() {
  const id = open?.id;
  say("lab-msg", "");
  try {
    await load();
    openQuestion(set.questions.find((q) => q.id === id) ?? null);
  } catch (e) {
    say("lab-msg", e.message);
  }
}

function bindLabelling() {
  $("lab-store").addEventListener("click", async (e) => {
    if (!e.target.closest("#lab-link")) return;
    say("lab-link-msg", "");
    try {
      await api(`/eval-sets/${enc}`, { method: "PATCH", body: { store: $("lab-pick").value } });
      await load();
    } catch (err) {
      say("lab-link-msg", err.message);
    }
  });
  $("lab-questions").addEventListener("click", (e) => {
    const button = e.target.closest("button");
    if (!button) return;
    const d = button.dataset;
    if (d.label !== undefined) {
      unlessUnsaved(() => {
        $("lab-msg").textContent = "";
        openQuestion(set.questions.find((q) => q.id === d.label) ?? null);
        $("lab-panel").scrollIntoView({ block: "start" });
      });
    }
    if (d.del !== undefined) {
      unlessUnsaved(() => askDelete($("lab-del"), {
        question: "Delete this question and its marks?",
        yes: "Yes, delete it",
        path: `/eval-sets/${enc}/questions/${encodeURIComponent(d.del)}`,
        gone: `/e/${enc}`,
      }));
    }
  });
  $("lab-discard").addEventListener("click", (e) => {
    if (e.target.closest("#lab-discard-keep")) closeDiscard();
    if (e.target.closest("#lab-discard-yes")) {
      const then = discarding;
      closeDiscard();
      then?.();
    }
  });
  $("lab-text").addEventListener("input", () => {
    markUnsaved();
    forgetAnswer();
    canSave();
  });
  $("lab-answer").addEventListener("input", () => {
    markUnsaved();
    showRestore();
    say("lab-ai-msg", ""); // edited by hand, the field no longer holds the AI's answer
  });
  $("lab-find").addEventListener("click", findPassages);
  $("lab-search-go").addEventListener("click", searchStore);
  $("lab-search").addEventListener("keydown", (e) => {
    if (e.key === "Enter") searchStore();
  });
  $("lab-panel").addEventListener("click", (e) => {
    const button = e.target.closest("button[data-mark]");
    if (button) toggleMark(button.closest("li[data-passage]").dataset.passage, button.dataset.mark);
  });
  $("lab-ai").addEventListener("click", askAnswer);
  $("lab-ai-answer").addEventListener("click", (e) => {
    const button = e.target.closest("button[data-ai-mark]");
    if (button) takeAnswer(button.dataset.aiMark);
  });
  $("lab-save").addEventListener("click", () => saveQuestion(false));
  $("lab-next").addEventListener("click", () => saveQuestion(true));
  $("lab-msg").addEventListener("click", (e) => {
    if (e.target.closest("#lab-reload")) reloadQuestion();
  });
}

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
    await drawLabelling();
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
    importBox($("imp-questions"), {
      uploadPath: `/eval-sets/${enc}/uploads`,
      importPath: `/eval-sets/${enc}/imports`,
      kind: "questions",
      qaType: set.qa_type,
      typeLabel: cat.qa_types[set.qa_type]?.label,
      done: () => (dirty ? null : load()), // edits made while it ran are kept; Save then says it moved on
    });
  } catch (e) {
    $("head").innerHTML = head({ title: name, crumb: [{ label: "Eval sets", href: "/eval-sets" }] }) +
      `<p class="err">${esc(e.message)}</p>`;
  }
}

bindTable();
bindLeaked();
bindDelete();
bindLabelling();
bindVersions();
start();
