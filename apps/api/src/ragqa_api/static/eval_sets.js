import { $, api, catalog, clock, esc, head, plural, shell, short } from "/static/ui.js";

let cat = null;

shell("eval-sets");
$("head").innerHTML = head({
  title: "Eval sets",
  sub: "The questions a run asks. One set can be run against any store of the same type.",
});

function rowHtml(e) {
  const type = cat.qa_types[e.qa_type];
  const leaked = e.include_leaked ? " · leaked questions included" : "";
  return `<a class="row-link" href="/e/${encodeURIComponent(e.name)}">
    <div><strong>${esc(e.name)}</strong> · ${esc(type?.label ?? e.qa_type)}</div>
    <div>${esc(plural(e.questions, "question"))}${esc(leaked)}</div>
    <div class="small">version ${esc(short(e.version))} · ${esc(clock(e.created_at))}</div></a>`;
}

async function refresh() {
  try {
    const [sets] = await Promise.all([api("/eval-sets"), catalog().then((c) => (cat = c))]);
    $("list").innerHTML = sets.length
      ? `<div class="table"><div class="th"><div>Eval set</div><div>Questions</div><div>Version</div></div>${sets
          .map(rowHtml)
          .join("")}</div>`
      : '<p class="muted">No eval sets yet. Create one, then add questions or import a file.</p>';
  } catch (e) {
    $("list").innerHTML = `<p class="err">${esc(e.message)}</p>`;
  }
}

async function showForm() {
  cat = await catalog();
  const types = Object.entries(cat.qa_types)
    .map(([id, t]) => `<option value="${esc(id)}">${esc(t.label)}</option>`)
    .join("");
  $("form").hidden = false;
  $("form").innerHTML = `<div class="card form">
    <label>Name<input id="f-name" autocomplete="off"></label>
    <label>Type<select id="f-type">${types}</select></label>
    <div class="row"><button type="button" id="f-create">Create</button>
      <button type="button" id="f-cancel" class="secondary">Cancel</button>
      <span id="f-msg" class="err" role="status"></span></div></div>`;
  $("f-cancel").addEventListener("click", () => ($("form").hidden = true));
  $("f-create").addEventListener("click", create);
}

async function create() {
  const name = $("f-name").value.trim();
  try {
    await api("/eval-sets", { method: "POST", body: { name, qa_type: $("f-type").value } });
    $("form").hidden = true;
    await refresh();
  } catch (e) {
    $("f-msg").textContent = e.message;
  }
}

$("new").addEventListener("click", showForm);
refresh();
