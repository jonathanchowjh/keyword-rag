import { $, api, catalog, clock, esc, head, plural, shell, short } from "/static/ui.js";

let cat = null;
let linked = null; // the store the open form's set is made for (Create one on a store page), or null

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

// The new-set form. For a store (`store`), it is named after it and takes its type; otherwise a
// Documents set may pick a Documents store to label against.
async function showForm(store = null) {
  let stores;
  try {
    [cat, stores] = await Promise.all([catalog(), api("/collections")]);
  } catch (e) {
    $("form").hidden = false;
    $("form").innerHTML = `<p class="err">${esc(e.message)}</p>`;
    return;
  }
  linked = store;
  const own = stores.find((c) => c.name === store);
  const types = Object.entries(cat.qa_types)
    .map(([id, t]) => `<option value="${esc(id)}"${own?.qa_type === id ? " selected" : ""}>${esc(t.label)}</option>`)
    .join("");
  const documents = stores.filter((c) => c.qa_type === "documents").map((c) => c.name).sort();
  const pick = ['<option value="">(none)</option>']
    .concat(documents.map((s) => `<option value="${esc(s)}">${esc(s)}</option>`))
    .join("");
  const link = store
    ? `<p>${esc(`Linked store: ${store}`)}</p>`
    : `<label id="f-store-wrap" hidden>Store to label against<select id="f-store">${pick}</select></label>`;
  $("form").hidden = false;
  $("form").innerHTML = `<div class="card form">
    <label>Name<input id="f-name" autocomplete="off" value="${esc(store ?? "")}"></label>
    <label>Type<select id="f-type"${own ? " disabled" : ""}>${types}</select></label>
    ${link}
    <div class="row"><button type="button" id="f-create">Create</button>
      <button type="button" id="f-cancel" class="secondary">Cancel</button>
      <span id="f-msg" class="err" role="status"></span></div></div>`;
  const showPick = () => {
    if ($("f-store-wrap")) $("f-store-wrap").hidden = $("f-type").value !== "documents";
  };
  showPick();
  $("f-type").addEventListener("change", showPick);
  $("f-cancel").addEventListener("click", () => ($("form").hidden = true));
  $("f-create").addEventListener("click", create);
}

async function create() {
  const name = $("f-name").value.trim();
  const qa_type = $("f-type").value;
  const store = linked ?? (qa_type === "documents" ? $("f-store").value || null : null);
  try {
    await api("/eval-sets", { method: "POST", body: { name, qa_type, ...(store ? { store } : {}) } });
    if (store) {
      location.href = `/e/${encodeURIComponent(name)}`; // labelled against its store: open it
      return;
    }
    $("form").hidden = true;
    await refresh();
  } catch (e) {
    $("f-msg").textContent = e.message;
  }
}

$("new").addEventListener("click", () => showForm());
refresh();
const asked = new URLSearchParams(location.search).get("store"); // ?store=: Create one on a store page
if (asked) showForm(asked);
