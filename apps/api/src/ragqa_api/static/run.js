import { api, catalog, clock, confirmSpend, esc, gapBar, head, methodLabel, metricLabel, pct, pill,
  pts, refreshSpend, shell, verdictText } from "/static/ui.js";

const POLL_MS = 3000;
const BUSY = new Set(["queued", "running"]);
const RESUMABLE = new Set(["interrupted", "cancelled", "stopped_budget", "stopped_no_key", "failed"]);
const STATE_TEXT = { ahead: "ahead", cant_tell: "can't tell", behind: "behind" };

const parts = location.pathname.split("/");
const name = decodeURIComponent(parts[2] ?? "");
const id = decodeURIComponent(parts[4] ?? "");
const enc = encodeURIComponent(name);
const $ = (x) => document.getElementById(x);

let cat = null;
let run = null; // the latest GET /eval/{id}
let resumeProbe = null; // for a resumable run: {refusal} or {} when it may resume
let probedStatus = null; // the status resumeProbe was computed for; a new status probes again
let qaType = null; // the collection's, fetched once, for metric labels
const painted = {}; // section id -> html last drawn, so a refresh never rebuilds what has not changed

shell(`c/${name}`);

function paint(section, html) {
  if (painted[section] === html) return false;
  painted[section] = html;
  $(section).innerHTML = html;
  return true;
}

const plural = (n, noun) => `${n} ${noun}${n === 1 ? "" : "s"}`;
const metric = (m) => metricLabel(m, qaType, cat);
const method = (m) => methodLabel(m, cat);
const isControl = (m) => cat.methods[m]?.group === "control";
const show = (v) => (Array.isArray(v) ? v.join(", ") : v == null || v === "" ? "–" : String(v));
const range = (m) => (m ? `${pct(m.lo)}–${pct(m.hi)}` : "");

// -- head and answer ---------------------------------------------------------------------

function headHtml() {
  const s = run.setup;
  const model = s.answer ? s.model : "retrieval only";
  const progress = run.status === "running" ? ` ${pct(run.progress)}` : "";
  const failed = run.error ? `<p class="err">${esc(run.error)}</p>` : "";
  return head({
    title: `Run ${id.slice(0, 4)}`,
    crumb: [{ label: "Collections", href: "/" }, { label: name, href: `/c/${enc}` }, "Runs"],
    aside: [clock(run.created_at), model, plural(run.question_count, "question"),
      `$${Number(run.usd_total ?? 0).toFixed(2)}`].map(esc).join(" · ") + ` · ${pill(run.status)}${esc(progress)}`,
  }) + failed;
}

function answerHtml() {
  const a = run.answer;
  if (!a) {
    const [cand, ref] = cat.answer_pair;
    return `<p class="verdict"><strong>Do keywords help?</strong> This run does not include ${esc(
      method(cand))} and ${esc(method(ref))}, so it gives no answer.</p>`;
  }
  return `<p class="verdict"><strong>Do keywords help?</strong> ${esc(verdictText({ ...a, proxy_trusted: run.proxy_trusted }, cat))}, on the same ${esc(
    plural(run.question_count, "question"))}.</p>`;
}

// -- the gap chart ------------------------------------------------------------------------

// The reference's own score; reasoning match is rescaled against the run's random example.
function referenceScore() {
  const mean = run.methods[run.reference]?.[run.headline]?.mean;
  if (!Number.isFinite(mean)) return "n/a";
  if (run.scale.unit !== "match") return pct(mean);
  const chance = run.methods.random_example?.[run.headline]?.mean;
  if (!Number.isFinite(chance)) return `${pct(mean)} (no chance level in this run)`;
  return `${Math.round((mean - chance) * run.scale.factor)} match`;
}

function gapRow(g) {
  return `<tr data-gap-row><td>${esc(method(g.method))}<code>${esc(g.method)}</code></td>
    <td>${gapBar(g.lo, g.diff, g.hi, span())}</td><td class="num">${esc(pts(g.diff, g.lo, g.hi))}</td>
    <td>${esc(STATE_TEXT[g.state] ?? g.state)}</td></tr>`;
}

function span() {
  const biggest = Math.max(0, ...run.gaps.flatMap((g) => [g.lo, g.hi, g.diff].map(Math.abs)).filter(Number.isFinite));
  return Math.max(10, Math.ceil(biggest / 5) * 5);
}

function gapsHtml() {
  if (!run.gaps.length) {
    return `<p class="muted">No gaps to draw: this run has no ${esc(method(run.reference))} to compare against.</p>`;
  }
  const real = run.gaps.filter((g) => !isControl(g.method));
  const controls = run.gaps.filter((g) => isControl(g.method));
  const behind = real.findIndex((g) => g.diff < 0);
  const at = behind < 0 ? real.length : behind;
  const reference = `<tr data-ref-row><td>${esc(method(run.reference))}<code>${esc(run.reference)}</code>
    (reference)</td><td></td><td class="num">${esc(referenceScore())}</td><td></td></tr>`;
  const rows = [...real.slice(0, at).map(gapRow), reference, ...real.slice(at).map(gapRow)];
  if (controls.length) {
    rows.push('<tr class="rule"><td colspan="4">chance level</td></tr>', ...controls.map(gapRow));
  }
  return `<p>${esc(metric(run.headline))}, compared with ${esc(method(run.reference))}</p>
    <table><tbody>${rows.join("")}</tbody></table>`;
}

function bestHtml() {
  const b = run.best_of_each;
  if (!b) return "";
  return `<p>Best keyword method against the best other: ${esc(method(b.candidate))} ${esc(
    pts(b.diff, b.lo, b.hi))} vs ${esc(method(b.reference))}.
    <span class="muted">Picked after the fact, so the gap is likely overstated.</span></p>`;
}

// -- the sections under the chart -------------------------------------------------------------

function scoresHtml() {
  const names = [...new Set(Object.values(run.methods).flatMap((m) => Object.keys(m)))].filter(
    (k) => k !== "usd" && k !== "n"
  );
  const heads = names.map((m) => `<th>${esc(metric(m))}</th>`).join("");
  const rows = Object.entries(run.methods)
    .map(([m, row]) => {
      const cells = names
        .map((k) => (row[k] ? `<td class="num">${esc(pct(row[k].mean))}
          <span class="muted">${esc(range(row[k]))}</span></td>` : '<td class="num muted">n/a</td>'))
        .join("");
      return `<tr><td>${esc(method(m))}<code>${esc(m)}</code></td>${cells}</tr>`;
    })
    .join("");
  return `<table><thead><tr><th>Method</th>${heads}</tr></thead><tbody>${rows}</tbody></table>`;
}

function proxyHtml() {
  const p = run.proxy;
  if (p) {
    const verdict = p.trusted ? "Probably yes" : "No";
    return `<p>${verdict}: the free score ${p.trusted ? "ranks" : "does not rank"} methods like the answered
      score (checked on ${esc(plural(p.methods, "method"))}). With so few methods this is a hint, not proof.</p>`;
  }
  if (run.proxy_trusted != null) {
    return `<p>From this collection's latest answered run, the free score is
      ${run.proxy_trusted ? "" : "not "}trusted.</p>`;
  }
  return '<p class="muted">Not checked here: it needs an answered run of at least 4 retrieval methods.</p>';
}

const x100 = (v) => (v == null ? NaN : v * 100); // a NaN-cleaned null stays missing, not 0

function pairsHtml() {
  if (!run.pairs.length) return '<p class="muted">Fewer than two methods, so nothing to compare.</p>';
  const rows = run.pairs
    .map((p) => {
      const result = !p.separated ? "can't tell" : p.diff > 0 ? `${method(p.a)} ahead` : `${method(p.b)} ahead`;
      return `<tr><td>${esc(metric(p.metric))}</td><td>${esc(method(p.a))} vs ${esc(method(p.b))}</td>
        <td class="num">${esc(pts(x100(p.diff), x100(p.lo), x100(p.hi)))}</td><td>${esc(result)}</td></tr>`;
    })
    .join("");
  return `<table><thead><tr><th>Measure</th><th>Pair</th><th>Gap</th><th>Result</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

function setupHtml() {
  const rows = Object.entries(run.setup)
    .map(([k, v]) => {
      const text = k === "question_ids" ? plural(v.length, "question") : Array.isArray(v) ? v.join(", ") : show(v);
      return `<tr><td>${esc(k)}</td><td>${esc(text)}</td></tr>`;
    })
    .join("");
  return `<table><tbody>${rows}</tbody></table>`;
}

// -- questions --------------------------------------------------------------------------------

function questionHtml(row) {
  const a = row.answer;
  const right = a ? (a.accuracy ?? a.em) === 1 : null;
  const mark = right == null ? "" : right ? "right " : "wrong ";
  const choices = (row.choices ?? []).map((c, i) => `${String.fromCharCode(65 + i)}. ${c}`);
  const hits = (row.hit_ids ?? [])
    .map((h, i) => {
      const skill = row.hit_skills?.[i];
      const words = show(row.hit_matched?.[i]); // rows from before hit_matched lack it
      return `<li>${esc(skill ? `${h} (${skill})` : h)}, matched keywords:
        <span data-matched>${esc(words)}</span></li>`;
    })
    .join("");
  return `<details><summary>${esc(mark)}${esc(row.question ?? row.question_id)}</summary>
    ${choices.length ? `<p>${choices.map(esc).join("<br>")}</p>` : ""}
    <p>Gold: ${esc(show(row.gold_answers?.length ? row.gold_answers : row.gold_ids))}</p>
    <p>Parsed answer: ${esc(show(row.pred))} <span class="muted">${esc(show(row.raw))}</span></p>
    <p>Top hits:</p><ul>${hits || "<li>–</li>"}</ul></details>`;
}

async function loadQuestions() {
  const m = $("q-method").value;
  if (!m) return;
  try {
    const rows = await api(`/eval/${encodeURIComponent(id)}/items?method=${encodeURIComponent(m)}&wrong=${
      $("q-wrong").checked}`);
    $("questions").innerHTML = rows.length ? rows.map(questionHtml).join("") : '<p class="muted">None.</p>';
  } catch (e) {
    $("questions").innerHTML = `<p class="err">${esc(e.message)}</p>`;
  }
}

function fillMethodPicker() {
  if ($("q-method").options.length) return;
  $("q-method").innerHTML = run.setup.methods
    .map((m) => `<option value="${esc(m)}">${esc(method(m))}</option>`)
    .join("");
}

// -- actions ------------------------------------------------------------------------------------

function actionsHtml() {
  const s = run.status;
  const buttons = [];
  if (BUSY.has(s)) buttons.push('<button type="button" id="act-cancel">Cancel</button>');
  if (RESUMABLE.has(s) && resumeProbe?.refusal) {
    buttons.push(`<span class="warn" id="act-refusal">${esc(resumeProbe.refusal)}</span>`);
  } else if (RESUMABLE.has(s)) {
    buttons.push('<button type="button" id="act-resume">Resume</button>');
  }
  buttons.push('<button type="button" id="act-rerun" class="secondary">Re-run</button>');
  buttons.push('<button type="button" id="act-delete" class="secondary">Delete…</button>');
  return `<div class="actions">${buttons.join("")}</div>
    <div id="act-confirm"></div><p id="act-msg" class="err"></p>`;
}

const say = (text) => ($("act-msg").textContent = text);

// Free: without confirm the endpoint only estimates, or refuses with the reason in `detail`.
async function probeResume() {
  try {
    await api(`/jobs/${encodeURIComponent(id)}/resume`, { method: "POST", body: { confirm: false } });
    resumeProbe = {};
  } catch (e) {
    resumeProbe = { refusal: e.message };
  }
}

async function cancel() {
  say("");
  try {
    await api(`/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" });
    await refresh();
  } catch (e) {
    say(e.message);
  }
}

async function resume() {
  say("");
  const path = `/jobs/${encodeURIComponent(id)}/resume`;
  try {
    const { estimate } = await api(path, { method: "POST", body: { confirm: false } });
    confirmSpend($("act-confirm"), estimate, async () => {
      try {
        await api(path, { method: "POST", body: { confirm: true } });
        $("act-confirm").innerHTML = "";
        refreshSpend();
        await refresh();
      } catch (e) {
        say(e.message);
      }
    });
  } catch (e) {
    say(e.message); // the refusal text
  }
}

async function rerun() {
  say("");
  const { collection, split, methods, answer, model, k, rrf_k } = run.setup;
  const request = { collection, split, methods, answer, model, k, rrf_k };
  try {
    const { estimate } = await api("/eval", { method: "POST", body: { ...request, confirm: false } });
    confirmSpend($("act-confirm"), estimate, async () => {
      try {
        const queued = await api("/eval", { method: "POST", body: { ...request, confirm: true } });
        refreshSpend();
        location.href = `/c/${enc}/runs/${encodeURIComponent(queued.job_id)}`;
      } catch (e) {
        say(e.message);
      }
    });
  } catch (e) {
    say(e.message);
  }
}

function askDelete() {
  say("");
  $("act-confirm").innerHTML = `<p>Delete this run and its results? This cannot be undone.</p>
    <div class="actions"><button type="button" id="del-yes" class="danger">Yes, delete this run</button>
    <button type="button" id="del-no" class="secondary">Keep it</button></div>`;
  $("del-no").addEventListener("click", () => ($("act-confirm").innerHTML = ""));
  $("del-yes").addEventListener("click", async () => {
    try {
      await api(`/eval/${encodeURIComponent(id)}`, { method: "DELETE" });
      location.href = `/c/${enc}`;
    } catch (e) {
      $("act-confirm").innerHTML = "";
      say(e.message);
    }
  });
}

function bindActions() {
  $("act-cancel")?.addEventListener("click", cancel);
  $("act-resume")?.addEventListener("click", resume);
  $("act-rerun").addEventListener("click", rerun);
  $("act-delete").addEventListener("click", askDelete);
}

// -- refresh --------------------------------------------------------------------------------------

let timer = null;
async function refresh() {
  clearTimeout(timer);
  try {
    cat = await catalog();
    run = await api(`/eval/${encodeURIComponent(id)}`);
    qaType ??= (await api(`/collections/${encodeURIComponent(run.setup.collection)}`)).qa_type;
    if (run.status !== probedStatus) resumeProbe = null;
    if (RESUMABLE.has(run.status) && resumeProbe === null) {
      await probeResume();
      probedStatus = run.status;
    }
    $("head").innerHTML = headHtml();
    paint("answer", answerHtml());
    paint("gaps", gapsHtml());
    $("note").textContent = run.gaps.length
      ? "Leading is not winning: the order above is this run's, and the declared pair is the only comparison the answer rests on."
      : "";
    paint("best", bestHtml());
    paint("scores", scoresHtml());
    paint("proxy", proxyHtml());
    paint("pairs", pairsHtml());
    paint("setup", setupHtml());
    fillMethodPicker();
    if (paint("actions", actionsHtml())) bindActions();
  } catch (e) {
    $("head").innerHTML = head({ title: "Run", crumb: [{ label: "Collections", href: "/" }] }) +
      `<p class="err">${esc(e.message)}</p>`;
  }
  if (run && BUSY.has(run.status)) timer = setTimeout(refresh, POLL_MS);
}

$("q-method").addEventListener("change", loadQuestions);
$("q-wrong").addEventListener("change", loadQuestions);
$("questions-box").addEventListener("toggle", () => {
  if ($("questions-box").open) loadQuestions();
});
refresh();
