import { $, BUSY, NO_CHANCE, RESUMABLE, api, askDelete, askResume, axisHtml, between, catalog, clock,
  confirmSpend, esc, gapBar, gapSpan, head, legendHtml, methodLabel, metricLabel, metricSays, num, paint, pct,
  pill, plural, pts, refreshSpend, shell, short, usd, verdictNotes, verdictRest, wordHtml } from "/static/ui.js";

const POLL_MS = 3000;
const STATE_TEXT = { ahead: "ahead", cant_tell: "can't tell", behind: "behind" };

const parts = location.pathname.split("/");
const name = decodeURIComponent(parts[2] ?? "");
const id = decodeURIComponent(parts[4] ?? "");
const enc = encodeURIComponent(name);

let cat = null;
let run = null; // the latest GET /eval/{id}
let resumeProbe = null; // for a resumable run: {refusal} or {} when it may resume
let probedStatus = null; // the status resumeProbe was computed for; a new status probes again
let qaType = null; // the collection's, fetched once, for metric labels

shell(`c/${name}`);

const metric = (m) => metricLabel(m, qaType, cat);
const says = (m) => metricSays(m, qaType, cat);
const method = (m) => methodLabel(m, cat);
const isControl = (m) => cat.methods[m]?.group === "control";
const show = (v) => (Array.isArray(v) ? v.join(" · ") : v == null || v === "" ? "–" : String(v));
const busy = () => BUSY.has(run.status);
// The headline is the reasoning match shown raw: this run has no random example to set 0 by.
const rawScale = () => run.scale?.unit === "raw";

// -- head and answer ---------------------------------------------------------------------

function headHtml() {
  const s = run.setup;
  const model = s.answer ? s.model : "retrieval only";
  const progress = run.status === "running" ? ` ${pct(run.progress)}` : "";
  const since = run.since.untracked
    ? ' · <span class="muted">recorded before settings were tracked</span>'
    : run.since.changes.map((c) => ` · <span class="warn">${esc(c)} changed after this run</span>`).join("");
  const failed = run.error ? `<p class="err">${esc(run.error)}</p>` : "";
  // A run that has not started has asked no questions yet: "–", not "0".
  const asked = busy() && !run.question_count ? "– questions" : plural(run.question_count, "question");
  return head({
    title: `Run ${short(id)}`,
    crumb: [{ label: "Collections", href: "/" }, { label: name, href: `/c/${enc}` }, "Runs"],
    aside: [clock(run.created_at), model, asked, usd(run.usd_total)].map(esc).join(" · ") +
      ` · ${pill(run.status)}${esc(progress)}${since}`,
  }) + failed;
}

function answerHtml() {
  const lead = "<strong>Do keywords help?</strong>";
  if (busy()) return `<p class="verdict">${lead} The answer appears when the run finishes.</p>`;
  const a = run.answer;
  if (!a) {
    const missing = cat.answer_pair.filter((m) => !run.setup.methods.includes(m));
    const why = missing.length
      ? `This run does not include ${missing.map(method).join(" and ")}`
      : `This run has no results for ${cat.answer_pair.map(method).join(" and ")} to compare`;
    return `<p class="verdict">${lead} ${esc(why)}, so it gives no answer.</p>`;
  }
  const notes = verdictNotes({ ...a, proxy_trusted: run.proxy_trusted }, { raw: rawScale() })
    .map((n) => `${n.charAt(0).toUpperCase()}${n.slice(1)}.`)
    .join(" ");
  return `<p class="verdict">${lead} ${wordHtml(a)}: ${esc(verdictRest(a, qaType, cat))}, on the same ${esc(
    plural(run.question_count, "question"))}.</p>
    ${notes ? `<p class="muted">${esc(notes)}</p>` : ""}
    ${legendHtml("Ahead, behind and can't tell in the table below read the same way.")}`;
}

// -- the gap chart ------------------------------------------------------------------------

// Reasoning match is shown above the random example's: (score - random) / (1 - random) x 100,
// the scale verdict.scale gives the gaps. Null when this run has no random example to judge by.
const SIM = "solution_sim@1";
function matchFactor() {
  const chance = run.methods.random_example?.[SIM]?.mean;
  return Number.isFinite(chance) && chance < 1 ? 100 / (1 - chance) : null;
}

// One cell of the own-score table: the mean and its range, true minus, "n/a" when missing.
// What a metric means in this run: without a random example the reasoning match is raw.
function meaning(m) {
  return m === SIM && matchFactor() === null
    ? `raw similarity of the solutions; this run has no random example, so ${NO_CHANCE}`
    : says(m);
}

function ownScore(m, k) {
  const cell = m[k];
  if (!Number.isFinite(cell?.mean)) return '<span class="muted">n/a</span>';
  // A metric measured on fewer questions than the run says so; the rest could not be measured.
  const part = cell.n < run.question_count ? ` <span class="muted">on ${cell.n} of ${run.question_count} questions</span>` : "";
  return scoreText(cell, k) + part;
}

function scoreText(cell, k) {
  const times = (v, f, less = 0) => (Number.isFinite(v) ? (v - less) * f : NaN);
  if (k !== SIM) {
    return `${esc(pct(cell.mean))} <span class="muted">${esc(between(times(cell.lo, 100), times(cell.hi, 100)))}</span>`;
  }
  const f = matchFactor();
  if (f === null) {
    return `${esc(pct(cell.mean))} <span class="muted">${esc(between(times(cell.lo, 100), times(cell.hi, 100)))} · ${NO_CHANCE}</span>`;
  }
  const chance = run.methods.random_example[SIM].mean;
  const at = (v) => times(v, f, chance);
  return `${esc(num(at(cell.mean)))} match <span class="muted">${esc(between(at(cell.lo), at(cell.hi)))}</span>`;
}

// The reference's own score; reasoning match is rescaled against the run's random example.
function referenceScore() {
  const mean = run.methods[run.reference]?.[run.headline]?.mean;
  if (!Number.isFinite(mean)) return "n/a";
  if (run.scale.unit !== "match") return rawScale() ? `${pct(mean)} (${NO_CHANCE} in this run)` : pct(mean);
  const chance = run.methods.random_example?.[run.headline]?.mean;
  if (!Number.isFinite(chance)) return `${pct(mean)} (${NO_CHANCE} in this run)`;
  return `${num((mean - chance) * run.scale.factor)} match`;
}

function gapRow(g, span) {
  const raw = rawScale() ? ` <span class="muted">${NO_CHANCE}</span>` : "";
  return `<tr data-gap-row><td>${esc(method(g.method))}<code>${esc(g.method)}</code></td>
    <td>${gapBar(g.lo, g.diff, g.hi, span)}</td><td class="num">${esc(pts(g.diff, g.lo, g.hi))}${raw}</td>
    <td>${esc(STATE_TEXT[g.state] ?? g.state)}</td></tr>`;
}

function gapsHtml() {
  if (busy()) return '<p class="muted">The gaps appear when the run finishes.</p>';
  const ref = method(run.reference);
  if (!run.gaps.length) {
    const why = run.setup.methods.includes(run.reference)
      ? `no other method has results to set against ${ref}`
      : `this run has no ${ref} to compare against`;
    return `<p class="muted">No gaps to draw: ${esc(why)}.</p>`;
  }
  const span = gapSpan(run.gaps.flatMap((g) => [g.lo, g.hi, g.diff]));
  const row = (g) => gapRow(g, span);
  const real = run.gaps.filter((g) => !isControl(g.method));
  const controls = run.gaps.filter((g) => isControl(g.method));
  const behind = real.findIndex((g) => g.diff < 0);
  const at = behind < 0 ? real.length : behind;
  const reference = `<tr data-ref-row><td>${esc(ref)}<code>${esc(run.reference)}</code>
    (reference)</td><td></td><td class="num">${esc(referenceScore())}</td><td></td></tr>`;
  const rows = [...real.slice(0, at).map(row), reference, ...real.slice(at).map(row)];
  if (controls.length) {
    rows.push('<tr class="rule"><td colspan="4">chance level</td></tr>', ...controls.map(row));
  }
  const raw = rawScale() ? ` (shown raw, ${NO_CHANCE})` : "";
  const meaning = rawScale()
    ? "This run has no random example, so the score is raw similarity with no chance level to set 0 by."
    : says(run.headline);
  return `<p>${esc(metric(run.headline) + raw)}, compared with ${esc(ref)}</p>
    ${meaning ? `<p class="muted">${esc(meaning)}</p>` : ""}
    <table><thead><tr><th>Method</th><th>${axisHtml(span)}</th><th class="num">Gap</th><th>Result</th></tr></thead>
    <tbody>${rows.join("")}</tbody></table>`;
}

function bestHtml() {
  const b = run.best_of_each;
  if (!b || busy()) return "";
  return `<p>Best keyword method against the best other: ${esc(method(b.candidate))} ${esc(
    pts(b.diff, b.lo, b.hi))} vs ${esc(method(b.reference))}.
    <span class="muted">Picked after the fact, so the gap is likely overstated.</span></p>`;
}

// -- the sections under the chart -------------------------------------------------------------

function scoresHtml() {
  const names = [...new Set(Object.values(run.methods).flatMap((m) => Object.keys(m)))].filter(
    (k) => k !== "usd" && k !== "n"
  );
  const heads = names.map((m) => `<th title="${esc(meaning(m))}">${esc(metric(m))}</th>`).join("");
  const rows = Object.entries(run.methods)
    .map(([m, row]) => {
      const cells = names
        .map((k) => (row[k] ? `<td class="num">${ownScore(row, k)}</td>` : '<td class="num muted">n/a</td>'))
        .join("");
      return `<tr><td>${esc(method(m))}<code>${esc(m)}</code></td>${cells}</tr>`;
    })
    .join("");
  const meanings = names
    .filter((m) => meaning(m))
    .map((m) => `<p class="muted legend">${esc(metric(m))}: ${esc(meaning(m))}.</p>`)
    .join("");
  return `<table><thead><tr><th>Method</th>${heads}</tr></thead><tbody>${rows}</tbody></table>${meanings}`;
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

function pairsHtml() {
  if (!run.pairs.length) return '<p class="muted">Fewer than two methods, so nothing to compare.</p>';
  const rows = run.pairs
    .map((p) => {
      const result = !p.separated ? "can't tell" : p.diff > 0 ? `${method(p.a)} ahead` : `${method(p.b)} ahead`;
      const f = p.metric === SIM ? matchFactor() : null; // null: shown as it is
      const raw = p.metric === SIM && f === null ? ` (${NO_CHANCE})` : "";
      const k = f ?? 100;
      const scaled = (v) => (v == null ? NaN : v * k);
      return `<tr><td title="${esc(meaning(p.metric))}">${esc(metric(p.metric) + raw)}</td><td>${esc(method(p.a))} vs ${esc(method(p.b))}</td>
        <td class="num">${esc(pts(scaled(p.diff), scaled(p.lo), scaled(p.hi)))}</td><td>${esc(result)}</td></tr>`;
    })
    .join("");
  return `<table><thead><tr><th>Measure</th><th>Pair</th><th>Gap</th><th>Result</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

// What the run left out; setups recorded before these fields lack them.
const LEFT_OUT = ["disabled_sources", "evidence_off", "leaked_excluded"];

function leftOutHtml() {
  const s = run.setup;
  const notes = [];
  if (s.disabled_sources?.length) notes.push(`Retrieval left out: ${s.disabled_sources.join(", ")}`);
  if (s.evidence_off > 0) notes.push(`${plural(s.evidence_off, "question")}' evidence is switched off`);
  if (s.leaked_excluded > 0) notes.push(`${plural(s.leaked_excluded, "leaked question")} left out`);
  return notes.map((n) => `<p class="warn">${esc(n)}</p>`).join("");
}

function setupHtml() {
  const rows = Object.entries(run.setup)
    .filter(([k]) => !LEFT_OUT.includes(k))
    .map(([k, v]) => {
      const text = k === "question_ids" ? plural(v.length, "question") : show(v);
      return `<tr><td>${esc(k)}</td><td>${esc(text)}</td></tr>`;
    })
    .join("");
  return `${leftOutHtml()}<table><tbody>${rows}</tbody></table>`;
}

// -- questions --------------------------------------------------------------------------------

// One method's hits and parsed answer, one column of an opened question.
function methodHtml(m, row) {
  const hits = (row?.hit_ids ?? [])
    .map((h, i) => {
      const skill = row.hit_skills?.[i];
      const words = show(row.hit_matched?.[i]); // rows from before hit_matched lack it
      return `<li>${esc(skill ? `${h} (${skill})` : h)}, matched keywords:
        <span data-matched>${esc(words)}</span></li>`;
    })
    .join("");
  const parsed = row?.pred == null ? "" : `<p>Parsed answer: ${esc(show(row.pred))}
    <span class="muted">${esc(show(row.raw))}</span></p>`;
  return `<div data-method-hits><strong>${esc(method(m))}</strong>${parsed}
    <p>Top hits:</p><ul>${hits || "<li>–</li>"}</ul></div>`;
}

function questionHtml(q, picked) {
  const outcome = q.methods[picked]?.outcome;
  const mark = outcome ? `${outcome} ` : "";
  const choices = (q.choices ?? []).map((c, i) => `${String.fromCharCode(65 + i)}. ${c}`);
  const columns = run.setup.methods.filter((m) => q.methods[m]).map((m) => methodHtml(m, q.methods[m]));
  return `<details><summary>${esc(mark)}${esc(q.question ?? q.question_id)}</summary>
    ${choices.length ? `<p>${choices.map(esc).join("<br>")}</p>` : ""}
    <p>Gold: ${esc(show(q.gold_answers?.length ? q.gold_answers : q.gold_ids))}</p>
    <div class="sides">${columns.join("")}</div></details>`;
}

// Rows per "where methods disagree" value, kept once the run has finished so Method and Wrong
// only filter; a busy run's rows are still coming, so they are fetched each time and not kept.
const cached = {};
let showing = null; // the rows drawn now
let asking = 0; // bumped by each load, so only the newest one draws
const byId = (a, b) => String(a.question_id).localeCompare(String(b.question_id), undefined, { numeric: true });

// The picker names the method whose outcome heads each question and "wrong only" judges.
function showQuestions() {
  const m = $("q-method").value;
  const rows = showing;
  if (!m || !rows) return;
  const shown = $("q-wrong").checked ? rows.filter((q) => q.methods[m]?.outcome === "wrong") : rows;
  $("questions").innerHTML = shown.length ? shown.map((q) => questionHtml(q, m)).join("") : '<p class="muted">None.</p>';
}

async function loadQuestions() {
  const disagree = $("q-disagree").checked;
  const mine = ++asking;
  const keep = !busy(); // decided when asked: rows fetched while busy are never kept
  let rows = keep ? cached[disagree] : undefined;
  if (!rows) {
    try {
      rows = (await api(`/eval/${encodeURIComponent(id)}/questions?disagree=${disagree}`)).sort(byId);
    } catch (e) {
      if (mine === asking) $("questions").innerHTML = `<p class="err">${esc(e.message)}</p>`;
      return;
    }
    if (keep) cached[disagree] = rows;
  }
  if (mine === asking) {
    showing = rows;
    showQuestions();
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
  if (BUSY.has(s)) {
    buttons.push('<button type="button" id="act-cancel">Cancel</button>');
  } else {
    if (RESUMABLE.has(s) && resumeProbe?.refusal) {
      buttons.push(`<span class="warn" id="act-refusal">${esc(resumeProbe.refusal)}</span>`);
    } else if (RESUMABLE.has(s)) {
      buttons.push('<button type="button" id="act-resume">Resume</button>');
    }
    // Re-run and Delete wait for the run to stop: deleting a queued or running run is refused.
    buttons.push('<button type="button" id="act-rerun" class="secondary">Re-run</button>');
    buttons.push('<button type="button" id="act-delete" class="secondary">Delete…</button>');
  }
  return `<div class="actions">${buttons.join("")}</div>
    <div id="act-confirm"></div><p id="act-msg" class="err" role="status"></p>`;
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

function bindActions() {
  $("act-cancel")?.addEventListener("click", cancel);
  $("act-resume")?.addEventListener("click", () => {
    say("");
    askResume($("act-confirm"), id, refresh);
  });
  $("act-rerun")?.addEventListener("click", rerun);
  $("act-delete")?.addEventListener("click", () => {
    say("");
    askDelete($("act-confirm"), {
      question: "Delete this run and its results? This cannot be undone.",
      yes: "Yes, delete this run",
      path: `/eval/${encodeURIComponent(id)}`,
      gone: `/c/${enc}`,
    });
  });
}

// -- refresh --------------------------------------------------------------------------------------

let timer = null;
async function refresh() {
  clearTimeout(timer);
  try {
    cat = await catalog();
    const was = run?.status;
    run = await api(`/eval/${encodeURIComponent(id)}`);
    // Finished while the questions were open: what they show was fetched while rows were coming.
    if (BUSY.has(was) && !busy() && $("questions-box").open) loadQuestions();
    qaType ??= (await api(`/collections/${encodeURIComponent(run.setup.collection)}`)).qa_type;
    if (run.status !== probedStatus) resumeProbe = null;
    if (RESUMABLE.has(run.status) && resumeProbe === null) {
      await probeResume();
      probedStatus = run.status;
    }
    $("head").innerHTML = headHtml();
    paint("answer", answerHtml());
    paint("gaps", gapsHtml());
    $("note").textContent = run.gaps.length && !busy()
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
$("q-disagree").addEventListener("change", loadQuestions);
$("questions-box").addEventListener("toggle", () => {
  if ($("questions-box").open) loadQuestions();
});
refresh();
