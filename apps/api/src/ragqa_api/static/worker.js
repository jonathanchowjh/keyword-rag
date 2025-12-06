import { api, clock, confirmSpend, esc, head, pct, pill, refreshSpend, shell } from "/static/ui.js";

const POLL_MS = 5000;
const FINISHED = 50;
const BUSY = new Set(["queued", "running"]);
const RESUMABLE = new Set(["interrupted", "cancelled", "stopped_budget", "stopped_no_key", "failed"]);
const KIND_WORDS = { load: "Loading", rekey: "Re-keying", eval: "Evaluating" };

const $ = (x) => document.getElementById(x);
const painted = {}; // section id -> html last drawn, so a poll never rebuilds what has not changed

shell("worker");
$("head").innerHTML = head({ title: "Worker", sub: "What the background worker is doing." });

function paint(section, html) {
  if (painted[section] === html) return;
  painted[section] = html;
  $(section).innerHTML = html;
}

const words = (job) => `${KIND_WORDS[job.kind] ?? job.kind} ${job.collection}`;
const say = (text) => ($("msg").textContent = text);

// -- sections -----------------------------------------------------------------------------

function problemsHtml(problems) {
  if (!problems.length) return '<p class="muted">Nothing needs attention.</p>';
  return `<ul>${problems.map((p) => `<li class="err" data-problem>${esc(p)}</li>`).join("")}</ul>`;
}

function runningHtml(job, batches) {
  if (!job) return '<p class="muted">Nothing is running.</p>';
  const batch = batches.find((b) => b.job_id === job.id);
  const waiting = batch
    ? `<p class="warn">waiting on Claude batch since ${esc(clock(batch.created_at))}, can take up to 24 h</p>`
    : "";
  const log = (job.log ?? []).slice(-10)
    .map((l) => `<li>${esc(clock(l.at))} ${esc(l.message)}</li>`)
    .join("");
  const cancel = job.cancel_requested
    ? '<span class="muted">cancelling…</span>'
    : `<button type="button" class="secondary" data-cancel="${esc(job.id)}">Cancel</button>`;
  return `<div class="card" data-running-job><strong>${esc(words(job))}</strong>
    <code>${esc(job.id.slice(0, 8))}</code> ${esc(pct(job.progress))} ${cancel}${waiting}
    <ul class="muted">${log}</ul></div>`;
}

function queuedHtml(queue) {
  if (!queue.length) return '<p class="muted">Nothing is waiting.</p>';
  const rows = queue
    .map((j) => `<tr data-queued-job="${esc(j.id)}"><td>${esc(words(j))}<code>${esc(j.id.slice(0, 8))}</code></td>
      <td>${esc(clock(j.created_at))}</td><td><button type="button" class="secondary"
      data-cancel="${esc(j.id)}">Cancel</button></td></tr>`)
    .join("");
  return `<table><tbody>${rows}</tbody></table>`;
}

function batchesHtml(batches) {
  if (!batches.length) return '<p class="muted">No Claude batches are pending.</p>';
  const rows = batches
    .map((b) => `<tr><td><code>${esc(b.id)}</code></td><td>${esc(b.model)}</td><td>${esc(b.purpose)}</td>
      <td>since ${esc(clock(b.created_at))}</td><td>${esc(b.job_id ? b.job_id.slice(0, 8) : "")}</td></tr>`)
    .join("");
  return `<table><thead><tr><th>Batch</th><th>Model</th><th>Purpose</th><th>Submitted</th><th>Job</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

function finishedHtml(jobs) {
  if (!jobs.length) return '<p class="muted">Nothing has finished yet.</p>';
  const rows = jobs
    .map((j) => {
      const resume = j.kind === "eval" && RESUMABLE.has(j.status)
        ? `<button type="button" class="secondary" data-resume="${esc(j.id)}">Resume</button>` : "";
      const error = j.error ? ` <span class="err">${esc(j.error)}</span>` : "";
      return `<tr data-finished-job="${esc(j.id)}"><td>${esc(words(j))}<code>${esc(j.id.slice(0, 8))}</code></td>
        <td>${pill(j.status)}${error}</td><td>${esc(clock(j.updated_at))}</td><td>${resume}</td></tr>`;
    })
    .join("");
  return `<table><tbody>${rows}</tbody></table>`;
}

// -- actions ------------------------------------------------------------------------------

async function cancel(jobId) {
  say("");
  try {
    await api(`/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
  } catch (e) {
    say(e.message);
  }
  await refresh();
}

// Money rule: the click only asks for the estimate (or the refusal); spending needs Start.
async function resume(jobId) {
  say("");
  const box = $("resume-box");
  box.innerHTML = "";
  const path = `/jobs/${encodeURIComponent(jobId)}/resume`;
  try {
    const { estimate } = await api(path, { method: "POST", body: { confirm: false } });
    confirmSpend(box, estimate, async () => {
      try {
        await api(path, { method: "POST", body: { confirm: true } });
        box.innerHTML = "";
        refreshSpend();
        await refresh();
      } catch (e) {
        say(e.message);
      }
    });
  } catch (e) {
    box.innerHTML = `<p class="warn" data-refusal>${esc(e.message)}</p>`; // the refusal text
  }
}

$("main").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-cancel], button[data-resume]");
  if (!button) return;
  if (button.dataset.cancel) cancel(button.dataset.cancel);
  else resume(button.dataset.resume);
});

// -- refresh ------------------------------------------------------------------------------

async function refresh() {
  try {
    const [status, jobs] = await Promise.all([api("/status"), api("/jobs?limit=200")]);
    paint("problems", problemsHtml(status.problems));
    paint("running", runningHtml(status.worker.running, status.batches));
    paint("queued", queuedHtml(status.queue));
    paint("batches", batchesHtml(status.batches));
    paint("finished", finishedHtml(jobs.filter((j) => !BUSY.has(j.status)).slice(0, FINISHED)));
  } catch (e) {
    say(e.message);
  }
}

refresh();
setInterval(refresh, POLL_MS);
