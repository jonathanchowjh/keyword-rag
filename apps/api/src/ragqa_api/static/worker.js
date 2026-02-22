import { $, BUSY, RESUMABLE, api, askResume, clock, esc, head, paint, pct, pill, shell, short } from "/static/ui.js";

const POLL_MS = 5000;
const FINISHED = 50;
const KIND_WORDS = { load: "Loading", rekey: "Re-keying", eval: "Evaluating" };

let jobs = []; // the latest GET /jobs, newest first
let resumeFor = null; // the finished job whose Resume was clicked; its estimate shows under its row
const resumeBox = document.createElement("div"); // kept across repaints of the finished table

shell("worker");
$("head").innerHTML = head({ title: "Worker", sub: "What the background worker is doing." });

const say = (text) => ($("msg").textContent = text);

// Where a job's results live: an eval's run page, or its collection for loads and re-keys.
function target(job) {
  const c = `/c/${encodeURIComponent(job.collection)}`;
  return job.kind === "eval" ? `${c}/runs/${encodeURIComponent(job.id)}` : c;
}

// "Evaluating sqa 1a2b3c4d", linked to where its results live.
function named(job) {
  return `<a href="${esc(target(job))}">${esc(`${KIND_WORDS[job.kind] ?? job.kind} ${job.collection}`)}</a>
    <code>${esc(short(job.id))}</code>`;
}

// A problem line is server text; it names a job by "<4 or 8 id characters> on <collection> ", or a
// collection's lists by "lists on <collection> ". Linked when a known job matches, plain otherwise.
function problemHref(text) {
  const names = (j, id) => text.includes(`${id} on ${j.collection} `);
  const job = jobs.find((j) => names(j, j.id.slice(0, 4)) || names(j, short(j.id)));
  if (job) return target(job);
  const lists = jobs.find((j) => text.includes(`lists on ${j.collection} `));
  return lists ? `/c/${encodeURIComponent(lists.collection)}` : null;
}

// -- sections -----------------------------------------------------------------------------

function problemsHtml(problems) {
  if (!problems.length) return '<p class="muted">Nothing needs attention.</p>';
  const line = (p) => {
    const href = problemHref(p);
    return href ? `<a href="${esc(href)}">${esc(p)}</a>` : esc(p);
  };
  return `<ul>${problems.map((p) => `<li class="err" data-problem>${line(p)}</li>`).join("")}</ul>`;
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
  return `<div class="card" data-running-job><strong>${named(job)}</strong>
    ${esc(pct(job.progress))} ${cancel}${waiting}
    <ul class="muted">${log}</ul></div>`;
}

function queuedHtml(queue) {
  if (!queue.length) return '<p class="muted">Nothing is waiting.</p>';
  const rows = queue
    .map((j) => `<tr data-queued-job="${esc(j.id)}"><td>${named(j)}</td>
      <td>${esc(clock(j.created_at))}</td><td><button type="button" class="secondary"
      data-cancel="${esc(j.id)}">Cancel</button></td></tr>`)
    .join("");
  return `<table><thead><tr><th>Job</th><th>Queued at</th><th></th></tr></thead><tbody>${rows}</tbody></table>`;
}

function batchesHtml(batches) {
  if (!batches.length) return '<p class="muted">No Claude batches are pending.</p>';
  const rows = batches
    .map((b) => `<tr><td><code>${esc(b.id)}</code></td><td>${esc(b.model)}</td><td>${esc(b.purpose)}</td>
      <td>since ${esc(clock(b.created_at))}</td><td>${esc(short(b.job_id))}</td></tr>`)
    .join("");
  return `<table><thead><tr><th>Batch</th><th>Model</th><th>Purpose</th><th>Submitted</th><th>Job</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

function finishedHtml(finished) {
  if (!finished.length) return '<p class="muted">Nothing has finished yet.</p>';
  const rows = finished
    .map((j) => {
      const resume = j.kind === "eval" && RESUMABLE.has(j.status)
        ? `<button type="button" class="secondary" data-resume="${esc(j.id)}">Resume</button>` : "";
      const error = j.error ? ` <span class="err">${esc(j.error)}</span>` : "";
      const row = `<tr data-finished-job="${esc(j.id)}"><td>${named(j)}</td>
        <td>${pill(j.status)}${error}</td><td>${esc(clock(j.updated_at))}</td><td>${resume}</td></tr>`;
      // The clicked Resume's estimate, refusal or error goes in a row of its own just under it.
      return j.id === resumeFor ? `${row}<tr class="resume-row" data-resume-row><td colspan="4"></td></tr>` : row;
    })
    .join("");
  return `<table><thead><tr><th>Job</th><th>Status</th><th>Ended at</th><th></th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

// Puts the kept resume box back in its row after a repaint (moving it keeps its Start button).
function placeResumeBox() {
  const cell = $("finished").querySelector("[data-resume-row] td");
  if (cell && resumeBox.parentNode !== cell) cell.append(resumeBox);
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
  resumeFor = jobId;
  resumeBox.innerHTML = "";
  await refresh(); // draws the row the box goes in
  await askResume(resumeBox, jobId, async () => {
    resumeFor = null;
    await refresh();
  });
  resumeBox.scrollIntoView({ block: "nearest" });
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
    let status;
    [status, jobs] = await Promise.all([api("/status"), api("/jobs?limit=200")]);
    const finished = jobs.filter((j) => !BUSY.has(j.status)).slice(0, FINISHED);
    if (!finished.some((j) => j.id === resumeFor)) resumeFor = null;
    paint("problems", problemsHtml(status.problems));
    paint("running", runningHtml(status.worker.running, status.batches));
    paint("queued", queuedHtml(status.queue));
    paint("batches", batchesHtml(status.batches));
    paint("finished", finishedHtml(finished));
    placeResumeBox();
    $("poll-err").textContent = "";
  } catch (e) {
    $("poll-err").textContent = `Couldn't refresh at ${clock(new Date().toISOString())}: ${e.message}`;
  }
}

refresh();
setInterval(refresh, POLL_MS);
