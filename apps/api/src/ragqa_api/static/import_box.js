import { CONNECT_AI, api, count, esc, followJob, needsAI, plural, priceText, say, shownPrice, upload, usd, short, UNPRICED, refreshSpend } from "/static/ui.js";

// The upload-preview-import widget for a store's items (queues a job, priced first) or an eval set's
// questions (imported in the request, no estimate). Fills `el`; `done` runs after an import.
// Items: the price sits beside Import, and the click on Import is the confirm; the line then
// follows the job to what it added, counted in `noun` (the store type's item noun).
// A Documents store's items (`qaType` "documents") are .md and .txt files of passages: the
// preview shows the passages, with no column mapping to pick, and the guessed one is sent.
export class ImportBox {
  constructor(el, { uploadPath, importPath, kind, qaType, typeLabel, noun = "item", done, next }) {
    this.el = el;
    this.noun = noun;
    this.uploadPath = uploadPath;
    this.importPath = importPath;
    this.kind = kind;
    this.done = done;
    this.next = next; // () => html appended to a finished import's line, or undefined
    this.passages = kind === "items" && qaType === "documents";
    this.guessed = {}; // the preview's own mapping, sent when there are no selects
    this.uploadId = null; // the upload being previewed, or null
    this.batch = null; // several files being previewed (a Documents store), or null
    this.ticket = 0; // the newest preview request, so an older answer is ignored
    this.own = (what) => el.querySelector(`#imp-${kind}-${what}`);
    const accept = this.passages ? ".md,.txt" : ".jsonl,.json,.csv,.tsv,.txt,.md,.zip";
    const hint = kind === "questions" && qaType ? `<p class="muted">Questions to add to this eval set (${esc(typeLabel ?? qaType)}).</p>` : "";
    el.innerHTML = `<div class="card form">${hint}
      <label>File<input type="file" id="imp-${kind}-file" accept="${accept}"${this.passages ? " multiple" : ""}></label>
      <label>Source name<input id="imp-${kind}-source" maxlength="64"></label>
      <div class="row"><button type="button" id="imp-${kind}-go">Preview</button>
        <span id="imp-${kind}-msg" class="err" role="status"></span></div>
      <div id="imp-${kind}-preview"></div></div>`;
    this.own("file").addEventListener("change", (e) => {
      // Several files: each is its own source, named by its upload.
      this.own("source").closest("label").hidden = e.target.files.length > 1;
      this.uploadId = null; // the old preview belongs to the old file
      this.batch = null;
      this.ticket++; // and no answer still on its way may paint it back
      this.own("preview").innerHTML = "";
      say(this.own("msg"), "");
      this.own("source").value = ""; // the upload names it, the route's rule
    });
    this.own("go").addEventListener("click", () => this.upload());
    // Items are priced under the source name, so a new name or mapping needs a new estimate.
    // Questions have no estimate to redo.
    if (kind === "items") {
      this.own("source").addEventListener("input", () => this.uploadId && this.previewAgain());
      this.own("preview").addEventListener("change", (e) => {
        if (this.uploadId && e.target.closest("select[data-field]")) this.previewAgain();
      });
    }
  }

  // Uploads one file to the store's upload route; resolves to its preview.
  send(file) {
    return upload(`${this.uploadPath}?filename=${encodeURIComponent(file.name)}&kind=${this.kind}`, file);
  }

  // Draws into the preview unless a newer file or upload has taken over.
  line(ticket, html) {
    if (ticket === this.ticket) this.own("preview").innerHTML = html;
  }

  async upload() {
    const files = [...this.own("file").files];
    const file = files[0];
    const msg = this.own("msg");
    say(msg, "");
    if (!file) {
      say(msg, "Pick a file first.");
      return;
    }
    if (files.length > 1) return this.previewMany(files);
    try {
      const got = await this.send(file);
      this.uploadId = got.upload_id;
      this.ticket++; // a new preview: an import line or an estimate still on its way paints nothing
      this.show(got.preview, got.estimate);
      // The upload is priced under its cleaned file name; a name typed first is priced anew.
      const typed = this.own("source").value;
      if (!typed) this.own("source").value = got.source_name;
      else if (typed !== got.source_name && this.kind === "items") this.previewAgain();
    } catch (e) {
      this.uploadId = null;
      this.own("preview").innerHTML = "";
      say(msg, e.message);
    }
  }

  mapping() {
    if (this.passages) return this.guessed;
    const out = {};
    for (const select of this.own("preview").querySelectorAll("select[data-field]")) {
      out[select.dataset.field] = select.value || null;
    }
    return out;
  }

  body(confirm) {
    const body = { upload_id: this.uploadId, mapping: this.mapping(), source_name: this.own("source").value };
    return this.kind === "items" ? { ...body, kind: "items", confirm } : body;
  }

  async previewAgain() {
    const msg = this.own("msg");
    const ticket = ++this.ticket;
    say(msg, "");
    this.own("import")?.setAttribute("disabled", ""); // until the fresh estimate is in
    try {
      const got = await api(this.importPath, { method: "POST", body: this.body(false) });
      if (ticket === this.ticket) this.show(got.preview, got.estimate);
    } catch (e) {
      if (ticket === this.ticket) say(msg, e.message);
    }
  }

  // A Documents store's preview: how many passages, and the first few.
  passagesHtml(p) {
    const sample = p.sample.length
      ? `<div class="scroll"><table><thead><tr><th>Passage</th></tr></thead>
          <tbody>${p.sample.map((row) => `<tr><td>${esc(row.text ?? "")}</td></tr>`).join("")}</tbody></table></div>`
      : "";
    return `<p>${esc(p.format)} · ${esc(plural(p.rows, "passage"))} · ${esc(count(p.usable))} usable</p>${sample}`;
  }

  html(p, estimate) {
    const problems = p.problems.length
      ? `<ul class="warn">${p.problems.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`
      : "";
    if (this.passages) return this.passagesHtml(p) + problems + this.footHtml(p, estimate);
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
    const missing = p.missing.length ? `<p class="err">Map these fields first: ${esc(p.missing.join(", "))}.</p>` : "";
    return `<p>${esc(p.format)} · ${esc(plural(p.rows, "row"))} · ${esc(count(p.usable))} usable</p>${problems}${sample}
      <div class="row">${selects}</div>${missing}${this.footHtml(p, estimate)}`;
  }

  // The Import button with what the import costs.
  footHtml(p, estimate) {
    const cost = estimate
      ? `<span class="muted">${esc(priceText(estimate))} · ${esc(plural(estimate.calls ?? 0, "AI call"))}</span>${
        needsAI(estimate) ? ` ${CONNECT_AI}` : ""}`
      : "";
    // A questions import cannot be previewed again under a new mapping: the server judges it.
    const open = (p.ready || this.kind === "questions") && !needsAI(estimate);
    // Items have no Start: the click on Import, beside the price, is the confirm.
    const removed = estimate?.removed > 0
      ? `<p class="warn" data-removed>${esc(`${plural(estimate.removed, "old " + this.noun)} will be removed, since the text changed.`)}</p>`
      : "";
    return `<div class="row"><button type="button" id="imp-${this.kind}-import" ${open ? "" : "disabled"}>Import</button>
        ${cost}</div>${removed}`;
  }

  show(preview, estimate) {
    this.guessed = preview.mapping;
    this.own("preview").innerHTML = this.html(preview, estimate);
    this.own("import").addEventListener("click", () => this.start(estimate));
  }

  async start(estimate) {
    const msg = this.own("msg");
    say(msg, "");
    if (this.kind === "items") return this.queueItems(estimate);
    try {
      const got = await api(this.importPath, { method: "POST", body: this.body(true) });
      this.own("preview").innerHTML = `<p class="ok">Imported: ${esc(got.new)} new, ${esc(got.updated)} updated, ${esc(
        got.unchanged)} unchanged, ${esc(got.skipped)} skipped. Version ${esc(short(got.version))}.</p>`;
      this.uploadId = null;
      this.own("file").value = "";
      refreshSpend();
      await this.done?.();
    } catch (e) {
      say(msg, e.message);
    }
  }

  // Items: the click on Import, with its price beside it, queues the import (the server prices it
  // again first, and refuses it when its calls are not the `estimate` drawn here). The button stays
  // disabled, so a second click queues nothing; a refusal frees it. The click takes the upload
  // over: an edit while it is sent previews nothing, so no Import is armed for it again. The line
  // then follows the job, until a new file takes the box over.
  async queueItems(estimate) {
    const button = this.own("import");
    button.disabled = true;
    const ticket = this.ticket;
    const body = this.body(true);
    const uploadId = this.uploadId;
    this.uploadId = null;
    let got;
    try {
      got = await api(this.importPath, { method: "POST", body: { ...body, ...shownPrice(estimate) } });
    } catch (e) {
      if (ticket === this.ticket) {
        this.uploadId = uploadId; // a refusal can be tried again
        // edited while it was sent, or refused as it stands (a 409, such as a price that
        // changed since it was drawn): priced again first
        const same = JSON.stringify(this.body(true)) === JSON.stringify(body);
        if (same && e.status !== 409) button.disabled = false;
        else this.previewAgain();
      }
      say(this.own("msg"), e.message);
      return;
    }
    if (ticket === this.ticket) this.own("file").value = ""; // else a new file or upload took over
    let job;
    try {
      job = await followJob(got.job_id, (j) => this.line(ticket, j.status === "queued"
        ? "<p>Waiting for the store's other job…</p>"
        : `<p>Adding… ${esc(Math.round((j.progress ?? 0) * 100))}%</p>`));
    } catch (e) {
      this.line(ticket, `<p class="err">${esc(e.message)}</p>`);
      return;
    }
    this.line(ticket, this.doneHtml(job));
    refreshSpend();
    await this.done?.();
  }

  // Several files (a Documents store): each is previewed in turn, then priced and imported as one.
  async previewMany(files) {
    const ticket = ++this.ticket;
    this.uploadId = null;
    this.batch = null;
    this.own("preview").innerHTML = "";
    const entries = [];
    for (const file of files) {
      try {
        entries.push({ name: file.name, up: await this.send(file) });
      } catch (e) {
        entries.push({ name: file.name, error: e.message }); // wrong type, too big, unreadable
      }
      if (ticket !== this.ticket) return; // a new file took the box over
    }
    this.showMany(entries);
  }

  // The files' previews in one summary line and a line each, with one Import for them all. Each
  // file's own price is drawn on a store as it stands, so with several the sum errs high.
  showMany(entries, before = "") {
    this.batch = entries;
    const ready = entries.filter((f) => f.up?.estimate);
    const adds = (f) => Number(f.up?.estimate?.new ?? 0) + Number(f.up?.estimate?.updated ?? 0);
    const added = ready.reduce((n, f) => n + adds(f), 0);
    const removed = ready.reduce((n, f) => n + Number(f.up.estimate.removed ?? 0), 0);
    const unpriced = ready.find((f) => f.up.estimate.est_usd === null && f.up.estimate.calls > 0);
    const noAI = ready.find((f) => needsAI(f.up.estimate));
    const total = ready.reduce((n, f) => n + Number(f.up.estimate.est_usd ?? 0), 0);
    const price = unpriced ? UNPRICED : ready.length > 1 ? `at most ${usd(total)}` : usd(total);
    const gone = removed > 0 ? `, ${count(removed)} to remove` : "";
    const each = entries
      .map((f) => `<li>${esc(f.error ? `${f.name}: ${f.error}` : `${f.name}: ${plural(adds(f), this.noun)}`)}</li>`)
      .join("");
    const why = unpriced
      ? `<span class="err">${esc(`Can't price ${unpriced.name}: set up a model in Settings`)}</span>`
      : noAI
        ? ` ${CONNECT_AI}`
        : "";
    const summary = ready.length
      ? `<p>${esc(`${plural(ready.length, "file")}: ${plural(added, this.noun)} to add${gone} · ${price}`)}</p>`
      : "";
    const foot = ready.length
      ? `<div class="row"><button type="button" id="imp-${this.kind}-import" ${unpriced || noAI ? "disabled" : ""}>${esc(
          unpriced ? "Import" : `Import (${price})`)}</button>${why}</div>`
      : "";
    this.own("preview").innerHTML = `${before}${summary}<ul>${each}</ul>${foot}`;
    this.own("import")?.addEventListener("click", () => this.importMany());
  }

  // Imports the files in order, each after the one before is in. The first sends the calls and
  // price it was shown; a later one only its price, since an earlier file can change its calls
  // (the server then refuses only a price that rose). A refusal stops there: the files not yet
  // in are priced again, for the next click.
  async importMany() {
    this.own("import").disabled = true;
    say(this.own("msg"), ""); // the last round's message is not this one's
    const ticket = this.ticket;
    const todo = this.batch.filter((f) => f.up?.estimate);
    this.batch = null; // the click takes the files over
    const sum = { new: 0, updated: 0, unchanged: 0, removed: 0 };
    const sofar = () => (sum.new + sum.updated + sum.unchanged + sum.removed > 0
      ? this.doneHtml({ status: "done", result: sum }) : "");
    // The store's page shows what is in, as the single-file import does after its job.
    const settled = async () => {
      refreshSpend();
      await this.done?.();
    };
    for (const [i, f] of todo.entries()) {
      if (ticket !== this.ticket) { // a new file choice took the box over: send no more
        if (sofar()) await settled(); // the files before this one are in
        return;
      }
      const shown = i === 0 ? shownPrice(f.up.estimate) : { est_usd: f.up.estimate.est_usd };
      let got;
      try {
        got = await api(this.importPath, { method: "POST", body: { ...manyBody(f.up, true), ...shown } });
      } catch (e) {
        if (ticket !== this.ticket) {
          if (sofar()) await settled();
          return;
        }
        say(this.own("msg"), e.status === 409 ? `The price of ${f.name} changed: check it again` : e.message);
        await this.repriceMany(todo.slice(i), ticket, sofar());
        if (sofar()) await settled(); // the files before this one are in
        return;
      }
      let job;
      try {
        job = await followJob(got.job_id, (j) => this.line(ticket, j.status === "queued"
          ? "<p>Waiting for the store's other job…</p>"
          : `<p>${esc(`Adding ${f.name}… ${Math.round((j.progress ?? 0) * 100)}%`)}</p>`));
      } catch (e) {
        this.line(ticket, sofar() + `<p class="err">${esc(e.message)}</p>`);
        if (sofar()) await settled();
        return;
      }
      if (job.status !== "done") {
        this.line(ticket, sofar() + this.doneHtml(job));
        await settled(); // a stopped job may have added some, as the single-file import allows
        return;
      }
      for (const key of Object.keys(sum)) sum[key] += Number(job.result?.[key] ?? 0);
    }
    this.line(ticket, sofar());
    if (ticket === this.ticket) {
      this.own("file").value = ""; // else a new file took over
      this.own("source").closest("label").hidden = false;
    }
    await settled();
  }

  // The files not yet imported, priced again on the store as it now is.
  async repriceMany(entries, ticket, before) {
    const priced = [];
    for (const f of entries) {
      try {
        const got = await api(this.importPath, { method: "POST", body: manyBody(f.up, false) });
        priced.push({ name: f.name, up: { ...f.up, ...got } });
      } catch (e) {
        priced.push({ name: f.name, error: e.message });
      }
      if (ticket !== this.ticket) return;
    }
    this.showMany(priced, before);
  }

  // What a finished import job did, in a line.
  doneHtml(job) {
    if (job.status !== "done") {
      const why = job.error ? `: ${job.error}` : ".";
      return `<p class="err">${esc(`The import ${job.status.replaceAll("_", " ")}${why}`)}</p>`;
    }
    const r = job.result ?? {};
    const added = Number(r.new ?? 0) + Number(r.updated ?? 0);
    const kept = Number(r.unchanged ?? 0);
    const already = kept > 0 ? ` ${count(kept)} already in the store.` : "";
    const removed = Number(r.removed ?? 0);
    const gone = removed > 0 ? ` ${plural(removed, "old " + this.noun)} removed.` : "";
    const next = this.next ? ` Next: ${this.next()}` : "";
    return `<p class="ok">${esc(`Added ${plural(added, this.noun)}.${already}${gone}`)}${next}</p>`;
  }
}

// One of several files' import, under the upload's own mapping and source name.
export function manyBody(up, confirm) {
  return { upload_id: up.upload_id, kind: "items", mapping: up.preview.mapping, source_name: up.source_name, confirm };
}
