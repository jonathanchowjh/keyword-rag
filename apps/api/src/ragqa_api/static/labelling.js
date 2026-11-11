import { $, NONE, api, askDelete, confirmSpend, count, esc, passageHtml, refreshSpend, say, short, whereHtml } from "/static/ui.js";

// -- a Documents set: labelling -------------------------------------------------------------------
// Its questions are labelled against the linked store: Find passages, then y or n on each. Every
// save writes that one question; only Get the AI's answer spends, after its estimate is confirmed.

const LABEL = { y: "Right", n: "Wrong" }; // a mark as the buttons say it

// The labelling section of eval_set.js's page. The page passes what it owns: `currentSet` and
// `currentRows` read its set and rows, which each load replaces; `name` and `enc` are the set's
// name, plain and encoded; `load` and `showRestore` are its own functions.
export class Labelling {
  constructor({ currentSet, currentRows, name, enc, load, showRestore }) {
    this.currentSet = currentSet;
    this.currentRows = currentRows;
    this.name = name;
    this.enc = enc;
    this.load = load;
    this.showRestore = showRestore;
    this.open = null; // the question in the panel as read, its conflict check; null for a new one
    this.shown = []; // the passages listed: the pool's, then labelled ones it missed, then lost ids
    this.found = []; // passages from Search the store not listed above
    this.notes = []; // the pool's notes
    this.marks = new Map(); // passage id -> "y" or "n"; unmarked is not judged
    this.finding = 0; // the latest Find passages, so an older reply is dropped
    this.saving = false; // a save is in flight: nothing may change which question the panel holds
    this.restoring = false; // Restore's PUT is in flight: the same holds, and no mark changes
    this.unsaved = false; // a mark, the text or the reference answer changed since the panel opened or saved
    this.discarding = null; // what Label or Delete will do once its unsaved marks are discarded
    this.asking = 0; // the latest Get the AI's answer: a price drawn for an older one is dropped
    this.answering = 0; // the question the panel holds: a paid answer for an older one is not shown
    this.searched = false; // Find passages or Search the store has answered for this question
    this.saveEmpty = false; // Save anyway was pressed: this save goes ahead with no passage marked Right
    this.paying = false; // the confirmed AI's answer is out: Start can't buy it again
  }

  get set() {
    return this.currentSet(); // the latest GET /eval-sets/{name}
  }

  get rows() {
    return this.currentRows(); // the page's working copy of its questions
  }

  busy() {
    return this.saving || this.restoring;
  }

  // Runs Restore's `work` with the panel held: no mark, Find, Save, Label or Delete can change what
  // it holds. `work` resolves true once the PUT landed; the question it held may then be gone.
  // Typed in while the PUT was out, the panel is left as it is: the next Save is checked against
  // the question as read, and offers Reload.
  async restore(work) {
    this.restoring = true;
    this.canSave(); // set before the first await, so a second click finds it disabled
    const landed = await work();
    this.restoring = false;
    if (landed && this.open && !this.saving && !this.unsaved) this.openQuestion(null);
    this.canSave();
  }

  docStore() {
    return this.set.store;
  }

  lostIds() {
    return [...new Set(Object.values(this.set.missing ?? {}).flat())];
  }

  // `stores`: the Documents stores, in name order, each with its default eval set.
  storeLineHtml(stores) {
    if (this.docStore()) {
      const store = this.docStore();
      return `<p>Labelled against <a href="/c/${encodeURIComponent(store)}">${esc(store)}</a>.</p>`;
    }
    // An older set is usually its store's default, named after it: that store comes first.
    const pick = stores.find((s) => s.eval_set === this.name) ?? stores.find((s) => s.name === this.name) ?? stores[0];
    const options = stores
      .map((s) => `<option value="${esc(s.name)}" ${s === pick ? "selected" : ""}>${esc(s.name)}</option>`)
      .join("");
    return `<div class="row"><label>Pick a store to label against: <select id="lab-pick">${options}</select></label>
    <button type="button" id="lab-link" ${stores.length ? "" : "disabled"}>Link</button></div>`;
  }

  warningsHtml() {
    const out = [];
    const lost = this.lostIds().length;
    if (this.docStore() && lost) {
      out.push(lost === 1
        ? `1 labelled passage is no longer in ${this.docStore()}`
        : `${count(lost)} labelled passages are no longer in ${this.docStore()}`);
    }
    const unscored = this.rows.filter((q) => !q.gold_ids.length).length;
    if (unscored) {
      out.push(unscored === 1
        ? "1 question has no right passage; it is not scored"
        : `${count(unscored)} questions have no right passage; they are not scored`);
    }
    return out.map((line) => `<p class="warn">${esc(line)}</p>`).join("");
  }

  labelRowHtml(q) {
    const lost = this.set.missing?.[q.id]?.length ?? 0;
    const gone = lost && this.docStore() ? ` <span class="muted">(${count(lost)} no longer in ${esc(this.docStore())})</span>` : "";
    const label = this.docStore()
      ? `<button type="button" class="secondary" data-label="${esc(q.id)}" ${this.busy() ? "disabled" : ""}>Label</button> `
      : "";
    return `<tr><td>${esc(q.text)}</td><td>${count(q.gold_ids.length)}${gone}</td>
    <td>${q.judged_ids ? count(q.judged_ids.length) : NONE}</td>
    <td>${label}<button type="button" class="secondary" data-del="${esc(q.id)}" ${this.busy() ? "disabled" : ""}>Delete question…</button></td></tr>`;
  }

  labelTableHtml() {
    if (!this.rows.length) return '<p class="muted">No questions yet.</p>';
    return `<div class="scroll"><table><thead><tr><th>Question</th><th>Right passages</th><th>Judged</th><th></th></tr></thead>
    <tbody>${this.rows.map((q) => this.labelRowHtml(q)).join("")}</tbody></table></div>`;
  }

  async drawLabelling() {
    const stores = this.docStore()
      ? []
      : (await api("/collections"))
        .filter((c) => c.qa_type === "documents")
        .sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
    $("lab-store").innerHTML = this.storeLineHtml(stores);
    $("lab-warn").innerHTML = this.warningsHtml();
    $("lab-questions").innerHTML = this.labelTableHtml();
    $("lab-panel").hidden = !this.docStore();
    $("lab-ai").hidden = !this.set.answer_available; // no answer model: the typed field is all there is
    $("lab-ai-hint").hidden = !this.set.answer_available;
    this.canSave();
  }

  // -- the panel --

  // Every passage id in display order, then any marked id not listed (a Find that failed).
  listedIds() {
    const ids = [...this.shown, ...this.found].map((p) => p.id);
    return [...ids, ...[...this.marks.keys()].filter((id) => !ids.includes(id))];
  }

  markHtml(p) {
    const mark = this.marks.get(p.id);
    const button = (m) => `<button type="button" class="${mark === m ? "" : "secondary"}" data-mark="${m}"
    aria-pressed="${mark === m}">${LABEL[m]}</button>`;
    let body = `<code>${esc(p.id)}</code>`; // the store no longer holds it: its id is all there is
    if (p.text !== undefined) {
      body = passageHtml(p.text, { meta: whereHtml(p) });
    }
    return `<li data-passage="${esc(p.id)}">${body}<div class="actions">${button("y")}${button("n")}</div></li>`;
  }

  drawPanel() {
    $("lab-notes").innerHTML = this.notes.map((n) => `<li>${esc(n)}</li>`).join("");
    $("lab-passages").innerHTML = this.shown.map((p) => this.markHtml(p)).join("");
    $("lab-found").innerHTML = this.found.length
      ? `<h3>From your search</h3><ul class="evidence">${this.found.map((p) => this.markHtml(p)).join("")}</ul>`
      : "";
    const ids = this.listedIds();
    const y = ids.filter((id) => this.marks.get(id) === "y").length;
    const n = ids.filter((id) => this.marks.get(id) === "n").length;
    $("lab-count").textContent = ids.length
      ? `${count(y)} right, ${count(n)} wrong, ${count(ids.length - y - n)} not judged`
      : "";
    this.canSave();
  }

  canSave() {
    const blank = !$("lab-text").value.trim();
    const unfound = !this.open && !this.searched && this.set?.store != null; // a new question is saved after Find
    $("lab-find").disabled = blank || this.busy();
    for (const id of ["lab-save", "lab-next"]) $(id).disabled = blank || unfound || this.busy();
    $("lab-save-why").textContent = !blank && unfound ? "Find passages first, so you can mark the right ones." : "";
    for (const b of $("lab-questions").querySelectorAll("button[data-label], button[data-del]")) b.disabled = this.busy();
    for (const b of $("lab-panel").querySelectorAll("button[data-mark]")) b.disabled = this.restoring;
    const right = this.listedIds().some((id) => this.marks.get(id) === "y");
    $("lab-ai").disabled = blank || !right || this.busy();
    for (const b of $("lab-ai-answer").querySelectorAll("button[data-ai-mark]")) b.disabled = this.busy();
    for (const b of $("lab-ai-confirm").querySelectorAll("button")) b.disabled = this.busy() || this.paying;
    this.showRestore();
  }

  // A change to the panel. A Delete asked before it checked for unsaved marks, so it is withdrawn.
  markUnsaved() {
    this.unsaved = true;
    $("lab-del").innerHTML = "";
  }

  // Label and Delete would drop the panel's unsaved marks (Delete reloads the page), so while
  // there are any they ask first; `then` runs on Discard them.
  unlessUnsaved(then) {
    if (!this.unsaved) {
      then();
      return;
    }
    this.discarding = then;
    $("lab-discard").innerHTML = `<p class="warn">${esc("This question has unsaved marks.")}</p>
    <div class="actions"><button type="button" class="danger" id="lab-discard-yes">Discard them</button>
      <button type="button" class="secondary" id="lab-discard-keep">Cancel</button></div>`;
    $("lab-discard").scrollIntoView({ block: "nearest" });
  }

  closeDiscard() {
    this.discarding = null;
    $("lab-discard").innerHTML = "";
  }

  // Opens a question in the panel (null: a new, empty one) and, for a question, finds its passages.
  openQuestion(q) {
    this.closeDiscard();
    this.closeSaveAsk();
    this.searched = false;
    this.forgetAnswer();
    this.answering++; // a paid answer still under way is for what was here before
    $("lab-ai-answer").innerHTML = "";
    this.open = q;
    this.unsaved = false;
    $("lab-text").value = q?.text ?? "";
    $("lab-answer").value = q?.gold_answers[0] ?? "";
    $("lab-search").value = "";
    this.marks = new Map();
    for (const id of q?.judged_ids ?? []) this.marks.set(id, "n");
    for (const id of q?.gold_ids ?? []) this.marks.set(id, "y");
    this.shown = [];
    this.found = [];
    this.notes = [];
    this.finding++; // a Find still under way is for what was here before
    for (const id of ["lab-find-msg", "lab-search-msg", "lab-ai-msg"]) say($(id), "", "err");
    this.drawPanel();
    if (q) this.findPassages();
  }

  async findPassages() {
    const question = $("lab-text").value.trim();
    if (!question) return;
    // The passages already marked come back even when the pool misses them.
    const stored = this.open ? [...this.open.gold_ids, ...(this.open.judged_ids ?? [])] : [];
    const labelled = [...new Set([...stored, ...this.marks.keys()])];
    const mine = ++this.finding;
    say($("lab-find-msg"), "", "err");
    try {
      const pool = await api(`/eval-sets/${this.enc}/pool`, { method: "POST", body: { question, labelled } });
      if (mine !== this.finding) return;
      this.shown = [...pool.passages, ...pool.labelled, ...pool.missing.map((id) => ({ id }))];
      this.found = [];
      this.notes = pool.notes;
      this.searched = true;
      this.drawPanel();
    } catch (e) {
      if (mine === this.finding) {
        this.searched = true; // a failed Find must not lock Save
        say($("lab-find-msg"), e.message, "err");
        this.canSave();
      }
    }
  }

  // Search the store: hybrid at 10, never Settings' method, which may be a paid one.
  async searchStore() {
    const question = $("lab-search").value.trim();
    if (!question) return;
    say($("lab-search-msg"), "", "err");
    try {
      const reply = await api("/query", {
        method: "POST",
        body: { collection: this.docStore(), question, methods: ["hybrid"], k: 10 },
      });
      const result = reply.results.hybrid;
      if (result.error) say($("lab-search-msg"), result.error, "err");
      const listed = new Set([...this.shown, ...this.found].map((p) => p.id));
      const before = this.found.length;
      for (const hit of result.hits) {
        if (listed.has(hit.id)) continue;
        const meta = hit.meta ?? {};
        this.found.push({ id: hit.id, text: hit.text, file: meta.file, part: meta.part, parts: meta.parts });
      }
      if (!result.error && this.found.length === before) {
        say($("lab-search-msg"), "Nothing new: every passage the search found is already listed.", "err");
      }
      this.searched = true;
      this.drawPanel();
    } catch (e) {
      this.searched = true;
      say($("lab-search-msg"), e.message, "err");
      this.canSave();
    }
  }

  toggleMark(id, m) {
    if (this.restoring) return; // what the panel holds may be replaced when it lands
    this.forgetAnswer();
    if (this.marks.get(id) === m) this.marks.delete(id); // clicked again: back to not judged
    else this.marks.set(id, m);
    this.closeSaveAsk();
    this.markUnsaved();
    this.drawPanel();
  }

  // -- the AI's answer --

  // Any mark, text or question change drops a price not yet started: it priced what was here.
  forgetAnswer() {
    this.asking++;
    $("lab-ai-confirm").innerHTML = "";
  }

  // The AI's answer from the passages marked Right: its estimate first, one call only on Start.
  async askAnswer() {
    this.forgetAnswer();
    const mine = this.asking;
    const opened = this.answering;
    const passages = this.listedIds().filter((id) => this.marks.get(id) === "y");
    const body = { question: $("lab-text").value.trim(), passages, confirm: false };
    const path = `/eval-sets/${this.enc}/answer`;
    say($("lab-ai-msg"), "", "err");
    try {
      const { estimate } = await api(path, { method: "POST", body });
      if (this.asking !== mine) return; // the panel changed while it was priced
      confirmSpend($("lab-ai-confirm"), estimate, async (shown) => {
        // A save or Restore may reset the panel, dropping what was paid for; one call per confirm.
        if (this.busy() || this.paying) return;
        if (this.asking !== mine) {
          $("lab-ai-confirm").innerHTML = "";
          return;
        }
        this.paying = true;
        try {
          const { answer } = await api(path, { method: "POST", body: { ...body, ...shown } });
          refreshSpend();
          if (this.asking === mine) $("lab-ai-confirm").innerHTML = "";
          if (this.answering === opened) this.showAnswer(answer); // a mark change keeps it; a question change doesn't
        } catch (e) {
          if (this.answering !== opened) return; // the question changed: its error is not this one's
          throw e;
        } finally {
          this.paying = false;
          this.canSave();
        }
      }, $("lab-ai-msg"));
      this.canSave(); // Start waits out a save or Restore under way
    } catch (e) {
      if (this.asking === mine) say($("lab-ai-msg"), e.message, "err");
    }
  }

  showAnswer(text) {
    const button = (m) => `<button type="button" class="secondary" data-ai-mark="${m}" aria-pressed="false">${LABEL[m]}</button>`;
    $("lab-ai-answer").innerHTML = `<p>Is this the right answer?</p><div class="full">${esc(text)}</div>
    <div class="actions">${button("y")}${button("n")}</div>`;
    this.canSave();
  }

  // y puts the answer in the reference answer field, n drops it; neither saves. An answer not
  // yet taken is not unsaved: asking again is free from the cache.
  takeAnswer(m) {
    if (this.busy()) return; // a save or Restore is deciding what the panel holds
    const box = $("lab-ai-answer");
    if (m === "y") {
      $("lab-answer").value = box.querySelector(".full").textContent;
      this.markUnsaved();
      this.showRestore();
      say($("lab-ai-msg"), "Added as the reference answer. Save question to keep it.", "ok");
    }
    box.innerHTML = "";
  }

  closeSaveAsk() {
    $("lab-save-ask").innerHTML = "";
  }

  // A question with no passage marked Right isn't scored: ask before saving one.
  askSaveEmpty(next) {
    $("lab-save-ask").innerHTML = `<p class="warn">${esc("No passage marked Right: this question won't be scored.")}</p>
    <div class="actions"><button type="button" id="lab-save-anyway">Save anyway</button>
      <button type="button" class="secondary" id="lab-save-cancel">Cancel</button></div>`;
    $("lab-save-anyway").addEventListener("click", () => {
      this.saveEmpty = true;
      this.closeSaveAsk();
      this.saveQuestion(next);
    });
    $("lab-save-cancel").addEventListener("click", () => this.closeSaveAsk());
    $("lab-save-ask").scrollIntoView({ block: "nearest" });
  }

  async saveQuestion(next) {
    const anyway = this.saveEmpty; // Save anyway's flag is spent by this call, whatever it does
    this.saveEmpty = false;
    if (this.busy()) return;
    this.closeSaveAsk();
    if (!anyway && !this.listedIds().some((id) => this.marks.get(id) === "y")) {
      this.askSaveEmpty(next);
      return;
    }
    this.closeDiscard(); // what it would discard is being saved
    this.saving = true;
    this.canSave(); // set before the first await, so a second click finds the buttons disabled
    try {
      await this.writeQuestion(next);
    } finally {
      this.saving = false;
      this.canSave();
    }
  }

  async writeQuestion(next) {
    say($("lab-msg"), "", "err");
    const ids = this.listedIds();
    const gold_ids = ids.filter((id) => this.marks.get(id) === "y");
    const judged_ids = ids.filter((id) => this.marks.has(id));
    const typed = $("lab-answer").value.trim();
    const rest = (this.open?.gold_answers ?? []).slice(1); // the other accepted answers, as they were
    const gold_answers = typed ? [typed, ...rest] : rest;
    const text = $("lab-text").value.trim();
    const before = this.open; // what is saved; the panel is rebound only while it still holds it
    this.unsaved = false; // what is in the panel now is being saved; a change from here on is not
    try {
      let id = this.open?.id;
      let version;
      if (this.open) {
        const question = { ...this.open, text, gold_answers, gold_ids, judged_ids };
        ({ version } = await api(`/eval-sets/${this.enc}/questions/${encodeURIComponent(id)}`, {
          method: "PUT",
          body: { question, base: this.open },
        }));
      } else {
        ({ id, version } = await api(`/eval-sets/${this.enc}/questions`, {
          method: "POST",
          body: { question: { text, gold_answers, gold_ids, judged_ids } },
        }));
      }
      await this.load();
      if (this.open === before) {
        if (next) {
          this.openQuestion(null);
          $("lab-text").focus();
        } else {
          this.open = this.set.questions.find((q) => q.id === id) ?? null; // the next Save is checked against it
        }
      }
      say($("lab-msg"), `Saved as version ${short(version)}.`, "ok");
      say($("lab-ai-msg"), "", "err"); // "Save question to keep it" reads wrong once it is kept
    } catch (e) {
      say($("lab-msg"), e.message, "err"); // every mark stays where it is
      this.markUnsaved();
      if (e.status === 404 && this.open === before && before) {
        // Deleted meanwhile: its row goes, and the next Save writes these marks as a new question.
        try {
          await this.load();
        } catch {
          // the detail already shown is the one that matters
        }
        if (this.open === before) this.open = null;
      }
      if (e.status === 409 && this.open) {
        $("lab-msg").insertAdjacentHTML("beforeend",
          ' <button type="button" id="lab-reload" class="secondary">Reload it (discard my marks)</button>');
      }
    }
  }

  async reloadQuestion() {
    const id = this.open?.id;
    say($("lab-msg"), "", "err");
    try {
      await this.load();
      this.openQuestion(this.set.questions.find((q) => q.id === id) ?? null);
    } catch (e) {
      say($("lab-msg"), e.message, "err");
    }
  }

  bindLabelling() {
    $("lab-store").addEventListener("click", async (e) => {
      if (!e.target.closest("#lab-link")) return;
      say($("lab-link-msg"), "", "err");
      try {
        await api(`/eval-sets/${this.enc}`, { method: "PATCH", body: { store: $("lab-pick").value } });
        await this.load();
      } catch (err) {
        say($("lab-link-msg"), err.message, "err");
      }
    });
    $("lab-questions").addEventListener("click", (e) => {
      const button = e.target.closest("button");
      if (!button) return;
      const d = button.dataset;
      if (d.label !== undefined) {
        this.unlessUnsaved(() => {
          $("lab-msg").textContent = "";
          this.openQuestion(this.set.questions.find((q) => q.id === d.label) ?? null);
          $("lab-panel").scrollIntoView({ block: "start" });
        });
      }
      if (d.del !== undefined) {
        this.unlessUnsaved(() => askDelete($("lab-del"), {
          question: "Delete this question and its marks?",
          yes: "Yes, delete it",
          path: `/eval-sets/${this.enc}/questions/${encodeURIComponent(d.del)}`,
          gone: `/e/${this.enc}`,
        }));
      }
    });
    $("lab-discard").addEventListener("click", (e) => {
      if (e.target.closest("#lab-discard-keep")) this.closeDiscard();
      if (e.target.closest("#lab-discard-yes")) {
        const then = this.discarding;
        this.closeDiscard();
        then?.();
      }
    });
    $("lab-text").addEventListener("input", () => {
      this.markUnsaved();
      this.forgetAnswer();
      this.canSave();
    });
    $("lab-answer").addEventListener("input", () => {
      this.markUnsaved();
      this.showRestore();
      say($("lab-ai-msg"), "", "err"); // edited by hand, the field no longer holds the AI's answer
    });
    $("lab-find").addEventListener("click", () => this.findPassages());
    $("lab-search-go").addEventListener("click", () => this.searchStore());
    $("lab-search").addEventListener("keydown", (e) => {
      if (e.key === "Enter") this.searchStore();
    });
    $("lab-panel").addEventListener("click", (e) => {
      const button = e.target.closest("button[data-mark]");
      if (button) this.toggleMark(button.closest("li[data-passage]").dataset.passage, button.dataset.mark);
    });
    $("lab-ai").addEventListener("click", () => this.askAnswer());
    $("lab-ai-answer").addEventListener("click", (e) => {
      const button = e.target.closest("button[data-ai-mark]");
      if (button) this.takeAnswer(button.dataset.aiMark);
    });
    $("lab-save").addEventListener("click", () => this.saveQuestion(false));
    $("lab-next").addEventListener("click", () => this.saveQuestion(true));
    $("lab-msg").addEventListener("click", (e) => {
      if (e.target.closest("#lab-reload")) this.reloadQuestion();
    });
  }
}
