import { head, shell } from "/static/ui.js";

shell("worker");
document.getElementById("main").innerHTML =
  head({ title: "Worker", crumb: [], sub: "What the background worker is doing." }) +
  '<p class="muted">Coming in the next step.</p>';
