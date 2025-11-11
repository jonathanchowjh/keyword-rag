import { head, shell } from "/static/ui.js";

shell("collections");
document.getElementById("main").innerHTML =
  head({ title: "Collections", crumb: [], sub: "Each question bank, and whether keyword retrieval helps on it." }) +
  '<p class="muted">Coming in the next step.</p>';
