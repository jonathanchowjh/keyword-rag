import { head, shell } from "/static/ui.js";

shell("collections");
document.getElementById("main").innerHTML =
  head({ title: "Collection", crumb: [{ label: "Collections", href: "/" }], sub: "One question bank." }) +
  '<p class="muted">Coming in the next step.</p>';
