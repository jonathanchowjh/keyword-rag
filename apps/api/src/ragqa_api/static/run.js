import { head, shell } from "/static/ui.js";

shell("collections");
document.getElementById("main").innerHTML =
  head({ title: "Run", crumb: [{ label: "Collections", href: "/" }], sub: "One evaluation run." }) +
  '<p class="muted">Coming in the next step.</p>';
