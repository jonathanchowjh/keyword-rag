// Stamps the saved theme on <html> before first paint. ui.js owns the button and the same key.
(function () {
  try {
    const t = localStorage.getItem("theme");
    if (t === "light" || t === "dark") document.documentElement.dataset.theme = t;
  } catch (e) {}
})();
