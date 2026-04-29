window.addEventListener("load", function () {
  if (window.mermaid) {
    window.mermaid.initialize({ startOnLoad: true });
  }
});
document$.subscribe(function () {
  if (typeof mermaid !== "undefined") {
    mermaid.initialize({
      startOnLoad: true,
      securityLevel: "loose",
      theme: "default",
    });
    mermaid.run();
  }
});
