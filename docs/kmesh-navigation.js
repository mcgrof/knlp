// SPDX-License-Identifier: MIT
// Keep published question fragments working after the page split.
(() => {
  const pages = {
    "redundancies": "kmesh-design.html",
    "current": "kmesh-design.html",
    "generalization": "kmesh-design.html",
    "unresolved": "kmesh-design.html",
    "milestone": "kmesh-design.html",
    "baseline": "kmesh-baselines.html",
    "reproduce": "kmesh-baselines.html",
    "results": "kmesh-baselines.html",
    "fork-accounting": "kmesh-baselines.html",
    "serving-replay": "kmesh-baselines.html",
    "prefix-sharing": "kmesh-baselines.html",
    "restore-latency": "kmesh-storage.html",
    "cascade": "kmesh-storage.html",
    "coldburst": "kmesh-storage.html",
    "singleflight": "kmesh-storage.html",
    "preload": "kmesh-storage.html",
    "preload-timeline-title": "kmesh-storage.html",
    "preload-timeline-desc": "kmesh-storage.html",
    "preload-time-arrow": "kmesh-storage.html",
    "preload-diagram-caption": "kmesh-storage.html",
    "retained": "kmesh-retention.html",
    "trace-audit": "kmesh-retention.html",
    "family-residency": "kmesh-retention.html",
    "reasoning-demand": "kmesh-retention.html",
    "prefetch-demand": "kmesh-retention.html"
  };

  function followQuestion() {
    const anchor = window.location.hash.slice(1);
    if (Object.prototype.hasOwnProperty.call(pages, anchor)) {
      window.location.replace(pages[anchor] + "#" + anchor);
    }
  }

  followQuestion();
  window.addEventListener("hashchange", followQuestion);
})();
