// SPDX-License-Identifier: MIT
// Keep the original ReadPlan section fragments working after the page split.
(() => {
  const pages = {
    "improve": "lmcache-readplan-l2-analysis.html#serving-effect",
    "flexfiles": "lmcache-readplan-l2-analysis.html#location-changes",
    "falsify": "lmcache-readplan-l2-analysis.html#evidence"
  };

  function followSection() {
    const anchor = window.location.hash.slice(1);
    if (Object.prototype.hasOwnProperty.call(pages, anchor)) {
      window.location.replace(pages[anchor]);
    }
  }

  followSection();
  window.addEventListener("hashchange", followSection);
})();
