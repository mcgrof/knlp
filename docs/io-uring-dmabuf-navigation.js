// SPDX-License-Identifier: MIT
// Preserve links to sections moved from the original overview.
(() => {
  const pages = {
    "path": "io-uring-dmabuf-memory.html",
    "implementations": "io-uring-dmabuf-memory.html",
    "nixl": "io-uring-dmabuf-memory.html",
    "platform": "io-uring-dmabuf-results.html",
    "tests": "io-uring-dmabuf-results.html",
    "validation-fixes": "io-uring-dmabuf-results.html",
    "process": "io-uring-dmabuf-results.html",
    "kernel": "io-uring-dmabuf-linux.html",
    "history": "io-uring-dmabuf-linux.html",
    "reproduce": "io-uring-dmabuf-linux.html",
    "source": "io-uring-dmabuf-linux.html"
};
  function followSection() {
    const anchor = window.location.hash.slice(1);
    if (Object.prototype.hasOwnProperty.call(pages, anchor)) {
      window.location.replace(pages[anchor] + "#" + anchor);
    }
  }
  followSection();
  window.addEventListener("hashchange", followSection);
})();
