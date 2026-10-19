// The old paper URL is a Git symlink to the canonical guide.
if (location.pathname.endsWith('/paper_memory_decode.html')) {
  location.replace('asymmetric-quantization.html' + location.search + location.hash);
}
// Preserve historical result identities when following old fragment links.
const archivedAnchors = new Set([
  "october-followup-checks",
  "prefill-routing",
  "routing-title",
  "routing-caption",
  "next-tests",
  "dense-transfer",
  "d64-model-outcome",
  "forward-tests",
  "saturation",
  "capacity-title",
  "capacityStartValue",
  "capacityStart",
  "capacityFactor",
  "capacityChart",
  "capacityResult",
  "capacityDetail",
  "historical-speedup",
  "historicalBatch",
  "speedupVal",
  "batchSlider",
  "satChart",
  "fused-int4-rd",
  "traffic-regime",
  "kv-asymmetry",
  "cross-gpu",
  "speculation",
  "tiering",
  "future",
  "related-papers",
  "earlier-measurements",
  "october-serving",
  "batch32-study-status",
  "october-b32-findings",
  "b32-primary-results",
  "b32-primary-quality",
  "b32-bridge-results",
  "october-repair-checks",
  "full-stack",
  "asym-k16v8",
  "per-model-throughput",
  "reproduce"
]);
function openArchivedAnchor() {
  const id = location.hash.slice(1);
  if (archivedAnchors.has(id)) location.replace("asymmetric-history.html#" + id);
}
openArchivedAnchor();
addEventListener("hashchange", openArchivedAnchor);
