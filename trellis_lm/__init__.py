"""trellis_lm — a faithful, minimal Trellis bounded-memory language model.

Trellis (arXiv:2512.23852) replaces the growing KV cache with a fixed-size
learned memory that is read/written by a two-pass recurrent compression rule
and updated at test time by online gradient descent with a forget gate. This
package implements that as a real sequence-mixing LAYER (TrellisMixer) that
directly produces its output from the compressed memory state — NOT a
[B,H,T,T] attention mask. The mask-only "Trellis-KRI" selector is a separate,
parked line; this is the architecture.

Phase 0 (this build): exact sequential update (autograd VJP), unit tests, and
a synthetic associative-recall toy task proving correctness and bounded memory.
Chunked/parallel updates and the full eval matrix come later.
"""

from .config import TrellisConfig  # noqa: F401
