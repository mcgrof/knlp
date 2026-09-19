# SPDX-License-Identifier: GPL-2.0
"""Receiver-sensitive KV cache screening and cross-model cache translation.

Post-hoc and calibration-only: both models stay frozen.  The question this
package exists to answer, and to answer cheaply enough to abandon, is whether a
receiver-sensitive Jacobian second moment ranks real KV-cache damage better
than the cheap incumbents, and whether it then improves a *constrained* mapper
at matched bytes.  See ``tests/jacobian_kv/test_linearization.py`` for the
invariance that bounds what such a metric can possibly do.
"""
