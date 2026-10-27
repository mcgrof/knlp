# SPDX-License-Identifier: MIT
"""Context cartridges: generated prefix key/value memory for a frozen reader.

A small trained writer reads a context once and emits a fixed number of
per-layer key/value rows. A frozen reader answers later questions from those
rows after the source is removed. The package holds the synthetic ledger
dataset, the writer/reader model, training, export, cold-start replay,
evaluation arms, and the report tooling.
"""
