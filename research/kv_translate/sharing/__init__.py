"""Does one stored prefix cache serve two models, and is that worth doing?

The reference mappers in ``published`` preserve a receiver's accuracy on
short multiple-choice prompts. This package asks the two questions that
result leaves open: whether quality holds on long document and code contexts
scored by what the model generates, and whether restoring and translating a
stored source cache reaches the first token sooner than the receiver
recomputing the prefix.

The workload is built on the processor and frozen before any model runs.
Every context is a reusable prefix of an exact length with fixed questions
asked after it, a ground truth that a program can check, and a source group
that tells one independent document or repository from another.
"""
