"""Evaluation harness.

Every table carries the frozen-encoder baseline. The bottleneck is expected to
lose accuracy against the encoder it was built on; the result is the shape of
that loss, and a table without the baseline row invites reading a mediocre number
as a good one.
"""

from . import hard_cases, interpret, retrieval, similarity

__all__ = ["retrieval", "similarity", "hard_cases", "interpret"]
