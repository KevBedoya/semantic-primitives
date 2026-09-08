"""Cross-lingual retrieval, always against the frozen-encoder baseline.

WHY THE BASELINE IS NOT OPTIONAL
--------------------------------
The bottleneck compresses 768 dimensions into N simplex coordinates. It is
expected to *lose* retrieval accuracy against the encoder it was built on. The
result of this project is the shape of that loss -- how much accuracy buys how
much interpretability, as a function of N and sparsity -- not its absence. A
table without the baseline row invites reading a mediocre number as a good one,
so every function here returns both and `format_table` refuses to hide it.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def precision_at_1(scores: torch.Tensor) -> float:
    """Fraction of rows whose argmax is the diagonal.

    ``scores[i, j]`` is the similarity of query i to candidate j, with the
    correct translation of query i on the diagonal.
    """
    return (scores.argmax(dim=1) == torch.arange(
        scores.shape[0], device=scores.device)).float().mean().item()


@torch.no_grad()
def baseline_scores(h_query: torch.Tensor, h_cand: torch.Tensor) -> torch.Tensor:
    """Frozen-encoder cosine -- the number everything else is measured against."""
    return F.normalize(h_query, dim=-1) @ F.normalize(h_cand, dim=-1).T


@torch.no_grad()
def kernel_scores(model, kernel, h_query: torch.Tensor,
                  h_cand: torch.Tensor) -> torch.Tensor:
    return kernel(model(h_query), model(h_cand))


@torch.no_grad()
def evaluate(model, kernel, pairs_by_lang: dict) -> dict:
    """P@1 per language for both the model and the baseline.

    ``pairs_by_lang`` maps a language code to ``(h_pivot, h_other)`` aligned row
    for row. Retrieval is pivot -> other, so a low score for a non-pivot pair is
    partly a property of the pivoted training data (see data/corpora.py), not
    only of the bottleneck.
    """
    rows = {}
    for lang, (h_q, h_c) in pairs_by_lang.items():
        rows[lang] = {
            "model": precision_at_1(kernel_scores(model, kernel, h_q, h_c)),
            "baseline": precision_at_1(baseline_scores(h_q, h_c)),
            "n": int(h_q.shape[0]),
        }
    if rows:
        rows["mean"] = {
            "model": float(np.mean([r["model"] for r in rows.values()])),
            "baseline": float(np.mean([r["baseline"] for r in rows.values()])),
            "n": int(sum(r["n"] for r in rows.values())),
        }
    return rows


def format_table(rows: dict, title: str = "Cross-lingual retrieval P@1") -> str:
    out = [title, f"{'lang':>6} {'n':>7} {'model':>8} {'LaBSE':>8} {'delta':>8}"]
    for lang, r in rows.items():
        delta = r["model"] - r["baseline"]
        out.append(f"{lang:>6} {r['n']:>7,} {r['model']:>8.3f} "
                   f"{r['baseline']:>8.3f} {delta:>+8.3f}")
    return "\n".join(out)
