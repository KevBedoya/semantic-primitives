"""Graded semantic similarity (STS), by Spearman correlation.

Spearman rather than Pearson because only the *ranking* is claimed. The kernels
here live on different scales -- Hellinger is in [0, 1] and normalised so
k(t,t)=1, the linear kernel is not -- so any correlation sensitive to scale would
compare the kernels' calibration rather than their semantics.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


@torch.no_grad()
def _pairwise(model, kernel, h_a: torch.Tensor, h_b: torch.Tensor):
    a, b = model(h_a), model(h_b)
    phi_a, phi_b = kernel.feature_map(a), kernel.feature_map(b)
    return (phi_a * phi_b).sum(dim=-1)          # row-wise, not the full matrix


@torch.no_grad()
def evaluate(model, kernel, h_a: torch.Tensor, h_b: torch.Tensor,
             gold: torch.Tensor) -> dict:
    """Spearman for the model and the frozen-encoder baseline."""
    model_sim = _pairwise(model, kernel, h_a, h_b).cpu().numpy()
    base_sim = (F.normalize(h_a, dim=-1) * F.normalize(h_b, dim=-1)
                ).sum(dim=-1).cpu().numpy()
    g = gold.cpu().numpy()
    return {
        "model": float(spearmanr(model_sim, g).statistic),
        "baseline": float(spearmanr(base_sim, g).statistic),
        "n": int(len(g)),
    }


def format_table(row: dict, title: str = "STS (Spearman)") -> str:
    return (f"{title}\n"
            f"{'n':>7} {'model':>8} {'LaBSE':>8} {'delta':>8}\n"
            f"{row['n']:>7,} {row['model']:>8.3f} {row['baseline']:>8.3f} "
            f"{row['model'] - row['baseline']:>+8.3f}")
