"""The k-sparse projection and the bound it defines.

These exist because the unmatched version of this diagnostic gave a confidently
wrong answer. Comparing a head that activates ~21 of 256 coordinates against a
*dense* fit read "29.5x above ceiling", which by the documented interpretation
means the head cannot compute the mapping. Matched to k=21 the same head is
1.84x above -- headroom, not failure. The opposite conclusion, from the same
run, because the bound was not a bound.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import pytest
import torch

from semprim.archetypal import (archetypal_fit, fit_alpha, project_simplex,
                                project_simplex_topk)


def test_topk_projection_respects_the_support_budget():
    torch.manual_seed(0)
    v = torch.randn(50, 32, dtype=torch.float64)
    for k in (1, 3, 8, 31):
        w = project_simplex_topk(v, k)
        nz = (w > 0).sum(dim=-1)
        assert (nz <= k).all(), f"k={k}: got up to {nz.max()} nonzeros"
        assert torch.allclose(w.sum(-1), torch.ones(50, dtype=torch.float64),
                              atol=1e-10)
        assert (w >= 0).all()


def test_topk_projection_falls_back_to_dense_when_k_covers_everything():
    torch.manual_seed(1)
    v = torch.randn(20, 12, dtype=torch.float64)
    for k in (12, 13, 100):
        assert torch.allclose(project_simplex_topk(v, k), project_simplex(v),
                              atol=1e-12)


def test_topk_puts_no_mass_outside_the_top_k():
    """The actual contract: support is a SUBSET of the top-k, not all of it.

    Note it is a subset, not an equality, and the difference is real rather than
    pedantic: projecting [5, 4] onto the 2-simplex gives [1, 0] exactly -- the
    closest point with non-negative entries summing to one -- so a dominant
    coordinate can legitimately zero out the rest of its own budget. Asserting
    that every selected coordinate survives would be asserting something false
    about Euclidean projection.
    """
    v = torch.tensor([[5.0, -1.0, 4.0, 0.0, 3.0]], dtype=torch.float64)
    w = project_simplex_topk(v, 2)
    top = torch.topk(v, 2, dim=-1).indices[0].tolist()
    outside = [i for i in range(v.shape[1]) if i not in top]
    assert (w[0, outside] == 0).all(), w
    assert w.sum().item() == pytest.approx(1.0, abs=1e-12)

    # A case where the top-k are close enough that all k do survive, so the
    # selection is observably doing something rather than always collapsing.
    v2 = torch.tensor([[1.0, 0.9, -3.0, 0.85]], dtype=torch.float64)
    w2 = project_simplex_topk(v2, 3)
    assert (w2[0, [0, 1, 3]] > 0).all(), w2
    assert w2[0, 2] == 0, w2


def test_sparse_fit_is_never_better_than_the_dense_fit():
    """The direction that makes k-ceil a safe bound.

    Restricting the support can only hurt, so the k-sparse error must be >= the
    dense error at the same N. That is what makes the matched ceiling safe to
    quote: as a greedy (non-exact) projection it can overstate the achievable
    error, which makes the head look worse rather than better.
    """
    torch.manual_seed(2)
    Y = torch.nn.functional.normalize(torch.randn(300, 24, dtype=torch.float64),
                                      dim=-1)
    D = torch.nn.functional.normalize(torch.randn(32, 24, dtype=torch.float64),
                                      dim=-1)
    dense = ((fit_alpha(Y, D, iters=300) @ D - Y) ** 2).sum(-1).mean()
    for k in (4, 12):
        sparse = ((fit_alpha(Y, D, iters=300, topk=k) @ D - Y) ** 2).sum(-1).mean()
        assert sparse >= dense - 1e-9, \
            f"k={k}: sparse fit {sparse:.6f} beat dense {dense:.6f}"


def test_matched_ceiling_is_looser_than_the_dense_one():
    """The whole point: a larger N does not make the sparse bound tight.

    At fixed k, growing N should improve the k-sparse ceiling (more archetypes
    to choose k from) but never reach the dense ceiling, which is free to use
    all N. If these ever coincide the diagnostic has stopped distinguishing the
    two regimes.
    """
    torch.manual_seed(3)
    Y = torch.nn.functional.normalize(torch.randn(400, 20, dtype=torch.float32),
                                      dim=-1)
    dense = archetypal_fit(Y, 48, outer=12, inner=60, seed=0,
                           normalize=False)[2]
    matched = archetypal_fit(Y, 48, outer=12, inner=60, seed=0,
                             normalize=False, topk=5)[2]
    assert matched > dense, (matched, dense)


def test_free_scale_with_topk_is_refused_rather_than_silently_dense():
    with pytest.raises(NotImplementedError, match="topk"):
        archetypal_fit(torch.randn(50, 8), 16, outer=2, inner=5,
                       free_scale=True, topk=4)
