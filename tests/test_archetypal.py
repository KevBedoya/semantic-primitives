"""The reconstruction ceiling must actually be a ceiling.

If the trained head ever reports lower reconstruction error than the
non-amortised fit at the same N, the two are not measuring the same thing --
different targets, different normalisation, or a different N -- and the
diagnostic that distinguishes "the head can't compute this" from "N is too
small" is silently broken. train.py warns at runtime; this asserts the property
the warning relies on.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import torch

from semprim.archetypal import archetypal_fit, fit_alpha, project_simplex


def _hull_data(n=400, d=12, k=5, seed=0):
    """Points genuinely inside the convex hull of k archetypes.

    Constructed rather than random so the ceiling has a known value: with the
    true archetypes recoverable, reconstruction error should approach zero.
    """
    g = torch.Generator().manual_seed(seed)
    arche = torch.randn(k, d, generator=g).double()
    w = torch.rand(n, k, generator=g).double()
    w = w / w.sum(dim=-1, keepdim=True)
    return w @ arche, arche


def test_fit_alpha_returns_points_on_the_simplex():
    Y, D = _hull_data()
    a = fit_alpha(Y, D, iters=300)
    assert (a >= -1e-9).all()
    assert torch.allclose(a.sum(-1), torch.ones(a.shape[0], dtype=a.dtype),
                          atol=1e-6)


def test_fit_alpha_recovers_known_mixtures():
    """Given the true archetypes, the codes should be nearly exact."""
    Y, D = _hull_data(n=200, d=10, k=4, seed=1)
    a = fit_alpha(Y, D, iters=1000)
    err = ((a @ D - Y) ** 2).sum(dim=-1).mean().item()
    assert err < 1e-6, f"expected near-exact reconstruction inside the hull, got {err:.3e}"


def test_more_archetypes_never_reconstruct_worse():
    """Monotonicity in N -- the sweep is meaningless without it.

    A larger hull contains a smaller one, so the optimum cannot get worse. This
    is an alternating minimisation with a k-means start, so it is not guaranteed
    to find the optimum; a small tolerance allows for that without allowing a
    real inversion to pass.
    """
    Y, _ = _hull_data(n=300, d=14, k=6, seed=2)
    errs = {n: archetypal_fit(Y, n, outer=12, inner=60, seed=0,
                              normalize=False)[2]
            for n in (3, 6, 12)}
    assert errs[6] <= errs[3] * 1.05, errs
    assert errs[12] <= errs[6] * 1.05, errs


def test_projection_handles_degenerate_rows():
    """Constant and already-extreme rows must not produce NaN."""
    v = torch.tensor([[0.0, 0.0, 0.0],
                      [1e6, -1e6, 0.0],
                      [0.5, 0.5, 0.0]], dtype=torch.float64)
    w = project_simplex(v)
    assert torch.isfinite(w).all()
    assert torch.allclose(w.sum(-1), torch.ones(3, dtype=torch.float64), atol=1e-9)
    assert (w >= 0).all()
