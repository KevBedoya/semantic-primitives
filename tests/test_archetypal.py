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


def test_free_scale_is_never_worse_than_the_simplex_fit():
    """A superset optimum cannot be worse -- and once wasn't.

    ``fit_nonneg`` optimises over ``b >= 0``, which contains the simplex, so at
    the same N its error must be <= the simplex fit's. The first free-scale
    sweep violated this badly (0.04071 vs 0.02431 at N=256, with a max archetype
    radius of 99), which is impossible for a true optimum and therefore
    diagnosed the optimiser rather than the model: with ``b`` unbounded,
    ``(b, D)`` and ``(cb, D/c)`` are identical, and the alternating fit wandered
    along that ray. ``archetypal_fit`` now pins ||d_i|| = 1 whenever the scale is
    free, which removes the redundancy without losing generality.

    A small tolerance because both sides are alternating minimisations, not
    guaranteed global optima -- but nothing like the 1.7x inversion seen before.
    """
    Y, _ = _hull_data(n=400, d=16, k=8, seed=3)
    Y = torch.nn.functional.normalize(Y, dim=-1)
    for n in (8, 24):
        simplex_mse = archetypal_fit(Y, n, outer=20, inner=80, seed=0,
                                     normalize=False)[2]
        scaled_mse = archetypal_fit(Y, n, outer=20, inner=80, seed=0,
                                    normalize=False, free_scale=True)[2]
        assert scaled_mse <= simplex_mse * 1.05, (
            f"N={n}: free-scale fit ({scaled_mse:.5f}) is worse than the "
            f"strictly more constrained simplex fit ({simplex_mse:.5f}); the "
            f"scale degeneracy is back")


def test_free_scale_keeps_archetypes_near_the_data():
    """The interpretability property the whole finding is about.

    Unit-norm data, so archetypes should stay near radius 1 rather than being
    driven outward to buy norm. Under squared error with a fixed scale they
    reached 17x the data radius on the real corpus.
    """
    Y, _ = _hull_data(n=400, d=16, k=8, seed=4)
    Y = torch.nn.functional.normalize(Y, dim=-1)
    D, _, _ = archetypal_fit(Y, 16, outer=20, inner=80, seed=0,
                             normalize=False, free_scale=True)
    assert float(D.norm(dim=-1).max()) < 1.01, \
        f"archetypes drifted off the unit sphere: max norm {D.norm(dim=-1).max():.3f}"


def test_projection_handles_degenerate_rows():
    """Constant and already-extreme rows must not produce NaN."""
    v = torch.tensor([[0.0, 0.0, 0.0],
                      [1e6, -1e6, 0.0],
                      [0.5, 0.5, 0.0]], dtype=torch.float64)
    w = project_simplex(v)
    assert torch.isfinite(w).all()
    assert torch.allclose(w.sum(-1), torch.ones(3, dtype=torch.float64), atol=1e-9)
    assert (w >= 0).all()
