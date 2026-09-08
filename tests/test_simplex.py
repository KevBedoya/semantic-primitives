"""Simplex invariants for the bottleneck and the projection.

If alpha ever leaves the simplex the whole design is void: the Hellinger kernel
takes a square root of it, the decoder's convex-hull reading depends on it, and
"share of semantic mass" stops meaning anything. Cheap to assert, so assert it
at every entmax setting rather than at the default only.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import pytest
import torch

from semprim.archetypal import project_simplex
from semprim.model import PrimitiveModel, SimplexBottleneck, simplex_project

TOL = 1e-6

try:
    import entmax  # noqa: F401
    HAVE_ENTMAX = True
except ImportError:
    HAVE_ENTMAX = False

ALPHAS = [1.0] + ([1.5, 2.0, 1.25] if HAVE_ENTMAX else [])


@pytest.mark.parametrize("alpha", ALPHAS)
def test_bottleneck_output_is_on_the_simplex(alpha):
    torch.manual_seed(0)
    b = SimplexBottleneck(in_dim=32, hidden=64, n_primitives=17, alpha=alpha)
    a = b(torch.randn(50, 32) * 5.0)          # large scores: stress the tails
    assert (a >= -TOL).all(), f"alpha={alpha}: negative entries"
    assert torch.allclose(a.sum(-1), torch.ones(50), atol=1e-5), \
        f"alpha={alpha}: rows do not sum to 1"


@pytest.mark.skipif(not HAVE_ENTMAX, reason="entmax not installed")
def test_higher_alpha_is_sparser():
    """The reason alpha is annealed: it must actually control sparsity."""
    torch.manual_seed(0)
    z = torch.randn(200, 32)
    tau = torch.tensor(1.0)
    counts = [(simplex_project(z, a, tau) > 1e-6).float().sum(-1).mean().item()
              for a in (1.0, 1.5, 2.0)]
    assert counts[0] > counts[1] >= counts[2], \
        f"expected support to shrink with alpha, got {counts}"


def test_softmax_path_needs_no_entmax():
    """alpha = 1.0 must work on a machine without the optional dependency."""
    a = simplex_project(torch.randn(4, 8), 1.0, torch.tensor(1.0))
    assert torch.allclose(a.sum(-1), torch.ones(4), atol=1e-6)


def test_project_simplex_matches_a_reference():
    """Duchi projection against a slow but obviously-correct implementation."""
    torch.manual_seed(0)
    v = torch.randn(40, 9, dtype=torch.float64) * 3

    def brute(row):
        # min ||w - row||^2 s.t. simplex, by scanning the threshold directly.
        u, _ = torch.sort(row, descending=True)
        css = u.cumsum(0)
        best = None
        for j in range(row.numel()):
            theta = (css[j] - 1) / (j + 1)
            if u[j] - theta > 0:
                best = theta
        return (row - best).clamp_min(0)

    got = project_simplex(v)
    want = torch.stack([brute(r) for r in v])
    assert torch.allclose(got, want, atol=1e-10)
    assert torch.allclose(got.sum(-1), torch.ones(40, dtype=torch.float64),
                          atol=1e-10)


def test_project_simplex_is_idempotent_on_the_simplex():
    torch.manual_seed(1)
    a = torch.softmax(torch.randn(20, 11, dtype=torch.float64), dim=-1)
    assert torch.allclose(project_simplex(a), a, atol=1e-10)


def test_embedding_is_identity_and_unnormalised():
    """f(t) = Q alpha with Q = I. If this ever normalises, the design breaks."""
    m = PrimitiveModel(in_dim=16, teacher_dim=16, hidden=32, n_primitives=8)
    a = m(torch.randn(6, 16))
    e = m.embedding(a)
    assert torch.equal(e, a)
    assert torch.allclose(e.sum(-1), torch.ones(6), atol=1e-5), \
        "embedding must remain a convex combination, not a unit vector"


def test_decoder_init_rejects_wrong_shape():
    m = PrimitiveModel(in_dim=16, teacher_dim=16, hidden=32, n_primitives=8)
    with pytest.raises(ValueError, match="expected archetypes"):
        m.init_decoder_(torch.randn(7, 16))
