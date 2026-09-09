"""Loss invariants, and the one that was decided by measurement.

test_cosine_reconstruction_ignores_scale pins the finding from
scripts/decoder_geometry.py: scoring direction rather than distance is what
keeps the decoder columns near the data, and therefore interpretable. If that
default is ever silently reverted to squared error, the archetypes drift to ~18x
the data radius and stop denoting anything a text resembles -- with no test
failing and no metric moving, because reconstruction *accuracy* is unaffected.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import pytest
import torch

from semprim import losses as L
from semprim.kernels import HellingerKernel


def simplex(n, dim, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(n, dim, generator=g), dim=-1)


def test_cosine_reconstruction_ignores_scale():
    """The property the whole decoder-geometry finding rests on.

    A cosine loss is invariant to the reconstruction's magnitude, so the decoder
    gains nothing by inflating its columns -- which is exactly why the
    archetypes stay on the data sphere under it and blow up under squared error.
    """
    torch.manual_seed(0)
    y = torch.randn(32, 64)
    r = torch.randn(32, 64)
    base = L.reconstruction_loss(r, y, mode="cosine")
    for scale in (0.01, 3.0, 100.0):
        scaled = L.reconstruction_loss(r * scale, y, mode="cosine")
        assert torch.allclose(base, scaled, atol=1e-5), \
            f"cosine loss changed by {abs(base - scaled):.3e} under scale {scale}"


def test_l2_reconstruction_does_not_ignore_scale():
    """The contrast: squared error charges for norm, which is the whole problem."""
    torch.manual_seed(0)
    y = torch.nn.functional.normalize(torch.randn(32, 64), dim=-1)
    r = torch.nn.functional.normalize(torch.randn(32, 64), dim=-1)
    assert not torch.allclose(L.reconstruction_loss(r, y, mode="l2"),
                              L.reconstruction_loss(r * 5.0, y, mode="l2"))


def test_cosine_reconstruction_is_zero_for_aligned_and_bounded():
    y = torch.randn(16, 32)
    assert L.reconstruction_loss(y, y, normalize_teacher=True,
                                 mode="cosine").item() == pytest.approx(0.0, abs=1e-6)
    assert 0.0 <= L.reconstruction_loss(torch.randn(16, 32), y,
                                        mode="cosine").item() <= 2.0


def test_unknown_reconstruction_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown reconstruction mode"):
        L.reconstruction_loss(torch.randn(4, 8), torch.randn(4, 8), mode="mse")


def test_align_loss_is_zero_on_identical_inputs():
    a = simplex(16, 12)
    for geom in ("l2", "hellinger"):
        assert L.align_loss(a, a, geometry=geom).item() == pytest.approx(0.0, abs=1e-6)


def test_align_geometries_disagree_on_ordering_sometimes():
    """They are different metrics; if they were interchangeable there would be
    no reason to offer the choice the plan calls for testing."""
    a, b = simplex(64, 16, seed=1), simplex(64, 16, seed=2)
    assert not torch.allclose(L.align_loss(a, b, geometry="l2"),
                              L.align_loss(a, b, geometry="hellinger"))


def test_l1_penalty_would_be_constant_on_the_simplex():
    """Documents why sparsity_loss uses entropy rather than L1.

    ||alpha||_1 is identically 1 on the simplex, so an L1 penalty is a constant
    with zero gradient. Asserting it keeps anyone from 'adding L1 sparsity' and
    wondering why nothing changes.
    """
    for seed in range(3):
        assert simplex(32, 20, seed=seed).abs().sum(-1).allclose(
            torch.ones(32), atol=1e-5)


def test_entropy_handles_exact_zeros():
    """entmax produces exact zeros; 0 log 0 must be 0, not NaN."""
    a = torch.tensor([[0.5, 0.5, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    h = L.entropy(a)
    assert torch.isfinite(h).all()
    assert h[1].item() == pytest.approx(0.0, abs=1e-5)


def test_dead_coordinates_counts_unused_primitives():
    a = torch.zeros(10, 8)
    a[:, :3] = 1.0 / 3
    assert L.dead_coordinates(a) == 5


def test_contrastive_loss_rewards_the_diagonal():
    """A perfect encoder (positives identical to anchors) must beat a shuffled one."""
    k = HellingerKernel()
    a = simplex(32, 16, seed=3)
    good = L.contrastive_loss(a, a.clone(), k, temperature=0.05)
    bad = L.contrastive_loss(a, a[torch.randperm(32)], k, temperature=0.05)
    assert good < bad
