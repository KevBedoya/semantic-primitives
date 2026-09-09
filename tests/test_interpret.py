"""Interpretability diagnostics -- including one that was measuring itself.

`cross_language` divided each coordinate's top-k language entropy by the
FULL-corpus entropy. Entropy from k samples is biased downward, so that charged
every coordinate for sampling bias rather than for language specialisation: on
this corpus a coordinate drawing its top-k exactly from the corpus distribution
scored 0.785 at k=10, never 1.0. A smoke run failed the check at 0.685 against
what was effectively a 0.785 ceiling. The denominator is now the expected
entropy of a random k-sample, estimated the same way the numerator is.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from eval import interpret


def _corpus(n=4000, seed=0):
    """50% English plus six languages at 1/12 -- this project's actual skew."""
    rng = np.random.default_rng(seed)
    names = ["en", "es", "de", "ru", "zh", "ar", "hi"]
    p = np.array([0.5] + [0.5 / 6] * 6)
    return [names[i] for i in rng.choice(7, size=n, p=p)]


def test_corpus_typical_coordinate_scores_about_one():
    """The property the fix exists for: no penalty for merely existing.

    Random alpha means each coordinate's top-k is an unbiased draw from the
    corpus language distribution, so a correct metric puts it at 1.0 regardless
    of k. The old one gave 0.785 at k=10 and 0.959 at k=50 -- k-dependent, and
    wrong in a direction that manufactures language detectors.
    """
    langs = _corpus()
    torch.manual_seed(0)
    alpha = torch.rand(len(langs), 24)
    for k in (10, 50):
        xl = interpret.cross_language(alpha, langs, k=k, n_control=2000, seed=0)
        ratios = np.array([xl[i]["entropy_ratio"] for i in range(24)])
        assert abs(np.median(ratios) - 1.0) < 0.12, \
            f"k={k}: corpus-typical coordinates scored {np.median(ratios):.3f}"


def test_language_detector_scores_near_zero():
    """A coordinate firing only on one language must be caught."""
    langs = _corpus()
    alpha = torch.rand(len(langs), 4)
    zh = torch.tensor([l == "zh" for l in langs])
    alpha[:, 0] = torch.where(zh, 10.0, 0.0)      # coordinate 0: pure detector
    xl = interpret.cross_language(alpha, langs, k=50, n_control=1000, seed=0)
    assert xl[0]["entropy_ratio"] < 0.05, xl[0]
    assert xl[0]["dominant_lang"] == "zh"
    assert xl[0]["dominant_share"] == pytest.approx(1.0)


def test_control_is_reported_for_auditing():
    xl = interpret.cross_language(torch.rand(500, 8), _corpus(500), k=25,
                                  n_control=500, seed=0)
    assert xl["_control"]["k"] == 25
    assert xl["_control"]["n_languages"] == 7
    assert 0 < xl["_control"]["expected_entropy"] < np.log(7)


def test_paired_consistency_detects_agreement_and_its_absence():
    torch.manual_seed(0)

    class Identity:
        """Stand-in model: alpha is whatever it is handed."""
        def __call__(self, h):
            return torch.softmax(h, dim=-1)

    h = torch.randn(200, 16) * 3
    perfect = interpret.paired_consistency(Identity(), h, h.clone(), k=5)
    assert perfect["top1"] == pytest.approx(1.0)
    assert perfect["top1_lift"] > 0.5, perfect

    unrelated = interpret.paired_consistency(Identity(), h, torch.randn(200, 16) * 3,
                                             k=5)
    assert unrelated["top1_lift"] < 0.15, unrelated


def test_paired_consistency_control_is_not_trivially_zero():
    """With few active coordinates, unrelated texts collide -- that's the point.

    If the control were always ~0 it would not be worth computing; it is worth
    computing precisely because a sparse code over a skewed usage distribution
    gives unrelated texts a real collision rate.
    """
    torch.manual_seed(1)

    class Peaked:
        def __call__(self, h):
            return torch.softmax(h * 8, dim=-1)

    # Only 3 coordinates ever win, so mismatched pairs agree about 1/3 of the time.
    h = torch.randn(400, 3) * 2
    r = interpret.paired_consistency(Peaked(), h, torch.randn(400, 3) * 2, k=2)
    assert r["top1_control"] > 0.15, r
