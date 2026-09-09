"""The train/held-out split must stay disjoint.

This is the test that was missing. stage_a.py flagged 6,000 pairs as held out,
train.build_texts returned all 120,000 including them, and the evaluation then
scored the 6,000 the model had trained on. Nothing raised. The numbers looked
plausible -- the bottleneck came in slightly *below* the frozen encoder, which
is what everyone expected -- and the contamination surfaced only when a learned
kernel beat the teacher by four points, which was implausible enough to chase.

Cheap test, whole milestone of invalid results.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import pytest

from semprim.corpus import (assert_disjoint, heldout_by_lang, split_pairs)


def _corpus(n=100, held_from=90):
    return {
        "texts": [f"t{i}" for i in range(2 * n)],
        "pairs": [[2 * i, 2 * i + 1] for i in range(n)],
        "pair_langs": ["es" if i % 2 else "de" for i in range(n)],
        "heldout": [i >= held_from for i in range(n)],
    }


def test_split_is_disjoint_and_exhaustive():
    c = _corpus()
    train, held = split_pairs(c)
    assert len(train) == 90 and len(held) == 10
    assert not (set(train) & set(held))
    assert len(set(train) | set(held)) == len(c["pairs"])


def test_assert_disjoint_catches_the_real_failure():
    """The exact bug: training on every pair while evaluating on a subset."""
    c = _corpus()
    all_pairs = [tuple(p) for p in c["pairs"]]
    _, held = split_pairs(c)
    with pytest.raises(AssertionError, match="held-out pairs are in the training set"):
        assert_disjoint(all_pairs, held)


def test_heldout_by_lang_covers_exactly_the_heldout_pairs():
    c = _corpus()
    _, held = split_pairs(c)
    by_lang = heldout_by_lang(c)
    flat = [p for v in by_lang.values() for p in v]
    assert sorted(flat) == sorted(held)


def test_missing_heldout_field_puts_everything_in_train():
    """An older cache still loads -- but then eval has nothing, which is loud."""
    c = _corpus()
    del c["heldout"]
    train, held = split_pairs(c)
    assert len(train) == len(c["pairs"]) and held == []


def test_real_cache_is_split_correctly_if_present():
    """Guards the actual artefact on disk, not just synthetic input."""
    cache = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "results", "cache", "opus7")
    if not os.path.exists(os.path.join(cache, "corpus.json")):
        pytest.skip("no corpus cache present")
    from semprim.corpus import load_corpus
    train, held = split_pairs(load_corpus(cache))
    assert held, "the real corpus must reserve a held-out set"
    assert_disjoint(train, held)
