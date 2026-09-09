"""The train/held-out split, defined once.

WHY THIS MODULE EXISTS
----------------------
It did not, and the consequence was silent contamination of every retrieval
number in Milestones 3 and 4. ``stage_a.py`` flagged 6,000 pairs as held out in
corpus.json; ``train.build_texts`` loaded that file and returned all 120,000
pairs for training, ignoring the flag; ``sweep_n.heldout_pairs_by_lang`` then
evaluated on the 6,000 the model had just trained on. Two functions, each
correct in isolation, disagreeing about what "held out" meant.

Nothing failed. The numbers looked plausible -- the bottleneck scored slightly
*below* the frozen encoder at first, which is exactly what everyone expected --
and the contamination only became visible when a learned kernel beat the teacher
by four points and that was implausible enough to investigate.

So the split is computed in one place, both consumers call it, and
``tests/test_corpus_split.py`` asserts the two sets are disjoint. A split that
lives in one function cannot drift from itself.
"""

from __future__ import annotations

import json
import os

CORPUS_FILE = "corpus.json"


def load_corpus(cache_dir: str) -> dict:
    with open(os.path.join(cache_dir, CORPUS_FILE), encoding="utf-8") as f:
        return json.load(f)


def split_pairs(corpus: dict):
    """Return ``(train_pairs, heldout_pairs)`` as lists of (i, j) index tuples.

    A corpus without a ``heldout`` field puts everything in train and returns an
    empty held-out set, so an old cache still loads -- but then the evaluation
    has nothing to score, which is loud rather than silent.
    """
    flags = corpus.get("heldout")
    pairs = [tuple(p) for p in corpus["pairs"]]
    if not flags:
        return pairs, []
    train = [p for p, h in zip(pairs, flags) if not h]
    held = [p for p, h in zip(pairs, flags) if h]
    return train, held


def heldout_by_lang(corpus: dict) -> dict:
    """``{lang: [(i, j), ...]}`` over held-out pairs only."""
    flags = corpus.get("heldout") or []
    langs = corpus.get("pair_langs") or []
    out = {}
    for p, h, lang in zip(corpus["pairs"], flags, langs):
        if h:
            out.setdefault(lang, []).append(tuple(p))
    return out


def assert_disjoint(train_pairs, heldout_pairs) -> None:
    """Raise if any held-out pair is in the training set.

    Called on every run rather than only in tests: this is the failure that
    produced a whole milestone of invalid numbers without anything going wrong
    on the surface, so it is worth the set construction each time.
    """
    overlap = set(train_pairs) & set(heldout_pairs)
    if overlap:
        raise AssertionError(
            f"{len(overlap):,} held-out pairs are in the training set. "
            f"Evaluation would score data the model trained on.")
