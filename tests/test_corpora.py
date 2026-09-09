"""The corpus filter, which is a correctness concern rather than tidiness.

OPUS-100 is subtitle- and web-derived. Inspecting the first rows of each
language turns up `'!' || '!'`, `'[SCREAMING]' || 'ऊबड़ .'` and
`'Vega.' || '- Vega.'`. Those are not merely low-value: a pair of punctuation
marks satisfies the alignment loss for *any* alpha, and identical strings on
both sides reward surface identity -- the exact shortcut the seven-language,
five-script setup exists to make unavailable.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import pytest

from semprim.data.corpora import LANGUAGES, synthetic_pairs, usable_pair


@pytest.mark.parametrize("a,b,why", [
    ("!", "!", "punctuation only"),
    ("Vega.", "- Vega.", "too short"),
    ("[SCREAMING]", "ऊबड़ .", "too short"),
    ("", "something long enough to pass", "empty side"),
    (None, "something long enough to pass", "missing side"),
    ("The very same sentence here", "The very same sentence here",
     "identical across languages: untranslated boilerplate"),
    ("x" * 500, "y" * 500, "beyond max_seq_length, so the halves are no "
                           "longer translations of each other once truncated"),
])
def test_rejects_unusable_pairs(a, b, why):
    assert not usable_pair(a, b), f"should have rejected ({why})"


@pytest.mark.parametrize("a,b", [
    ("It was the asbestos in here, that's what did it!",
     "Fueron los asbestos aqui. Eso es lo que ocurrio!"),
    ("The schedule below is tentative and subject to change",
     "Приводимое ниже расписание является предварительным"),
    ("Sixty-first session of the assembly", "第六十一届会议全体会议记录"),
])
def test_accepts_real_pairs(a, b):
    assert usable_pair(a, b)


def test_whitespace_is_stripped_before_length_check():
    """'   ok   ' padded to 12 chars is still too short to be a translation."""
    assert not usable_pair("   hi   ", "a properly long sentence here")


def test_language_set_is_script_diverse():
    """The claim rests on this: same-script languages let a coordinate cheat.

    Seven languages across five scripts. If someone trims this list to make a
    run cheaper, the cross-language interpretability result stops meaning what
    it says, so pin it.
    """
    assert LANGUAGES == ("en", "es", "de", "ru", "zh", "ar", "hi")
    assert len(LANGUAGES) == 7


def test_synthetic_pairs_are_offline_and_shaped_right():
    rows = synthetic_pairs(n=50, n_topics=4, seed=1)
    assert len(rows) == 50
    assert all(len(r) == 3 for r in rows)
    assert {r[2] for r in rows} <= set(range(4))
    # Deterministic, so a smoke run is reproducible.
    assert rows == synthetic_pairs(n=50, n_topics=4, seed=1)
