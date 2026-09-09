"""Corpus loading for the seven-language v1.

The language set is typologically diverse on purpose. If the languages were all
Indo-European and Latin-scripted, a coordinate could score well on the alignment
objective by detecting orthography rather than meaning, and every downstream
"cross-lingual primitive" claim would be contaminated. Seven families and five
scripts makes that shortcut unavailable:

    en  English    Germanic,  Latin
    es  Spanish    Romance,   Latin
    de  German     Germanic,  Latin
    ru  Russian    Slavic,    Cyrillic
    zh  Chinese    Sinitic,   Han
    ar  Arabic     Semitic,   Arabic
    hi  Hindi      Indo-Aryan, Devanagari

Everything here except ``synthetic_pairs`` needs network access on first call and
caches under HF_HOME. ``synthetic_pairs`` needs neither, which is what lets the
unit tests and the smoke config run offline and in seconds -- if the only way to
exercise the training loop were a multi-GB download, it would stop being
exercised.
"""

from __future__ import annotations

import random

LANGUAGES = ("en", "es", "de", "ru", "zh", "ar", "hi")


def usable_pair(a: str, b: str, min_chars: int = 12,
                max_chars: int = 400) -> bool:
    """Reject pairs that would teach the model the wrong thing.

    OPUS-100 is web- and subtitle-derived and genuinely noisy. Inspecting the
    first few rows of each language turns up ``'!' || '!'``,
    ``'[SCREAMING]' || 'ऊबड़ .'`` and ``'Vega.' || '- Vega.'``. Left in, these
    are not merely useless -- they are actively harmful to this particular
    design:

      * A pair of punctuation marks satisfies L_align perfectly for *any*
        alpha, so it contributes gradient that says nothing about meaning.
      * Identical strings on both sides are untranslated boilerplate. They teach
        alignment to reward surface identity, which is the exact shortcut the
        seven-language, five-script setup exists to make unavailable.
      * Very short strings carry too little context for a mixture over semantic
        primitives to be meaningful, and they dominate the corpus by count.

    ``max_chars`` is a cost control rather than a quality one: with
    max_seq_length at 64 tokens, anything past a few hundred characters is
    truncated anyway, so the pair's two halves would no longer be translations
    of each other.
    """
    if not a or not b:
        return False
    a, b = a.strip(), b.strip()
    if len(a) < min_chars or len(b) < min_chars:
        return False
    if len(a) > max_chars or len(b) > max_chars:
        return False
    if a == b:
        return False
    return True


def load_translation_pairs(langs=LANGUAGES, limit_per_lang: int = 20000,
                           pivot: str = "en", split: str = "train",
                           min_chars: int = 12, max_chars: int = 400,
                           dedup: bool = True):
    """Meaning-equivalent pairs for L_align, pivoted through one language.

    Returns ``[(text_pivot, text_other, lang), ...]``, filtered by
    ``usable_pair`` and, with ``dedup``, deduplicated on the pivot side.

    Deduplication is per-language rather than global: the same English sentence
    legitimately appears once per target language, and dropping those would
    remove exactly the cross-language anchors alignment depends on. Within one
    language, though, a repeated pivot is corpus duplication and would silently
    upweight whatever it says.

    Pivoting through English is a practical choice with a real cost: the model
    never sees, say, a Russian-Hindi pair directly, so consistency between two
    non-pivot languages is only ever learned transitively. Worth remembering
    when reading the per-language retrieval table -- non-pivot pairs are the
    harder case and should be expected to lag.
    """
    from datasets import load_dataset

    out = []
    for lang in langs:
        if lang == pivot:
            continue
        pair = f"{pivot}-{lang}" if pivot < lang else f"{lang}-{pivot}"
        ds = load_dataset("Helsinki-NLP/opus-100", pair, split=split,
                          streaming=True)
        seen, taken = set(), 0
        for row in ds:
            tr = row["translation"]
            a, b = tr.get(pivot), tr.get(lang)
            if not usable_pair(a, b, min_chars, max_chars):
                continue
            a, b = a.strip(), b.strip()
            if dedup:
                if a in seen:
                    continue
                seen.add(a)
            out.append((a, b, lang))
            taken += 1
            if taken >= limit_per_lang:
                break
    return out


def load_pawsx(langs=LANGUAGES, split: str = "test", limit_per_lang: int = 2000):
    """PAWS-X: high-lexical-overlap pairs labelled paraphrase / not.

    The load-bearing evaluation. Its negatives are word-scrambled rewrites that
    share almost all their vocabulary and differ in argument structure, so a
    bag-of-primitives mixture -- which has no mechanism for role binding -- should
    fail here even when it does well on retrieval. Getting this number early is
    what turns "the mixture cannot represent who-did-what-to-whom" from an
    admitted caveat into a measurement.

    Returns ``[(sentence1, sentence2, label, lang), ...]`` with label 1 =
    paraphrase. Not available for hi/ar; those are skipped rather than faked.
    """
    from datasets import load_dataset

    available = {"en", "es", "de", "fr", "ja", "ko", "zh"}
    out = []
    for lang in langs:
        if lang not in available:
            continue
        ds = load_dataset("google-research-datasets/paws-x", lang, split=split)
        for i, row in enumerate(ds):
            if i >= limit_per_lang:
                break
            out.append((row["sentence1"], row["sentence2"], int(row["label"]),
                        lang))
    return out


def load_sts(split: str = "test", limit: int | None = None):
    """STS-B: sentence pairs with graded human similarity in [0, 5].

    Returns ``[(s1, s2, score), ...]``. Evaluated by Spearman correlation, so
    only the ranking matters and the absolute scale of a kernel is irrelevant --
    which is what makes it a fair comparison between kernels whose ranges differ.
    """
    from datasets import load_dataset

    ds = load_dataset("mteb/stsbenchmark-sts", split=split)
    out = [(r["sentence1"], r["sentence2"], float(r["score"])) for r in ds]
    return out[:limit] if limit else out


def synthetic_pairs(n: int = 2000, n_topics: int = 16, seed: int = 0):
    """Offline stand-in with known structure, for tests and the smoke run.

    Each item is generated from one of ``n_topics`` latent topics and rendered
    into two different pseudo-languages, so translation pairs, a ground-truth
    topic label, and hard negatives (same wording, different topic) all exist
    without a download.

    This is a plumbing fixture, not a benchmark. It exists so the training loop,
    the caching, the kernels and the metrics can be exercised end to end in
    seconds. Any *semantic* claim has to come from the real corpora above --
    passing on synthetic data proves the code runs, nothing more, and results
    from it should never appear in a results table.
    """
    rng = random.Random(seed)
    vocab_a = [f"wa{i}" for i in range(200)]
    vocab_b = [f"wb{i}" for i in range(200)]
    topic_words = {
        t: (rng.sample(vocab_a, 8), rng.sample(vocab_b, 8))
        for t in range(n_topics)
    }
    out = []
    for _ in range(n):
        t = rng.randrange(n_topics)
        wa, wb = topic_words[t]
        sa = " ".join(rng.choices(wa, k=6))
        sb = " ".join(rng.choices(wb, k=6))
        out.append((sa, sb, t))
    return out
