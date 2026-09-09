"""Stage A: build the real corpus and cache its frozen-encoder embeddings.

    python scripts/stage_a.py --limit-per-lang 20000

Writes, under ``--out``:
    corpus.json    texts, per-text language, pair indices, train/heldout split
    vectors.f16    the embedding cache (see semprim.encode)
    manifest.json  corpus fingerprint, so a stale cache cannot be loaded silently

The held-out split exists so the retrieval baseline and every later comparison
are measured on pairs no training run has seen. Held out by *pair*, not by text:
splitting per text would put one side of a translation in train and the other in
eval, which leaks the answer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from semprim.data import LANGUAGES, load_translation_pairs   # noqa: E402
from semprim.encode import encode_corpus, load_cache          # noqa: E402

CORPUS = "corpus.json"


def build(limit_per_lang: int, heldout_per_lang: int, pivot: str = "en"):
    rows = load_translation_pairs(limit_per_lang=limit_per_lang, pivot=pivot)

    by_lang = {}
    for a, b, lang in rows:
        by_lang.setdefault(lang, []).append((a, b))

    texts, langs, pairs, pair_langs, is_heldout = [], [], [], [], []
    for lang, items in by_lang.items():
        # Held-out taken from the tail, which for a streamed corpus is simply a
        # different region of the shard -- not a random sample, but independent
        # of anything the training split saw.
        cut = max(len(items) - heldout_per_lang, 0)
        for i, (a, b) in enumerate(items):
            ia, ib = len(texts), len(texts) + 1
            texts.extend([a, b])
            langs.extend([pivot, lang])
            pairs.append([ia, ib])
            pair_langs.append(lang)
            is_heldout.append(i >= cut)
    return {"texts": texts, "langs": langs, "pairs": pairs,
            "pair_langs": pair_langs, "heldout": is_heldout, "pivot": pivot}


def main(argv=None):
    p = argparse.ArgumentParser(description="Build and cache the corpus.")
    p.add_argument("--limit-per-lang", type=int, default=20000)
    p.add_argument("--heldout-per-lang", type=int, default=1000)
    p.add_argument("--out", default=os.path.join("results", "cache", "opus7"))
    p.add_argument("--model", default="sentence-transformers/LaBSE")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-seq-length", type=int, default=64)
    a = p.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)

    t0 = time.perf_counter()
    print(f"loading OPUS-100, {a.limit_per_lang:,}/language, pivot en ...",
          flush=True)
    corpus = build(a.limit_per_lang, a.heldout_per_lang)
    n_pairs = len(corpus["pairs"])
    n_held = sum(corpus["heldout"])
    print(f"  {n_pairs:,} pairs ({n_held:,} held out), "
          f"{len(corpus['texts']):,} texts, "
          f"{time.perf_counter() - t0:.1f}s")
    for lang in LANGUAGES:
        n = sum(1 for l in corpus["pair_langs"] if l == lang)
        if n:
            print(f"    {lang}: {n:,}")

    with open(os.path.join(a.out, CORPUS), "w", encoding="utf-8") as f:
        json.dump(corpus, f)

    t1 = time.perf_counter()
    print(f"\nencoding {len(corpus['texts']):,} texts with {a.model} ...",
          flush=True)
    encode_corpus(corpus["texts"], a.out, model_name=a.model,
                  batch_size=a.batch_size, max_seq_length=a.max_seq_length)
    dt = time.perf_counter() - t1
    print(f"  {dt / 60:.1f} min "
          f"({len(corpus['texts']) / max(dt, 1e-9):,.0f} texts/s)")

    cache = load_cache(a.out, texts=corpus["texts"], model_name=a.model,
                       max_seq_length=a.max_seq_length)
    print(f"\ncache verified: {cache.shape} {cache.dtype}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
