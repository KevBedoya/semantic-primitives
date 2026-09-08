"""Stage A: freeze the encoder, embed the corpus once, cache to disk.

THIS IS THE DECISION THAT MAKES THE PROJECT TRACTABLE
-----------------------------------------------------
With the encoder frozen, every training batch is a lookup into a cached matrix:
no tokenisation, no transformer forward, no gradients through 471M parameters.
The trainable model is a 768 -> hidden -> N head, so a full run takes minutes
rather than hours, and the entire sweep over losses, kernels and N fits in an
afternoon. Fine-tuning the encoder (Stage C) is a decision to make *after* the
cheap sweep has produced a result worth defending -- not before.

The cache is fp16. LaBSE embeddings are L2-normalised with entries of order
1e-2, far inside fp16's range, and the downstream consumer immediately
normalises again; halving the file (and the page cache pressure) is worth more
than the last three decimal digits. tests assert agreement to fp16 tolerance.

A cache that silently belongs to a different corpus or a different model is the
expensive failure here -- every number downstream would be wrong and nothing
would raise. So each cache carries a manifest fingerprinting the model, the
text count and a digest of the texts themselves, and loading refuses on a
mismatch.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict

import numpy as np

DEFAULT_MODEL = "sentence-transformers/LaBSE"
MANIFEST = "manifest.json"
VECTORS = "vectors.f16"


@dataclass
class CacheManifest:
    model: str
    count: int
    dim: int
    dtype: str
    max_seq_length: int
    text_digest: str

    def matches(self, other: "CacheManifest") -> list[str]:
        """Fields that disagree; empty means the cache is usable."""
        return [f for f in ("model", "count", "dim", "max_seq_length",
                            "text_digest")
                if getattr(self, f) != getattr(other, f)]


def digest_texts(texts) -> str:
    """Order-sensitive digest. Row i of the cache must be texts[i]."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def encode_corpus(texts, out_dir: str, model_name: str = DEFAULT_MODEL,
                  batch_size: int = 64, max_seq_length: int = 64,
                  device: str | None = None, verbose: bool = True) -> str:
    """Embed ``texts`` with a frozen encoder and write the cache. Returns out_dir.

    Streams straight into a memmap rather than accumulating in RAM: at 2M
    sentences a float32 list would be 6 GB before it ever reached disk, and the
    peak is entirely avoidable.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    os.makedirs(out_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length
    model.eval()
    dim = model.get_sentence_embedding_dimension()

    path = os.path.join(out_dir, VECTORS)
    out = np.memmap(path, dtype=np.float16, mode="w+", shape=(len(texts), dim))

    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            vecs = model.encode(chunk, batch_size=batch_size,
                                convert_to_numpy=True,
                                normalize_embeddings=False,
                                show_progress_bar=False)
            out[start:start + len(chunk)] = vecs.astype(np.float16)
            if verbose and start % (batch_size * 200) == 0:
                print(f"  encoded {start + len(chunk):,}/{len(texts):,}",
                      flush=True)
    out.flush()
    del out

    manifest = CacheManifest(model=model_name, count=len(texts), dim=dim,
                             dtype="float16", max_seq_length=max_seq_length,
                             text_digest=digest_texts(texts))
    with open(os.path.join(out_dir, MANIFEST), "w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2)
    if verbose:
        print(f"  wrote {len(texts):,} x {dim} fp16 to {path}")
    return out_dir


def load_cache(out_dir: str, texts=None, model_name: str = DEFAULT_MODEL,
               max_seq_length: int = 64) -> np.memmap:
    """Open a cache read-only, refusing one that does not belong to ``texts``.

    Passing ``texts`` is optional but strongly recommended: without it the only
    thing checked is that the file exists, and a cache from a previous corpus
    would load happily and produce results attributable to nothing.
    """
    with open(os.path.join(out_dir, MANIFEST), encoding="utf-8") as f:
        stored = CacheManifest(**json.load(f))

    if texts is not None:
        expected = CacheManifest(
            model=model_name, count=len(texts), dim=stored.dim,
            dtype=stored.dtype, max_seq_length=max_seq_length,
            text_digest=digest_texts(texts))
        bad = stored.matches(expected)
        if bad:
            raise ValueError(
                f"cache in {out_dir} does not match this corpus; differing "
                f"fields: {', '.join(bad)}. Delete the directory to re-encode.")

    return np.memmap(os.path.join(out_dir, VECTORS), dtype=np.float16,
                     mode="r", shape=(stored.count, stored.dim))


def cache_is_valid(out_dir: str, texts, model_name: str = DEFAULT_MODEL,
                   max_seq_length: int = 64) -> bool:
    """True if a usable cache is already present -- lets callers skip Stage A."""
    if not os.path.exists(os.path.join(out_dir, MANIFEST)):
        return False
    try:
        load_cache(out_dir, texts=texts, model_name=model_name,
                   max_seq_length=max_seq_length)
        return True
    except (ValueError, OSError):
        return False


def ensure_cache(texts, out_dir: str, model_name: str = DEFAULT_MODEL,
                 **kwargs) -> np.memmap:
    """Encode if needed, then return the cache. The normal entry point."""
    max_seq_length = kwargs.get("max_seq_length", 64)
    if not cache_is_valid(out_dir, texts, model_name, max_seq_length):
        encode_corpus(texts, out_dir, model_name=model_name, **kwargs)
    return load_cache(out_dir, texts=texts, model_name=model_name,
                      max_seq_length=max_seq_length)
