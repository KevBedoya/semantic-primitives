# Semantic primitives via a simplex bottleneck

A multilingual encoder whose embedding is a convex combination of `N` learned
primitives, with similarity measured by a positive-semidefinite kernel over the
mixture coefficients.

```
h(t) = frozen LaBSE pooled output          cached fp16
z(t) = W2 GELU(W1 h + b1) + b2
a(t) = entmax(z / tau)                     on the simplex: a >= 0, sum(a) = 1
y^(t) = D a(t)                             back into teacher space
```

**v1 answers one question: can a simplex bottleneck learn coordinates that are
language-independent and inspectable at all?** Not "does it beat LaBSE at
retrieval" — it almost certainly will not. The deliverable is the trade curve
(retrieval accuracy vs `N` vs sparsity) plus interpretability diagnostics, and an
honest verdict. Every results table carries a frozen-LaBSE cosine baseline row.

## Two things to get right before reading any output

**`a_i` is a share, not a probability.** It is the fraction of a text's semantic
mass assigned to primitive `i`. On the simplex, raising one coordinate
necessarily lowers the others, so `a_i = 0.3` does not mean "concept `i` is
present with probability 0.3", and two texts with different numbers of active
primitives are not comparable coordinate-by-coordinate. Anything downstream that
treats a coordinate as an independent detector is misreading it.

**Never L2-normalise `a`.** It is a convex combination and normalising destroys
that. If retrieval needs unit vectors, normalise a *kernel feature map* — every
kernel in `kernels.py` exposes `feature_map()`, and `LearnedKernel` /
`MixtureKernel` normalise theirs internally.

## Setup

The environment lives in `.venv/` beside the code, created with conda at an
explicit prefix. **It is deliberately not the machine's base Anaconda env**,
which pins `numpy 1.24.3` for `numba 0.57.1` used by an unrelated project that
runs as a service; installing torch there would likely break it.

```powershell
$conda = "N:\Research\Package-installations\miniconda\installation\Scripts\conda.exe"
& $conda create -y -p .\.venv python=3.11
.\.venv\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121
.\.venv\python.exe -m pip install transformers sentence-transformers datasets accelerate scikit-learn faiss-cpu pyyaml entmax pytest
```

Model and dataset downloads go to `HF_HOME`; point it at this drive, not C:
(which has under 10 GB free):

```powershell
setx HF_HOME "N:\work\semantic-primitives\.hf"
```

## Run

```powershell
$env:HF_HOME    = "N:\work\semantic-primitives\.hf"
$env:PYTHONPATH = "N:\work\semantic-primitives\src"

.\.venv\python.exe -m pytest tests -q                              # invariants, offline
.\.venv\python.exe -m semprim.train --config configs\smoke.yaml    # ~1 min, offline

.\.venv\python.exe scripts\stage_a.py --limit-per-lang 20000       # corpus + cache
.\.venv\python.exe scripts\baseline_table.py                       # frozen-LaBSE row
.\.venv\python.exe scripts\ceiling_sweep.py --n 64 256 1024        # ceiling per N
.\.venv\python.exe -m semprim.train --config configs\base.yaml     # the real thing
```

Run them in that order: `baseline_table` and `ceiling_sweep` both read the cache
`stage_a` writes, and the ceiling is what a trained run's reconstruction number
is interpreted against.

The smoke config uses synthetic data with known latent topics. It proves the
pipeline runs; it proves nothing semantic, and its numbers must not appear in a
results table.

## Milestones

| # | Deliverable | State |
|---|---|---|
| 1 | Env, corpus, embedding cache, LaBSE baseline table | see `results/baseline/` |
| 2 | Non-amortised ceiling per `N` | see `results/ceiling/` |
| 3 | Head + align/contrast/reconstruct, Hellinger, `N` sweep | — |
| 4 | Learned `B`, sparsity annealing, interpretability diagnostics | — |
| 5 | *Conditional:* grounding, multi-simplex, or Stage-C fine-tune | — |

## GPU

One RTX 3060, 12 GB, shared with an unrelated biophysics solver.

| Stage | What | Cost |
|---|---|---|
| A | Encode the corpus once with frozen LaBSE → fp16 cache | ~2 GB, 30-60 min, once |
| B | Train the head over the cache | <1 GB, minutes per run |
| C | *Optional* encoder fine-tune, vocab table frozen | ~1.5 GB + activations, hours |

Stage B is what makes this tractable: with the encoder frozen, a batch is a
lookup into a cached matrix, so the whole sweep over losses, kernels and `N` fits
in an afternoon. Do not fine-tune until Stage B has produced a result worth
defending.

**Sharing the card.** The biophysics solver launches a *cooperative* CUDA kernel
that needs every block resident at once; if a torch process is holding memory,
a solver job submitted with `device="gpu"` **fails outright** rather than merely
running slower. Before a long run:

```powershell
Invoke-RestMethod "http://100.83.174.69:8000/health" | Select-Object -ExpandProperty queue
```

If solver jobs are queued or running, wait, or cap this side with
`torch.cuda.set_per_process_memory_fraction(0.45)`.

## Layout

```
src/semprim/
  model.py        head, entmax bottleneck, linear decoder D
  kernels.py      k_lin, k_H, k_B = R^T R + eps I, mixture; each with feature_map()
  losses.py       align, contrast, reconstruct, ground, sparsity, diversity + diagnostics
  archetypal.py   non-amortised simplex fit: reconstruction ceiling and D's init
  encode.py       Stage A cache, with a corpus fingerprint
  train.py        Stage B loop
  data/           Tatoeba/OPUS, PAWS-X, STS, and an offline synthetic fixture
eval/             retrieval, similarity, hard cases, interpretability
configs/          smoke.yaml, base.yaml
```

## Why the design looks like this

Three decisions that are not obvious from the equations:

**`Q` is absent.** The brief writes `f(t) = Q a(t)` with `Q` orthonormal, but
`Q^T Q = I` gives `f(t)·f(s) = a(t)·a(s)`, so under inner products the
orientation adds no expressive power — and without grounding supervision it is
not identifiable. `Q = I` is the same model with a redundant parameter removed.

**entmax with annealing, not sparsemax from the start.** Sparsemax gives the
exact zeros the design wants, but has zero gradient outside its support: a
primitive that drops out early can never return. Starting at softmax and
tightening keeps coordinates alive while the representation forms.

**`y ≈ D a` on a simplex is archetypal analysis** (Cutler & Breiman, 1994). That
gives a non-amortised baseline that bounds what the head can achieve, an
initialisation for `D`, and a way to tell "the head can't compute this mapping"
apart from "`N` is too small" — two failures with opposite remedies.

## Known limits

**Role binding.** A mixture over primitives has no mechanism to distinguish "the
dog chased the person" from "the person chased the dog". PAWS-X measures it. If
it fails there — the expected outcome — the planned fork is several simplex
distributions (entities, relations, roles, polarity) with a kernel over the
combined representation.

**Interpretability is not guaranteed by the architecture.** Orthogonality and
convexity constrain geometry, not meaning; a unique human-interpretable inventory
needs supervision or extra assumptions (Locatello et al., 2019). The diagnostics
are what would convert "useful compressed embedding" into a defensible claim
about primitives, and they may well return "no".

## References

- Feng et al., 2022 — LaBSE
- Martins & Astudillo, 2016 — sparsemax
- Peters, Martins & Vieira, 2019 — α-entmax
- Jebara et al., 2004 — probability product kernels
- Cutler & Breiman, 1994 — archetypal analysis
- Locatello et al., 2019 — limits of unsupervised disentanglement
