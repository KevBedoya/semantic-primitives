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
| 1 | Env, corpus, embedding cache, LaBSE baseline table | done — `results/baseline/` |
| 2 | Non-amortised ceiling per `N` | done — `results/ceiling_cosine/` |
| 3 | Head + align/contrast/reconstruct, Hellinger, `N` sweep | done — `results/sweep/` |
| 4 | Learned `B`, sparsity annealing, interpretability diagnostics | next |
| 5 | *Conditional:* grounding, multi-simplex, or Stage-C fine-tune | — |

### Results so far

**Frozen LaBSE, held-out, P@1** — the row every later table carries:
mean **0.9163** en→xx, **0.9162** xx→en over 6,000 pairs
(es 0.950, zh 0.948, ar 0.910, ru 0.909, de 0.894, hi 0.887).

**Reconstruction ceiling**, cosine objective, 50k sample
(`results/ceiling_cosine/ceiling.json`):

| N | ceiling | PCA@N | simplex cost | % of floor | archetype radius |
|---|---|---|---|---|---|
| 64 | 0.40161 | 0.39873 | +0.00288 | 47.5% | 1.00 |
| 256 | 0.02185 | 0.02059 | +0.00126 | 2.6% | 1.00 |
| 1024 | 0.00014 | 0.00000 | n/a | 0.0% | 1.00 |

The trained head cannot beat these; matching one means `N` is the thing to
change, not the head. `n/a` at N=1024 because a linear code of rank ≥ `d` = 768
reconstructs exactly, so the PCA reference carries no information there.
Reconstruction is close to solved by N=256, so it will not be the binding
constraint — the open question is whether the head reaches the ceiling and
whether the coordinates mean anything.

Every row beats the squared-error ceiling it replaced (0.40499 / 0.02431 /
0.00550) *and* keeps the archetypes on the data sphere.

**Caveat on "interpretable".** Radius 1.00 means the archetypes are no longer
absurd — they sit where texts live rather than 17× outside. It does not mean
they are text-like: the nearest real text to an archetype still has cosine
~0.46–0.52, because archetypes are extreme points of the hull and sit at the
periphery of the cloud rather than on top of any example. Necessary, not
sufficient. The real test is what *activates* each coordinate
(`eval/interpret.py::top_activating`), which is Milestone 4.

**Milestone 3 — the trade curve** (`results/sweep/`, 30 epochs, Hellinger
kernel, held-out P@1):

| N | P@1 | LaBSE | delta | dead | active/text | recon | k-ceiling | ratio |
|---|---|---|---|---|---|---|---|---|
| 64 | 0.8803 | 0.9163 | −0.0360 | 0/64 | 17.8 | 0.38164 | 0.28630 | 1.33× |
| 256 | **0.9032** | 0.9163 | **−0.0132** | 21/256 | 21.5 | 0.32434 | 0.17650 | 1.84× |
| 1024 | 0.8970 | 0.9163 | −0.0193 | 566/1024 | 21.4 | 0.31900 | 0.13032 | 2.45× |

**The compression is nearly free.** At N=256 the bottleneck retrieves within
**1.3 points** of the encoder it compresses, using ~21 active coordinates out of
768 dimensions. The plan expected a substantial loss; there isn't one. That is
the headline result, and it means interpretability here is not being bought with
much accuracy.

**N=256 is the operating point.** N=1024 is *worse* despite 4× the capacity, and
leaves 566/1024 coordinates dead — the extra inventory is not used, so the model
pays the sparsity cost without the benefit. Active-per-text is flat at ~21
regardless of N, which is the entmax setting, not the capacity.

**The ratio column grows with N** (1.33× → 1.84× → 2.45×), and that is the
actionable finding: the head increasingly fails to exploit a larger inventory.
The dead count corroborates it. So above N=256 the head and losses are the
limiting factor, not `N` — which points Milestone 4 at the diversity term and
head capacity rather than at more primitives.

> **`k-ceiling`, not the dense ceiling.** A sparse head compared against a dense
> fit is not compared against a bound. The head activates ~21 of 256
> coordinates; against the dense Milestone-2 ceiling it reads **29.5×**, which
> by the documented interpretation means it cannot compute the mapping. Matched
> to k=21 the same head is **1.84×** — headroom, not failure. Opposite
> conclusions from one run. `train.py` now fits the ceiling at the head's own
> observed support size, and `tests/test_topk_ceiling.py` pins the properties
> that make it a safe bound.

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

## Finding: reconstruct direction, not distance

Milestone 2 changed the design. Fitting the ceiling with squared error against
L2-normalised targets produced archetypes with a **median norm 17.6× the data
radius**, whose nearest real text had cosine ~0.45 out of 50,000 candidates.
That is fatal to the point of the model: column `d_i` is meant to be the
teacher-space meaning of primitive `i`, and a point no text resembles cannot
serve as one — the mixture was reconstructing by cancellation between distant
extremes rather than by combining recognisable concepts.

The cause is arithmetic, not optimisation. A convex combination of unit vectors
has norm ≤ 1, with equality only when the mass sits on coincident vertices,
while the targets have norm exactly 1. Squared error charges the model for a
norm that interpretable, data-adjacent archetypes structurally cannot reach, and
the only way to pay is to push them outward. The uninterpretable decoder was
what the objective asked for.

Three fits at N=256, same data (`scripts/decoder_geometry.py`):

| fit | cosine | archetype radius | nearest-text cos |
|---|---|---|---|
| free (unconstrained `D`) | 0.9879 | 17.59 | 0.447 |
| `D` pinned to unit sphere | 0.7629 | 1.00 | 0.592 |
| sphere + free per-example scale | **0.9892** | **1.00** | 0.503 |

Freeing the scale — identical to scoring cosine, since minimising `‖y − s·r‖²`
over `s` *is* maximising `cos(y, r)` — recovers all the lost direction quality
and slightly beats the unconstrained fit, with archetypes on the data's own
sphere. **Interpretable archetypes are free; the squared-error formulation was
paying for the wrong thing.** `reconstruct_mode: cosine` is now the default.

One trap this surfaced: with a free scale, `(b, D)` and `(cb, D/c)` reconstruct
identically, so archetype norms become redundant parameters and the alternating
fit wanders along that ray — it returned a *worse* optimum than the strictly
more constrained simplex fit (0.04071 vs 0.02431), which is impossible for a
true optimum. `archetypal_fit` now pins `‖d_i‖ = 1` whenever the scale is free,
and `test_free_scale_is_never_worse_than_the_simplex_fit` guards it.

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
