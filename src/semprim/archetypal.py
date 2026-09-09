"""Non-amortised simplex fitting: the reconstruction ceiling, and D's init.

With a frozen encoder and the teacher being that same encoder, the bottleneck's
reconstruction objective is

    minimise ||y - D a||^2   subject to   a >= 0, sum(a) = 1,

which is archetypal analysis (Cutler & Breiman, Technometrics 1994): every
embedding is expressed inside the convex hull of the columns of D, and those
columns are the archetypes. Two things follow, and both are why this file exists.

**A ceiling.** Fitting ``a`` directly per example, with no encoder in the way,
is strictly easier than predicting it from h(t) with an MLP. Its error therefore
lower-bounds the trained model's at the same N. If the trained head does worse,
the problem is amortisation -- the head cannot compute the mapping -- and no
amount of tuning the losses will fix it. If the head *ties* it, the bottleneck
is doing as well as the geometry allows and N is the thing to change. Without
this number the two failures are indistinguishable, and they have opposite
remedies.

**An initialisation.** Random columns mean D starts as N directions denoting
nothing, and the early epochs are spent dragging them somewhere reasonable.
Seeding from a fit means column i already denotes something before the first
gradient step.

NOTE ON FIDELITY TO THE ORIGINAL METHOD
---------------------------------------
Strict archetypal analysis also constrains each archetype to be a convex
combination of data points. That constraint is dropped here because
``PrimitiveModel.decoder`` is an unconstrained linear map: a ceiling computed
under a *tighter* constraint than the model operates under would not bound the
model at all. The relaxation is deliberate and keeps the comparison honest.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def project_simplex(v: torch.Tensor) -> torch.Tensor:
    """Euclidean projection of each row onto the probability simplex.

    The exact O(N log N) sort-based algorithm (Duchi et al., 2008), not an
    iterative approximation: the ceiling this file computes is only meaningful
    if the constraint is satisfied exactly at every step.
    """
    n = v.shape[-1]
    u, _ = torch.sort(v, dim=-1, descending=True)
    cssv = u.cumsum(dim=-1) - 1.0
    ind = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    cond = u - cssv / ind > 0
    # Index of the last True in each row.
    rho = cond.to(v.dtype).cumsum(dim=-1).argmax(dim=-1, keepdim=True)
    theta = cssv.gather(-1, rho) / (rho.to(v.dtype) + 1.0)
    return (v - theta).clamp_min(0.0)


@torch.no_grad()
def fit_alpha(Y: torch.Tensor, D: torch.Tensor, iters: int = 200,
              lr: float | None = None, tol: float = 1e-10,
              chunk: int | None = None) -> torch.Tensor:
    """Row-chunked wrapper around ``_fit_alpha_block``.

    Each row's fit is independent given D, so chunking is exact -- it changes
    peak memory and nothing else. It matters at the top of the sweep: at
    N = 1024 with 50k samples, FISTA holds x, y, x_new and grad at
    50000 x 1024 fp32 (205 MB each) and ``project_simplex`` adds a sort plus
    several intermediates of the same shape, which is several GB on a card that
    also has to hold the model and may be sharing with the solver.
    """
    if chunk is None or chunk >= Y.shape[0]:
        return _fit_alpha_block(Y, D, iters=iters, lr=lr, tol=tol)
    parts = [_fit_alpha_block(Y[i:i + chunk], D, iters=iters, lr=lr, tol=tol)
             for i in range(0, Y.shape[0], chunk)]
    return torch.cat(parts, dim=0)


@torch.no_grad()
def _fit_alpha_block(Y: torch.Tensor, D: torch.Tensor, iters: int = 200,
                     lr: float | None = None, tol: float = 1e-10) -> torch.Tensor:
    """argmin_a ||Y - a D||^2 over the simplex, by accelerated projected gradient.

    ``Y`` is (n, d), ``D`` is (N, d) with archetypes as rows, result is (n, N).

    FISTA (Beck & Teboulle, 2009) rather than plain projected gradient. That is
    not premature optimisation -- it is a correctness requirement. This routine
    defines the reconstruction *ceiling* that the trained head is measured
    against, so an iteration that stops short does not merely cost time: it
    reports a ceiling that is too high, and the head can then appear to beat a
    bound it cannot actually beat. Plain projected gradient converges as O(1/k)
    and was still at 1.4e-2 after 1000 iterations on data lying exactly inside a
    known hull, where the true optimum is 0. FISTA is O(1/k^2) and reaches 1e-9
    on the same problem.

    The step size is 1/L with L = 2 * lambda_max(D D^T), the true Lipschitz
    constant of the gradient of ||aD - y||^2 -- the factor of 2 comes from the
    squared norm and dropping it puts the step at the very edge of the stability
    region, where the iteration oscillates instead of converging.
    """
    n, N = Y.shape[0], D.shape[0]
    if lr is None:
        gram = (D @ D.t()).double()
        lipschitz = (2.0 * torch.linalg.eigvalsh(gram).max()).clamp_min(1e-12)
        lr = float(1.0 / lipschitz)

    x = torch.full((n, N), 1.0 / N, device=Y.device, dtype=Y.dtype)
    y = x.clone()
    t = 1.0
    for _ in range(iters):
        grad = 2.0 * (y @ D - Y) @ D.t()
        x_new = project_simplex(y - lr * grad)
        t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        shift = (x_new - x).abs().max()
        x, t = x_new, t_new
        if shift < tol:
            break
    return x


@torch.no_grad()
def fit_nonneg(Y: torch.Tensor, D: torch.Tensor, iters: int = 200,
               lr: float | None = None, tol: float = 1e-10,
               chunk: int | None = None) -> torch.Tensor:
    """argmin_b ||Y - b D||^2 over b >= 0, with NO sum-to-one constraint.

    WHY THIS IS THE RIGHT COMPARISON, NOT A DIFFERENT MODEL
    -------------------------------------------------------
    Allowing the decoder a free per-example scale -- reconstructing
    ``s(t) * D a(t)`` with ``a`` on the simplex and ``s > 0`` free -- is exactly
    this problem. Substituting ``b = s * a`` turns "non-negative, sums to one,
    times a free positive scale" into "non-negative", and the simplex code is
    recovered as ``a = b / sum(b)``.

    So a non-negative least squares fit answers the question "how well could an
    interpretable, unit-norm set of archetypes reconstruct, if the loss did not
    also charge for getting the norm right?" -- and it is the same question as
    "what if the reconstruction loss scored direction instead of distance".
    Minimising ||y - s*r||^2 over s is maximising cos(y, r); the two are one
    objective.

    Same FISTA as ``fit_alpha`` with the simplex projection replaced by a clamp
    at zero, which is the projection onto the non-negative orthant.
    """
    if chunk is not None and chunk < Y.shape[0]:
        return torch.cat([fit_nonneg(Y[i:i + chunk], D, iters=iters, lr=lr,
                                     tol=tol)
                          for i in range(0, Y.shape[0], chunk)], dim=0)

    n, N = Y.shape[0], D.shape[0]
    if lr is None:
        gram = (D @ D.t()).double()
        lipschitz = (2.0 * torch.linalg.eigvalsh(gram).max()).clamp_min(1e-12)
        lr = float(1.0 / lipschitz)

    x = torch.full((n, N), 1.0 / N, device=Y.device, dtype=Y.dtype)
    y = x.clone()
    t = 1.0
    for _ in range(iters):
        grad = 2.0 * (y @ D - Y) @ D.t()
        x_new = (y - lr * grad).clamp_min(0.0)
        t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        shift = (x_new - x).abs().max()
        x, t = x_new, t_new
        if shift < tol:
            break
    return x


@torch.no_grad()
def init_archetypes(Y: torch.Tensor, n_primitives: int,
                    method: str = "kmeans", seed: int = 0,
                    iters: int = 25) -> torch.Tensor:
    """Starting archetypes (N, d) from a sample of teacher embeddings.

    k-means centroids are the default. They are not archetypes -- centroids sit
    at the middle of clusters, archetypes at the extremes of the hull -- but
    they are cheap, stable, and cover the data, which is all an initialiser has
    to do. The alternating fit below moves them outward on its own.
    """
    if method != "kmeans":
        raise ValueError(f"unknown method {method!r}")
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(Y.shape[0], generator=g)[:n_primitives]
    C = Y[idx.to(Y.device)].clone()

    for _ in range(iters):
        assign = torch.cdist(Y, C).argmin(dim=1)
        for k in range(n_primitives):
            members = Y[assign == k]
            if members.numel():
                C[k] = members.mean(dim=0)
            # Empty cluster: leave the centroid where it is rather than
            # reseeding. A collapsed centroid here is itself information about
            # whether N is too large for this corpus.
    return C


@torch.no_grad()
def archetypal_fit(Y: torch.Tensor, n_primitives: int, outer: int = 30,
                   inner: int = 100, seed: int = 0,
                   normalize: bool = True, verbose: bool = False,
                   chunk: int | None = None, constrain: str = "none",
                   free_scale: bool = False):
    """Alternating fit of archetypes D and simplex codes A.

    Returns ``(D, A, mse)`` with D (N, d), A (n, N), and mse the mean squared
    reconstruction error per example -- the ceiling for this N.

    ``normalize`` matches ``losses.reconstruction_loss``: the model reconstructs
    L2-normalised teacher embeddings, so the ceiling has to be computed against
    the same targets or the two numbers are not comparable.

    ``constrain="sphere"`` keeps the archetypes on the unit sphere the data
    lives on, which is the interpretable regime -- column d_i then denotes a
    direction real texts occupy. It is not the default because of a geometric
    tension worth stating explicitly:

        a convex combination of unit vectors has norm <= 1, with equality only
        when all the mass sits on coincident vertices.

    So a simplex code over unit-norm archetypes *cannot* reach a unit-norm
    target unless it is effectively one-hot. Under a squared-error loss against
    L2-normalised targets, the only way for the fit to recover the missing norm
    is to push the archetypes outward -- which is exactly the radius blow-up
    measured at N=64 and N=256 (9x and 17x the data radius). The uninterpretable
    archetypes are not a fitting artefact; they are what this objective asks for.

    The remedy is to stop making norm the decoder's problem: score direction
    only (cosine), or allow a free per-example scale. See
    scripts/decoder_geometry.py, which measures both sides of this.
    """
    if normalize:
        Y = F.normalize(Y, dim=-1)

    D = init_archetypes(Y, n_primitives, seed=seed)
    A = None
    mse = float("inf")
    coder = fit_nonneg if free_scale else fit_alpha

    if free_scale and constrain == "none":
        # A FREE SCALE CREATES A DEGENERACY THAT MUST BE FIXED, NOT LEFT ALONE.
        #
        # With b >= 0 unbounded, (b, D) and (c*b, D/c) give identical
        # reconstructions for any c > 0, so the archetype norms are redundant
        # parameters. The alternating fit then wanders along that ray: measured
        # at N=256 it produced a *worse* optimum than the strictly more
        # constrained simplex fit (0.04071 vs 0.02431) with a max archetype
        # radius of 99 -- impossible for the true optimum, since NNLS optimises
        # over a superset of the simplex, and therefore a diagnosis of the
        # optimiser rather than the model.
        #
        # Pinning ||d_i|| = 1 removes the redundancy at no cost to generality,
        # because any scale is absorbed into b. This is the same normalisation
        # that made the sphere+scale fit in scripts/decoder_geometry.py both
        # well-behaved and the best of the three.
        constrain = "sphere"

    for step in range(outer):
        A = coder(Y, D, iters=inner, chunk=chunk)
        # D given A: unconstrained least squares, min_D ||Y - A D||^2.
        # lstsq rather than the normal equations because A is often rank
        # deficient once the codes go sparse, and forming A^T A squares an
        # already poor condition number.
        D = torch.linalg.lstsq(A, Y).solution
        if constrain == "sphere":
            # Project archetypes back onto the unit sphere the data lives on.
            # Alternating projected minimisation: the D-step is no longer the
            # exact minimiser, so this is a heuristic and its error is an upper
            # bound on the sphere-constrained optimum -- which is all it is used
            # for. See the note on the norm tension below.
            D = F.normalize(D, dim=-1)
        elif constrain != "none":
            raise ValueError(f"unknown constraint {constrain!r}")
        mse = ((A @ D - Y) ** 2).sum(dim=-1).mean().item()
        if verbose:
            print(f"  [archetypal] outer {step + 1}/{outer}  mse {mse:.6f}")
    return D, A, mse


@torch.no_grad()
def reconstruction_ceiling(Y: torch.Tensor, n_values, **kwargs) -> dict:
    """Ceiling per N, for the sweep. Returns {N: mse}."""
    return {int(n): archetypal_fit(Y, int(n), **kwargs)[2] for n in n_values}
