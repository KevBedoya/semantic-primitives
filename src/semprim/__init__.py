"""Semantic primitives via a simplex bottleneck.

A multilingual encoder whose embedding is a convex combination of N learned
primitives, with similarity measured by a PSD kernel over the mixture
coefficients.

Two invariants that everything here depends on, stated once:

  * ``alpha_i`` is a *share* of semantic mass, not the probability that
    primitive i is present. On the simplex, raising one coordinate necessarily
    lowers the others. Anything downstream reading a coordinate as an
    independent detection is misreading it.
  * Never L2-normalise the bottleneck output. It is a convex combination and
    normalising destroys that. Normalise a kernel feature map instead -- every
    kernel in ``kernels`` exposes one.
"""

__version__ = "0.1.0"
