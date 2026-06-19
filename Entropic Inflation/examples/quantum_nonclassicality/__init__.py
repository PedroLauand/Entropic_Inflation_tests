"""Generator of quantum triangle strategies that are nonlocal at the probability level.

This subpackage builds quantum probability distributions ``p(a,b,c)`` for the
triangle network from two families of *network-nonlocality* strategies, each
parametrised by the local source dimension ``d``:

* **Token-Counting (TC)** -- each source distributes a fixed number of tokens;
  each party counts the tokens it receives.
* **Color-Matching (CM)** -- each source carries one of ``d`` colours; each
  party checks whether its two sources' colours match.

Both live in ``coherent_strategies`` as ``token_counting(dim, angles)`` and
``color_matching(dim, angles)``: one shared block-diagonal real-rotation core,
differing only in the source and the sector partition.

These are the two general strategies of Renou & Beigi, *Network Nonlocality via
Rigidity of Token-Counting and Color-Matching* (arXiv:2202.00905); the qubit TC
case coincides with the triangle distribution of Renou, Baeumer, Boreiri,
Brunner, Gisin, Beigi, PRL 123, 140401 (2019) (arXiv:1905.04902). By rigidity,
these families contain members that **provably cannot be reproduced by any
classical (trilocal) model**, i.e. are nonlocal at the probability level, for
appropriate (coherent) measurement choices and at any dimension ``d``. (The
decohered / "coarse" variants are, by contrast, classically simulable.)

The purpose here is to use these as a *generic, dimension-scalable source of
known-nonlocal quantum distributions* and feed their observed entropy vectors
into the classical-triangle entropic tests of the parent package -- the seven
triangle inequalities and the spiral inequality Eq. (4) -- to study whether
probability-level nonlocality leaves a Shannon-entropic signature.

Modules:

* ``coherent_strategies`` -- the generator: ``token_counting`` / ``color_matching``
  ``(dim, angles)``, plus ``describe`` / ``angle_layout`` / ``random_angles``
* ``triangle_probability`` -- contract sources + POVMs into ``p(a,b,c)`` (any d)
* ``entropic_vector``      -- ``p(a,b,c)`` -> the seven observed entropies + a
  quick inequality pre-screen (including Eq. (4))
* ``validate_strategies``  -- checks vs the papers + a survey across dimensions
* ``optimize_angles``      -- minimise the Eq. (4) slack over measurement angles
"""
