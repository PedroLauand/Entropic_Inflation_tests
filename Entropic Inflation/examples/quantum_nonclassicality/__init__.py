"""Generator of quantum triangle strategies that are nonlocal at the probability level.

This subpackage builds quantum probability distributions ``p(a,b,c)`` for the
triangle network from two families of *network-nonlocality* strategies, each
parametrised by the local source dimension ``d``:

* **Token-Counting (TC)** -- ``triangle_token_counting`` -- each source
  distributes a fixed number of tokens; each party counts the tokens it receives.
* **Color-Matching (CM)** -- ``triangle_color_matching`` -- each source carries
  one of ``d`` colours; each party checks whether its sources' colours match.

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

Core helpers:

* ``triangle_probability`` -- contract sources + POVMs into ``p(a,b,c)`` (any d)
* ``entropic_vector``      -- ``p(a,b,c)`` -> the seven observed entropies + a
  quick inequality pre-screen (including Eq. (4))
"""
