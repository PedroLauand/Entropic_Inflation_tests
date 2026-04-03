"""Linear-algebra helpers for entropic LP construction."""

from __future__ import annotations

import itertools as itt
from functools import lru_cache

import numpy as np
from scipy.sparse import coo_array


def build_elemental_shannon_cone(n: int, M=None):
    if M is None:
        return _build_elemental_shannon_cone_cached(int(n))
    return _build_elemental_shannon_cone_uncached(int(n), M)


@lru_cache(maxsize=None)
def _build_elemental_shannon_cone_cached(n: int):
    return _build_elemental_shannon_cone_uncached(n, None)


def _build_elemental_shannon_cone_uncached(n: int, M=None):
    if M is None:
        x = np.arange(n)
        comb = [list(itt.combinations(x.tolist(), i)) for i in range(1, n + 1)]
        vetor_H = list(itt.chain(*comb))
    else:
        vetor_H = M

    index = {tuple(item): i for i, item in enumerate(vetor_H)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []
    linha = 0

    for k in range(n):
        elemento = tuple(np.arange(n))
        condicao = tuple(i for i in range(n) if i != k)
        rows.extend((linha, linha))
        cols.extend((index[elemento], index[condicao]))
        data.extend((-1, 1))
        linha += 1

    for k in range(n):
        for l in range(k + 1, n):
            cond_rest = [i for i in range(n) if i not in {k, l}]
            if cond_rest:
                cond_sets = [
                    list(itt.combinations(cond_rest, i))
                    for i in range(1, len(cond_rest) + 1)
                ]
                for c in itt.chain(*cond_sets):
                    kc = tuple(sorted((k, *c)))
                    lc = tuple(sorted((l, *c)))
                    klc = tuple(sorted((k, l, *c)))
                    c = tuple(c)
                    rows.extend((linha, linha, linha, linha))
                    cols.extend((index[kc], index[lc], index[klc], index[c]))
                    data.extend((-1, -1, 1, 1))
                    linha += 1

            rows.extend((linha, linha, linha))
            cols.extend((index[(k,)], index[(l,)], index[(k, l)]))
            data.extend((-1, -1, 1))
            linha += 1

    matriz_desig = coo_array(
        (np.asarray(data, dtype=np.int8), (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=(linha, len(vetor_H)),
        dtype=np.int8,
    )
    return matriz_desig, vetor_H


def desigualdades_basicas(n: int, M=None):
    """Backward-compatible alias for the elemental Shannon cone builder."""
    return build_elemental_shannon_cone(n, M=M)
