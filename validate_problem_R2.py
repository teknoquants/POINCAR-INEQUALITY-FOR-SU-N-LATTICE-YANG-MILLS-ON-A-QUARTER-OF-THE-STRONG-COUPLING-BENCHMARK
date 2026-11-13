# MIT License
#
# Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Validation of the computations quoted in Problem R2 (is the benchmark Hessian bound sharp in d >= 3?).

Problem R2 is an open question; nothing here proves or disproves it. What is checked:
  (a) the exact SU(2) lattice Hessian code agrees with finite differences (3^3 and 4^3, random configurations);
  (b) at the all-minus-one configuration the largest Hessian eigenvalue of W is 4 d N beta (sum_p x(U_p): 2d) for
      d = 2, 3, 4, below the Lemma 1 bound 8(d-1) N beta (4(d-1)) when d >= 3, and equal to it when d = 2;
  (c) the arithmetic of the question: if sup Hess W were 4 d N beta, the single-scale threshold would be 1/(8d),
      which is 1/32 in d = 4 against 1/48;
  (d) the reported search: hill climbing from random starts and from the all-minus-one configuration on 4^2 and 4^3
      (SU(2)) finds no configuration whose top eigenvalue exceeds 2d; random starts stay below it (the fraction reached
      depends on the length of the search; the longer run quoted in the paper reached about 80% on 4^3). The search covers
      a negligible part of configuration space; it is reported as an observation, not evidence.
"""
import sys

import numpy as np

from common import Report, Lattice, rand_su2_batch, exp_su2_batch, second_diff_vec


def climb(L, U, steps, rng):
    best = np.linalg.eigvalsh(L.hessian(U))[-1]
    start = best
    for _ in range(steps):
        m = rng.random(L.nE) < rng.choice([0.02, 0.1, 0.5, 1.0])
        if not m.any():
            continue
        V = U.copy()
        V[m] = V[m] @ rand_su2_batch(int(m.sum()), rng, 0.3 * rng.choice([0.1, 0.3, 1.0]))
        t = np.linalg.eigvalsh(L.hessian(V))[-1]
        if t > best + 1e-12:
            best, U = t, V
    return start, best


def main():
    rep = Report("Problem R2")
    rng = np.random.default_rng(2101)

    worst = 0.0
    for dims in ((3, 3, 3), (4, 4, 4)):
        L = Lattice(dims)
        U = rand_su2_batch(L.nE, rng)
        H = L.hessian(U)
        for _ in range(3):
            v = rng.normal(size=3 * L.nE)
            fd = second_diff_vec(lambda s: L.action(U @ exp_su2_batch(s * v)))
            worst = max(worst, abs(fd - v @ H @ v) / abs(fd))
    rep.check("(a) exact lattice Hessian agrees with finite differences", worst < 1e-7, f"max rel error {worst:.1e}")

    ok, det = True, []
    for dims in ((4, 4), (4, 4, 4), (4, 4, 4, 4)):
        L = Lattice(dims)
        d = L.d
        top = np.linalg.eigvalsh(L.hessian(L.all_minus_one()))[-1]
        ok &= abs(top - 2 * d) < 1e-9 and (top < 4 * (d - 1) - 1e-9 if d >= 3 else abs(top - 4 * (d - 1)) < 1e-9)
        det.append(f"d={d}: {top:.6f} (bound {4 * (d - 1)})")
    rep.check("(b) top eigenvalue at all-minus-one = 2d (i.e. 4dN beta for W), below 4(d-1) for d >= 3", ok, "; ".join(det))

    ok = True
    for d in (2, 3, 4):
        for N in (2, 3):
            beta_star = (N / 2) / (4 * d * N)          # N/2 - 4 d N beta = 0
            ok &= abs(beta_star - 1 / (8 * d)) < 1e-16
    ok &= abs(1 / 32 - 1 / (8 * 4)) < 1e-16
    ok &= all(1 / (8 * d) > 1 / (16 * (d - 1)) for d in (3, 4, 5)) and abs(1 / (8 * 2) - 1 / 16) < 1e-16
    rep.check("(c) sup Hess W = 4dN beta would give the threshold 1/(8d) (1/32 in d = 4, > 1/48)", ok)

    ok, det = True, []
    for dims, steps in (((4, 4), 300), ((4, 4, 4), 150)):
        L = Lattice(dims)
        d = L.d
        s0, b = climb(L, rand_su2_batch(L.nE, rng), steps, rng)
        s1, b1 = climb(L, L.all_minus_one(), steps // 3, rng)
        ok &= b <= 2 * d + 1e-9 and b1 <= 2 * d + 1e-9
        det.append(f"{'x'.join(map(str, dims))}: random start {s0:.2f} -> {b:.2f} ({100 * b / (2 * d):.0f}% of 2d); from all-minus-one {b1:.2f}")
    rep.check("(d) search: no configuration above 2d found (observation only, not evidence of a global bound)", ok, "; ".join(det))
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
