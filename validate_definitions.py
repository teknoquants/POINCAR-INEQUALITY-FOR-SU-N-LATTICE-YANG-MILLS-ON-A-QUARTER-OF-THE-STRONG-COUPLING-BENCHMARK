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
"""Validation of Definitions 1-7.

What is checked
  D1  On periodic cubical lattices (d = 2, 3, 4; L = 3, 4) each plaquette has four distinct edges and each edge lies
      in exactly 2(d-1) plaquettes; Ric = (N/2) metric for N = 2..5.
  D2  a = 8 N beta (d-1), K_SZZ = N/2 - a, and K_SZZ > 0 iff beta < beta_0(d) = 1/(16(d-1)).
  D3  (scope statement; nothing to compute) - the target inequality is checked in later scripts.
  D4  k_beta(t) = N/2 - 2 sup Hess log z_t on unit horizontal vectors: on one SU(2) loop the flow code of [1] gives
      the same value from both expressions.
  D5  (profile hypothesis; used in Lemma 4).
  D6  The SU(2) heat kernel p_t is strictly positive, central, has unit Haar integral, and eps1(t), eps2(t) are finite
      and decrease to 0, with eps2(t) ~ 2 exp(-3t/2).
  D7  The fluctuation measure nu_{U,t} is a probability measure; for U = 1 and t -> infinity it tends to the Wilson
      measure, and its mean of a function equals Q_t of that function at U (one SU(2) loop, deterministic quadrature).
"""
import math
import sys

import numpy as np

from common import (Report, cubical_box, su_basis, ip, eps12, heat_kernel_x, su2_grid, retr, haar_sun, Flow, X, heat,
                    class_fn_x)


def main():
    rep = Report("Definitions")
    rng = np.random.default_rng(1301)

    ok, det = True, []
    for dims in ((3, 3), (4, 4), (3, 3, 3), (4, 4, 4), (3, 3, 3, 3)):
        nv, E, P, d0, d1 = cubical_box(dims, periodic=True)
        d = len(dims)
        ok &= all(len({e for e, o in bd}) == 4 for bd in P)
        ok &= set(np.abs(d1).sum(axis=0)) == {2 * (d - 1)}
        det.append(f"{'x'.join(map(str, dims))}: {len(E)} edges")
    for N in range(2, 6):
        B = su_basis(N)
        Xm = sum(c * T for c, T in zip(rng.normal(size=len(B)), B))
        ok &= abs(0.25 * sum(ip(Xm @ T - T @ Xm, Xm @ T - T @ Xm) for T in B) - N / 2 * ip(Xm, Xm)) < 1e-10
    rep.check("D1 four distinct edges per plaquette, 2(d-1) plaquettes per edge; Ric = N/2", ok, "; ".join(det))

    ok = True
    for d in (2, 3, 4, 5):
        for N in (2, 3, 4):
            b0 = 1 / (16 * (d - 1))
            for beta in np.linspace(0, 2 * b0, 41):
                K = N / 2 - 8 * N * beta * (d - 1)
                ok &= (K > 1e-14) == (beta < b0 - 1e-14) or abs(beta - b0) < 1e-14
    rep.check("D2 K_SZZ = N/2 - 8 N beta (d-1) > 0 iff beta < 1/(16(d-1)); d = 4 gives 1/48", ok and abs(1 / (16 * 3) - 1 / 48) < 1e-16)

    ok, det = True, []
    for beta in (0.1, 0.3):
        F = Flow(-2 * beta * X)
        for t in (0.0, 0.2, 1.0):
            psi = class_fn_x(-F.V(t))                      # log z_t as a function of x
            d1p, d2p = psi.deriv(1), psi.deriv(2)
            sup = -np.inf
            Us = np.concatenate([haar_sun(3000, 2, rng), [np.eye(2), -np.eye(2)]])
            for U in Us:
                h = sum(c * T for c, T in zip(rng.normal(size=3), su_basis(2)))
                h = h / math.sqrt(ip(h, h))
                xU = np.real(np.trace(U)) / 2
                xp, xpp = np.real(np.trace(U @ h)) / 2, np.real(np.trace(U @ h @ h)) / 2
                sup = max(sup, d2p(xU) * xp ** 2 + d1p(xU) * xpp)      # Hess log z_t(h, h), exact chain rule
            k_samp = 1 - 2 * sup
            ok &= k_samp >= F.kappa(t) - 1e-6 and k_samp - F.kappa(t) < 2e-2
            det.append(f"beta={beta}, t={t}: {k_samp:.4f} vs {F.kappa(t):.4f}")
    rep.check("D4 N/2 - 2 sup Hess log z_t (sampled on SU(2) matrices) agrees with the profile of the flow code of [1]", ok, "; ".join(det[:3]))

    Y, W = su2_grid()
    x = retr(Y) / 2
    ok, det = True, []
    prev = (math.inf, math.inf)
    for t in (0.3, 0.6, 1.0, 2.0, 4.0, 8.0):
        p = heat_kernel_x(t)
        vals = p(x)
        ok &= vals.min() > 0 and abs(W @ vals - 1) < 1e-8
        e1, e2 = eps12(t)
        ok &= e1 < prev[0] and e2 < prev[1] and math.isfinite(e1)
        prev = (e1, e2)
    g = haar_sun(1, 2, rng)[0]
    p = heat_kernel_x(0.5)
    ok &= np.allclose(p(retr(g @ Y[:200] @ g.conj().T) / 2), p(x[:200]))
    e2_8 = eps12(8.0)[1]
    ok &= abs(e2_8 / math.exp(-12) - 2) < 0.01
    rep.check("D6 p_t > 0, central, unit integral; eps1, eps2 finite and decreasing to 0; eps2(8) = 2 e^{-12}(1 + o(1))", ok,
              f"eps(1) = {eps12(1.0)[0]:.3f}, {eps12(1.0)[1]:.3f}; eps2(8)/e^-12 = {e2_8 / math.exp(-12):.4f}")

    ok = True
    beta = 0.3
    for t in (0.5, 1.5):
        p = heat_kernel_x(t)
        U = haar_sun(1, 2, rng)[0]
        w = np.exp(beta * retr(U @ Y)) * p(x) * W           # W = beta ReTr = 2 beta x on one loop
        nu = w / w.sum()
        ok &= abs(nu.sum() - 1) < 1e-14 and nu.min() >= 0
        # mean of F(UY) under nu equals (Q_t F)(U) for a class function F, Q_t F = P_t(F e^W)/P_t(e^W)
        F = X ** 2 + 0.3 * X
        QF = class_fn_x(heat(F * np.exp(2 * beta * X), t) / heat(np.exp(2 * beta * X), t))
        lhs = nu @ (retr(U @ Y) / 2) ** 2 + 0.3 * (nu @ (retr(U @ Y) / 2))
        ok &= abs(lhs - QF(retr(U) / 2)) < 1e-6
    p = heat_kernel_x(30.0)
    w = np.exp(beta * retr(Y)) * p(x) * W; nu = w / w.sum()
    wil = np.exp(beta * retr(Y)) * W; wil = wil / wil.sum()
    ok &= np.abs(nu - wil).sum() < 1e-8
    rep.check("D7 nu_{U,t} is a probability measure; its mean is Q_t at U; nu_{1,t} -> Wilson measure", ok)
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
