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
"""Validation of Proposition 7 (the large-time inequality below the benchmark, uniformly in L) and its proof.

What is checked
  (a) Bakry-Emery condition for the fluctuation measure: on one SU(2) loop with W = c ReTr (a = c), the tilted potential
      Phi(Y) = -W(UY) - log p_t(Y) satisfies Ric + Hess Phi >= K - eps2(t), K = N/2 - a (sampled Hessians, 1000 points
      and directions per case).
  (b) The resulting Poincare inequality: Var_nu(g) <= E_nu|grad g|^2/(K - eps2) for random non-class g (quadrature).
  (c) The conclusion: the exact Hess log z_t(h,h) on one loop never exceeds eps2 + (eps2 + eps1/sqrt2)^2/(K - eps2).
  (d) The table: for SU(2), d = 4 (K = 1 - 48 beta), the smallest T with the lower bound on k_beta at least N/4 = 1/2
      for all t >= T, recomputed on a fine grid, agrees with the printed values to 2%.
  (e) eps2(t) e^{3t/2} -> 2, and T(beta) grows like (2/3) ln(1/K_SZZ) as beta -> 1/48.
"""
import math
import sys

import numpy as np

from common import (Report, su2_grid, retr, heat_kernel_x, heat, X, class_fn_x, eps12, su_basis, ip, haar_sun, expm_skew)

B = su_basis(2)
TABLE = [(200, 1.90), (96, 1.99), (64, 2.18), (56, 2.37), (50, 2.90), (49, 3.24), (48.1, 4.60), (48.01, 6.12)]


def unit_alg(rng):
    h = sum(c * T for c, T in zip(rng.normal(size=3), B))
    return h / math.sqrt(ip(h, h))


def lower(t, K):
    e1, e2 = eps12(t)
    return -math.inf if e2 >= K else 1 - 2 * e2 - 2 * (e2 + e1 / math.sqrt(2)) ** 2 / (K - e2)


def main():
    rep = Report("Proposition 7")
    rng = np.random.default_rng(1801)

    ok, det = True, []
    for beta, t in ((0.1, 1.5), (0.2, 2.0), (0.3, 2.5)):
        c = 2 * beta
        K = 1 - c
        e1, e2 = eps12(t)
        p = heat_kernel_x(t)
        lp = lambda V: math.log(p(np.real(np.trace(V)) / 2))
        U = haar_sun(1, 2, rng)[0]
        Phi = lambda V: -c * np.real(np.trace(U @ V)) - lp(V)
        m = math.inf
        hh = 1e-3
        for V in haar_sun(1000, 2, rng):
            k = unit_alg(rng)
            d2 = (Phi(V @ expm_skew(hh * k)) - 2 * Phi(V) + Phi(V @ expm_skew(-hh * k))) / hh ** 2
            m = min(m, 1 + d2)
        ok &= m >= K - e2 - 1e-4
        det.append(f"beta={beta}, t={t}: min {m:.4f} >= {K - e2:.4f}")
    rep.check("(a) Ric + Hess Phi >= K_SZZ - eps2(t) for the fluctuation measure", ok, "; ".join(det))

    Y, Wq = su2_grid()
    xY = retr(Y) / 2
    ok, worst = True, 0.0
    for beta, t in ((0.1, 1.0), (0.3, 2.5)):
        c = 2 * beta; K = 1 - c
        e1, e2 = eps12(t)
        p = heat_kernel_x(t)
        U = haar_sun(1, 2, rng)[0]
        w = np.exp(c * retr(U @ Y)) * p(xY) * Wq
        nu = w / w.sum()
        for _ in range(20):
            A, Bm = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)), rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            g = retr(A @ Y) + 0.3 * retr(Bm @ Y) ** 2
            gr = sum((retr(A @ Y @ k) + 0.6 * retr(Bm @ Y) * retr(Bm @ Y @ k)) ** 2 for k in B)
            var = nu @ g ** 2 - (nu @ g) ** 2
            ratio = var * (K - e2) / (nu @ gr)
            worst = max(worst, ratio)
    rep.check("(b) Poincare inequality for nu_{U,t} with constant 1/(K_SZZ - eps2)", worst <= 1 + 1e-9, f"max Var/(constant x energy) = {worst:.3f}")

    ok, det = True, []
    for beta in (0.1, 0.3):
        c = 2 * beta; K = 1 - c
        for t in (1.0, 2.0, 3.0):
            e1, e2 = eps12(t)
            if e2 >= K:
                continue
            bound = e2 + (e2 + e1 / math.sqrt(2)) ** 2 / (K - e2)
            psi = class_fn_x(np.log(heat(np.exp(2 * c * X), t)))
            sup = -math.inf
            for U in np.concatenate([haar_sun(600, 2, rng), [np.eye(2), -np.eye(2)]]):
                h = unit_alg(rng)
                xU = np.real(np.trace(U)) / 2
                val = psi.deriv(2)(xU) * (np.real(np.trace(U @ h)) / 2) ** 2 + psi.deriv(1)(xU) * np.real(np.trace(U @ h @ h)) / 2
                sup = max(sup, val)
            ok &= sup <= bound
            det.append(f"beta={beta}, t={t}: sup {sup:.4f} <= {bound:.4f}")
    rep.check("(c) Hess log z_t <= eps2 + (eps2 + eps1/sqrt2)^2/(K - eps2) (one loop, exact Hessians)", ok, "; ".join(det[:3]) + " ...")

    ok, det = True, []
    ts = np.linspace(0.3, 12, 2341)
    E = np.array([eps12(t) for t in ts])
    for inv, Tp in TABLE:
        K = 1 - 48 / inv
        e1, e2 = E[:, 0], E[:, 1]
        vals = np.where(e2 < K, 1 - 2 * e2 - 2 * (e2 + e1 / math.sqrt(2)) ** 2 / np.maximum(K - e2, 1e-300), -np.inf)
        bad = np.flatnonzero(vals < 0.5)
        T = ts[bad[-1] + 1] if len(bad) else ts[0]
        ok &= abs(T - Tp) / Tp < 0.02
        det.append(f"1/{inv}: {T:.2f} ({Tp})")
    rep.check("(d) table of T(beta), SU(2), d = 4", ok, "; ".join(det))

    r = eps12(10.0)[1] * math.exp(15)
    T1 = [Tp for inv, Tp in TABLE if inv == 48.1][0]; T2 = [Tp for inv, Tp in TABLE if inv == 48.01][0]
    K1, K2 = 1 - 48 / 48.1, 1 - 48 / 48.01
    slope = (T2 - T1) / math.log(K1 / K2)
    rep.check("(e) eps2(t) e^{3t/2} -> 2; T(beta) ~ (2/3) ln(1/K_SZZ)", abs(r - 2) < 1e-3 and abs(slope - 2 / 3) < 0.05,
              f"eps2(10) e^15 = {r:.5f}; dT/d ln(1/K) = {slope:.3f}")
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
