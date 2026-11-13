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
"""Validation of Lemma 4 (from a profile to a constant) and every step of its proof.

What is checked
  (a) Closed form: 2 int_0^T e^{2At} dt + 2 int_T^inf e^{2AT - 2r(t-T)} dt = (e^{2AT} - 1)/A + e^{2AT}/r, against
      adaptive quadrature on a grid of (A, T, r); the A -> 0 limit is 2T + 1/r.
  (b) The bound is attained by the extremal profile (-A on [0,T), r afterwards) and dominates
      2 int_0^inf exp(-2 int_0^t k) for 500 random profiles satisfying the hypothesis P(beta; T, A, r).
  (c) Monotonicity: replacing a profile by a smaller one can only increase the constant (random pairs).
  (d) r <= N/2: for one SU(2) loop and seven actions the optimal profile of [1] never exceeds N/2 = 1.
  (e) On the target interval (A >= N/6, r <= N/2) the bound is at least (2/N) e^{NT/3}.
  (f) The constant from the criterion of [1] bounds the optimal Poincare constant on one SU(2) loop (Wilson actions).
"""
import math
import sys

import numpy as np
from scipy.integrate import quad

from common import Report, Flow, X, ACTIONS, optimal_constant


def closed(A, T, r):
    return (2 * T if A == 0 else (math.exp(2 * A * T) - 1) / A) + math.exp(2 * A * T) / r


def constant_of(k, tmax=60.0, n=600001):
    t = np.linspace(0, tmax, n)
    kv = k(t)
    Lam = np.concatenate([[0], np.cumsum(0.5 * (kv[1:] + kv[:-1]) * np.diff(t))])
    g = np.exp(-2 * Lam)
    return 2 * np.trapezoid(g, t) + g[-1] / kv[-1]


def main():
    rep = Report("Lemma 4")
    rng = np.random.default_rng(1601)

    worst = 0.0
    for A in (0.0, 1e-6, 0.3, 1.0, 2.5):
        for T in (0.0, 0.4, 1.5, 3.0):
            for r in (0.05, 0.5, 1.0):
                q = 2 * quad(lambda t: math.exp(2 * A * t), 0, T)[0] + 2 * quad(lambda t: math.exp(2 * A * T - 2 * r * (t - T)), T, np.inf)[0]
                worst = max(worst, abs(q - closed(A, T, r)) / closed(A, T, r))
    rep.check("(a) closed form of the two integrals; A = 0 gives 2T + 1/r", worst < 1e-6, f"max rel error {worst:.1e}")

    ok, worst_ratio = True, 0.0
    for _ in range(500):
        A, T, r = rng.uniform(0, 2), rng.uniform(0, 3), rng.uniform(0.1, 1)
        ext = lambda t, A=A, T=T, r=r: np.where(t < T, -A, r)
        Ce = constant_of(ext)
        ok &= abs(Ce - closed(A, T, r)) / closed(A, T, r) < 2e-4
        bump = rng.uniform(0, 3, size=3)
        prof = lambda t, A=A, T=T, r=r, b=bump: np.where(t < T, -A + b[0] * np.sin(t) ** 2, r + b[1] * np.exp(-b[2] * t))
        Cp = constant_of(prof)
        worst_ratio = max(worst_ratio, Cp / closed(A, T, r))
    ok &= worst_ratio <= 1 + 2e-4
    rep.check("(b) the extremal profile attains the bound; 500 admissible profiles stay below it", ok, f"max C/bound {worst_ratio:.4f}")

    ok = True
    for _ in range(100):
        base = rng.uniform(0.2, 1.0)
        k1 = lambda t, b=base: b + 0.3 * np.sin(3 * t)
        k2 = lambda t, b=base: b + 0.3 * np.sin(3 * t) - 0.1 * np.exp(-t)
        ok &= constant_of(k2) >= constant_of(k1)
    rep.check("(c) a smaller profile gives a larger constant", ok)

    ok, det = True, []
    for name, mk in ACTIONS.items():
        F = Flow(mk())
        _, ts, kap, _ = F.constant()
        ok &= kap.max() <= 1 + 1e-9
        det.append(f"{name}: max {kap.max():.4f}")
    rep.check("(d) r <= N/2: the optimal profile never exceeds N/2 = 1 (one loop, seven actions)", ok, "; ".join(det[:3]) + " ...")

    ok = True
    for N in (2, 3, 4):
        for T in (0.0, 0.5, 2.0):
            for A in (N / 6, N / 3, N):
                for r in (0.01, N / 4, N / 2):
                    ok &= closed(A, T, r) >= (2 / N) * math.exp(N * T / 3) * (1 - 1e-12)
    rep.check("(e) A >= N/6, r <= N/2  =>  bound >= (2/N) e^{NT/3}", ok)

    ok, det = True, []
    for beta in (0.1, 0.4, 1.0):
        F = Flow(-2 * beta * X)
        C = F.constant()[0]
        o = optimal_constant(F.S)
        ok &= o <= C
        det.append(f"beta={beta}: optimal {o:.4f} <= {C:.4f}")
    rep.check("(f) the criterion's constant bounds the optimal constant (one loop, Wilson)", ok, "; ".join(det))
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
