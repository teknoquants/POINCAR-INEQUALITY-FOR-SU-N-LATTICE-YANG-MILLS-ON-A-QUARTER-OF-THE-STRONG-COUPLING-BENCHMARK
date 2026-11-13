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
"""Validation of Lemma 8, Theorem 9, Corollary 10 and their proofs.

Model for the inequalities that involve the true profile: one SU(2) loop, W = c ReTr (a = c), where the full curvature
profile k^(t), its integral and the fluctuation measures can be computed accurately (character basis and quadrature).
Lemma 8
  (a) Variance identity  Q_t(F^2) - (Q_t F)^2 = 2 int_0^t Q_{s,t}|grad Q_{0,s} F|^2 ds  at every grid point (class F);
  (b) gradient estimate with the full-Hessian profile: |grad Q_{0,s} F|^2 <= e^{-2 Lambda^(s)} Q_{0,s}|grad F|^2;
  (c) Poincare inequality Var_nu(g) <= c(t) E_nu|grad g|^2, c(t) = 2 int_0^t e^{-2 Lambda^}, for random non-class g.
Theorem 9
  (d) k^(t) >= G(t, c(t)) at every scale (true profile, three couplings);
  (e) G is decreasing in c;
  (f) comparison: the solution of the closed equation lies below the truth: Lambda^ >= Lambda_ and k^ >= Lambda_'.
Corollary 10
  (g) symbolic first integral of Lambda'' = -4 a^2 e^{-2 Lambda}: (Lambda')^2 - 4a^2 e^{-2 Lambda} is constant;
  (h) the first-entry equation stays above r0 = (N/2)(1 - 8a/N)^{1/2} for a < N/8 and breaks down for a >= N/8;
  (i) r0 = (N/2)(1 - 64 beta (d-1))^{1/2}; endpoint 1/(64(d-1)) = beta_0/4 (1/192 in d = 4); the benchmark constant
      (2/N)(1 - beta/beta_0)^{-1} is smaller than 1/r0 on the whole interval (d = 2..5, N = 2..4);
  (j) on one loop the true profile stays above r0 when a < N/8.
"""
import math
import sys

import numpy as np
import sympy as sp

from common import (Report, Flow, X, d1, integ, su2_grid, retr, heat_kernel_x, haar_sun, su_basis, EpsTable,
                    solve_closed, random_class_function)

B = su_basis(2)


def main():
    rep = Report("Lemma 8 / Theorem 9 / Corollary 10")
    rng = np.random.default_rng(1901)

    # one loop, W = c ReTr = 2 c x; Flow takes S = -W
    def flow(c):
        return Flow(-2 * c * X)

    worst_a = worst_b = 0.0
    for c in (0.2, 0.6):
        F = flow(c)
        C, ts, kap, Lam = F.constant(full=True)
        for _ in range(2):
            f = random_class_function(rng)
            t = 1.2
            ss = np.linspace(0, t, 241)
            integrand = np.array([F.Q(d1(F.Q(f, 0.0, s)) ** 2 / 2, s, t) for s in ss])
            rhs = 2 * np.trapezoid(integrand, ss, axis=0)
            lhs = F.Q(f ** 2, 0.0, t) - F.Q(f, 0.0, t) ** 2
            worst_a = max(worst_a, np.abs(lhs - rhs).max() / np.abs(lhs).max())
            for s in (0.05, 0.4, 1.5):
                g = d1(F.Q(f, 0.0, s)) ** 2 / 2
                bound = np.exp(-2 * np.interp(s, ts, Lam)) * F.Q(d1(f) ** 2 / 2, 0.0, s)
                worst_b = max(worst_b, ((g - bound) / bound.max()).max())
    rep.check("(a) Q_t(F^2) - (Q_t F)^2 = 2 int_0^t Q_{s,t}|grad Q_{0,s}F|^2 ds", worst_a < 2e-4, f"max rel error {worst_a:.1e}")
    rep.check("(b) |grad Q_{0,s}F|^2 <= e^{-2 Lambda^(s)} Q_{0,s}|grad F|^2 (full-Hessian profile)", worst_b < 1e-6, f"max excess {max(worst_b, 0):.1e}")

    Y, Wq = su2_grid()
    xY = retr(Y) / 2
    worst = 0.0
    for c in (0.3, 1.0):
        F = flow(c)
        C, ts, kap, Lam = F.constant(full=True)
        for t in (0.5, 1.5):
            ct = 2 * np.trapezoid(np.exp(-2 * Lam[ts <= t]), ts[ts <= t])
            p = heat_kernel_x(t)
            U = haar_sun(1, 2, rng)[0]
            w = np.exp(c * retr(U @ Y)) * p(xY) * Wq
            nu = w / w.sum()
            for _ in range(10):
                A, Bm = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)), rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
                g = retr(A @ Y) + 0.3 * retr(Bm @ Y) ** 2
                gr = sum((retr(A @ Y @ k) + 0.6 * retr(Bm @ Y) * retr(Bm @ Y @ k)) ** 2 for k in B)
                worst = max(worst, (nu @ g ** 2 - (nu @ g) ** 2) / (ct * (nu @ gr)))
    rep.check("(c) Lemma 8: Var_nu(g) <= c(t) E_nu|grad g|^2 for random non-class g", worst <= 1 + 1e-3, f"max ratio {worst:.3f}")

    eps = EpsTable()
    ok_d, ok_f, det = True, True, []
    for c in (0.1, 0.2, 0.4):
        F = flow(c)
        C, ts, kap, Lam = F.constant(full=True)
        cum = np.concatenate([[0], np.cumsum(0.5 * (2 * np.exp(-2 * Lam[1:]) + 2 * np.exp(-2 * Lam[:-1])) * np.diff(ts))])
        worst = math.inf
        for t, k, cc in zip(ts, kap, cum):
            e1, e2 = eps(t) if t > 0 else (math.inf, math.inf)
            b = c + c * c * cc
            if math.isfinite(e2):
                b = min(b, e2 + (e2 + e1 / math.sqrt(2)) ** 2 * cc)
            worst = min(worst, k - (1 - 2 * b))
        ok_d &= worst >= -1e-4
        sol = solve_closed(c, eps, N=2, tmax=20.0, dt=5e-4)
        Lam_ = np.concatenate([[0], np.cumsum(0.5 * (sol["kappa"][1:] + sol["kappa"][:-1]) * np.diff(sol["t"]))])
        for t in (0.1, 0.5, 1.0, 3.0, 10.0):
            ok_f &= np.interp(t, ts, Lam) >= np.interp(t, sol["t"], Lam_) - 2e-3 and np.interp(t, ts, kap) >= np.interp(t, sol["t"], sol["kappa"]) - 2e-3
        det.append(f"a={c}: min(k^ - G) = {worst:.1e}; C true {C:.3f} <= closed {sol['C']:.3f}")
        ok_f &= C <= sol["C"] * (1 + 1e-3)
    rep.check("(d) true profile k^(t) >= G(t, c(t)) at every scale", ok_d, "; ".join(det))
    ok = True
    for t in (0.3, 1.0, 4.0):
        e1, e2 = eps(t)
        G = lambda cc: 1 - 2 * min(0.3 + 0.09 * cc, e2 + (e2 + e1 / math.sqrt(2)) ** 2 * cc)
        vals = [G(cc) for cc in np.linspace(0, 20, 201)]
        ok &= all(vals[i + 1] <= vals[i] + 1e-15 for i in range(200))
    rep.check("(e) G(t, c) is decreasing in c", ok)
    rep.check("(f) comparison: Lambda^ >= Lambda_, k^ >= Lambda_', and the true constant is below the closed one", ok_f)

    t = sp.symbols("t")
    a = sp.symbols("a", positive=True)
    L = sp.Function("L")(t)
    expr = sp.diff(sp.diff(L, t) ** 2 - 4 * a ** 2 * sp.exp(-2 * L), t)
    expr = sp.simplify(expr.subs(sp.diff(L, t, 2), -4 * a ** 2 * sp.exp(-2 * L)))
    rep.check("(g) symbolic: (Lambda')^2 - 4a^2 e^{-2 Lambda} is a first integral", expr == 0)

    none = lambda tt: (math.inf, math.inf)
    ok, det = True, []
    for N in (2, 3):
        for frac in (0.3, 0.7, 0.95):
            aa = frac * N / 8
            sol = solve_closed(aa, none, N=N, tmax=30.0, dt=5e-4)
            r0 = (N / 2) * math.sqrt(1 - 8 * aa / N)
            ok &= sol["closed"] and sol["kappa"].min() >= r0 - 2e-3
            det.append(f"N={N}, a={frac:.2f}N/8: min {sol['kappa'].min():.4f} >= r0 {r0:.4f}")
        for frac in (1.0, 1.05, 1.3):
            sol = solve_closed(frac * N / 8, none, N=N, tmax=30.0, dt=5e-4)
            ok &= (not sol["closed"]) or sol["kappa"][-1] < 0.05
    rep.check("(h) first-entry equation: Lambda_1' >= r0 for a < N/8; breakdown for a >= N/8", ok, "; ".join(det[:3]) + " ...")

    ok = True
    for d in (2, 3, 4, 5):
        b0 = 1 / (16 * (d - 1))
        ok &= abs(b0 / 4 - 1 / (64 * (d - 1))) < 1e-16
        for N in (2, 3, 4):
            for beta in np.linspace(1e-4, b0 / 4 * 0.999, 50):
                aa = 8 * N * beta * (d - 1)
                r0 = (N / 2) * math.sqrt(1 - 8 * aa / N)
                ok &= abs(r0 - (N / 2) * math.sqrt(1 - 64 * beta * (d - 1))) < 1e-12 and aa < N / 8
                ok &= (2 / N) / (1 - beta / b0) < 1 / r0
    ok &= abs(1 / (64 * 3) - 1 / 192) < 1e-16
    rep.check("(i) r0 = (N/2)(1 - 64 beta(d-1))^{1/2}; endpoint beta_0/4 (1/192 in d = 4); benchmark constant smaller", ok)

    ok, det = True, []
    for c in (0.05, 0.15, 0.24):
        F = flow(c)
        kmin = F.constant(full=True)[2].min()
        r0 = math.sqrt(1 - 4 * c)
        ok &= kmin >= r0 - 1e-6
        det.append(f"a={c}: min k^ = {kmin:.4f} >= r0 = {r0:.4f}")
    rep.check("(j) one loop: the true profile stays above r0 when a < N/8", ok, "; ".join(det))
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
