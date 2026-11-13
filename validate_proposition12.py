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
"""Validation of Proposition 12 (why the method stops below the benchmark) and its proof.

What is checked
  (i)   For any profile with k >= -A on [0, t], c(t) = 2 int_0^t e^{-2 Lambda} <= (e^{2At} - 1)/A (500 random profiles),
        with equality for k = -A.
  (i')  With c(t) = (e^{2At} - 1)/A and the SU(2) heat-kernel bounds, the second entry of G tends to 0 when A < 3/2 and
        diverges when A > 3/2; the decay rate of eps1, eps2 is c_2 = 3/2 (the first eigenvalue of -Laplacian on SU(2)).
  (ii)  Proposition 7 needs K_SZZ > eps2(t) > 0, hence beta < beta_0(d): for beta >= beta_0 no scale qualifies.
  (iii) As t -> infinity, the variance and the Dirichlet energy of Y -> f(UY) under nu_{U,t} converge to those of f under
        the Wilson measure (one SU(2) loop, quadrature, random non-class f).
  The closure of Remark 11 occurs with A up to about 2.2 (> 3/2): the margin comes from the profile being negative only
  on part of the interval, which (i') does not see; this is reported, not asserted.
"""
import math
import sys

import numpy as np

from common import Report, eps12, su2_grid, retr, heat_kernel_x, haar_sun, su_basis

B = su_basis(2)


def main():
    rep = Report("Proposition 12")
    rng = np.random.default_rng(2001)

    ok = True
    for _ in range(500):
        A, t = rng.uniform(0.1, 3), rng.uniform(0.1, 3)
        s = np.linspace(0, t, 4001)
        k = -A + rng.uniform(0, 2) * np.sin(rng.uniform(0, 5) * s) ** 2
        Lam = np.concatenate([[0], np.cumsum(0.5 * (k[1:] + k[:-1]) * np.diff(s))])
        c = 2 * np.trapezoid(np.exp(-2 * Lam), s)
        ok &= c <= (math.exp(2 * A * t) - 1) / A * (1 + 1e-6)
    s = np.linspace(0, 1.3, 20001)
    cmin = 2 * np.trapezoid(np.exp(2 * 0.8 * s), s)
    ok &= abs(cmin - (math.exp(2 * 0.8 * 1.3) - 1) / 0.8) < 1e-6
    rep.check("(i) k >= -A on [0,t]  =>  c(t) <= (e^{2At} - 1)/A, equality for k = -A", ok)

    vals = {}
    for A in (1.2, 1.8):
        seq = []
        for t in (6.0, 8.0, 10.0):
            e1, e2 = eps12(t)
            seq.append(e2 + (e2 + e1 / math.sqrt(2)) ** 2 * (math.exp(2 * A * t) - 1) / A)
        vals[A] = seq
    rates = [math.log(eps12(6.0)[j] / eps12(9.0)[j]) / 3 for j in (0, 1)]
    ok = vals[1.2][2] < vals[1.2][0] and vals[1.8][2] > vals[1.8][0] and all(abs(r - 1.5) < 0.01 for r in rates)
    rep.check("(i') second entry -> 0 for A < 3/2 and diverges for A > 3/2; eps1, eps2 decay at rate 3/2", ok,
              f"A=1.2: {vals[1.2][0]:.2e} -> {vals[1.2][2]:.2e}; A=1.8: {vals[1.8][0]:.2e} -> {vals[1.8][2]:.2e}; rates {rates[0]:.4f}, {rates[1]:.4f}")

    ok = True
    for d in (2, 3, 4):
        b0 = 1 / (16 * (d - 1))
        for N in (2, 3):
            for beta in (b0, 1.01 * b0, 1.5 * b0):
                K = N / 2 - 8 * N * beta * (d - 1)
                ok &= all(not (K > eps12(t)[1]) for t in (0.5, 2.0, 10.0, 40.0))
            ok &= N / 2 - 8 * N * (0.99 * b0) * (d - 1) > eps12(40.0)[1]
    rep.check("(ii) Proposition 7 requires K_SZZ > eps2(t), possible only for beta < beta_0", ok)

    Y, Wq = su2_grid()
    xY = retr(Y) / 2
    worst = 0.0
    beta = 0.4
    wil = np.exp(beta * retr(Y)) * Wq; wil /= wil.sum()
    for _ in range(5):
        U = haar_sun(1, 2, rng)[0]
        A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        f = lambda V: retr(A @ V) + 0.2 * retr(A @ V) ** 2
        gradsq = lambda V: sum((retr(A @ V @ k) * (1 + 0.4 * retr(A @ V))) ** 2 for k in B)
        p = heat_kernel_x(25.0)
        w = np.exp(beta * retr(U @ Y)) * p(xY) * Wq; nu = w / w.sum()
        # with X = UY, the law of X under nu_{U,t} tends to the Wilson measure as t -> infinity
        vx = nu @ f(U @ Y) ** 2 - (nu @ f(U @ Y)) ** 2
        vw = wil @ f(Y) ** 2 - (wil @ f(Y)) ** 2
        ex, ew = nu @ gradsq(U @ Y), wil @ gradsq(Y)
        worst = max(worst, abs(vx - vw) / vw, abs(ex - ew) / ew)
    rep.check("(iii) variance and energy under nu_{U,t} converge to the Wilson values as t -> infinity", worst < 1e-8, f"max rel difference at t = 25: {worst:.1e}")
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
