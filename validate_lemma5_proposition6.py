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
"""Validation of Lemma 5 (two formulas for the Hessian), Proposition 6 (uniform and non-uniform terms), and their proofs.

Model: one SU(2) loop, W(U) = c ReTr U with c = N beta = 2 beta, so a = c (Lemma 1 for a single edge). All integrals
over SU(2) are computed by deterministic product quadrature (442,368 nodes); log z_t and log p_t are Chebyshev series in
x = Tr/2, so the reference Hessian of log z_t is exact up to quadrature and series truncation.
What is checked (beta = 0.2, 1.0; t = 0.4, 1.0, 2.5; 6 random U and h each)
  (I)  Hess log z_t(h,h) = E_nu[Hess W(h~,h~)] + Var_nu(f_h), with Hess W(h~,h~) = c ReTr(U h^2 Y) and f_h = c ReTr(U h Y);
  (II) Hess log z_t(h,h) = E_nu[m] + Var_nu(ell), with ell, m the first and second s-derivatives of log p_t(e^{-sh} Y);
  bounds |ell| <= eps1 |h|, |m| <= eps2 |h|^2, |grad ell| <= (eps2 + eps1/sqrt2) |h| at every quadrature node;
  the identity d_r d_s phi(e^{-sh} Y e^{rk}) = -Hess phi(L_k, L_h') + (1/2) L_[k,h'] phi (finite differences);
  |[k,h']| <= sqrt2 |k||h'| (random matrices);
  Proposition 6: |E_nu Hess W| <= a|h|^2 and |E_nu m| <= eps2 |h|^2; gradients of single-link observables are
  orthogonal, so |grad sum_e ell_e|^2 = sum_e |grad ell_e|^2 (two-link product, finite differences); the crude bound
  Var(sum_e ell_e) <= (sum_e eps1|h_e|)^2 <= eps1^2 |E| |h|^2.
"""
import math
import sys

import numpy as np

from common import (Report, su2_grid, retr, heat_kernel_x, heat, X, class_fn_x, eps12, su_basis, ip, haar_sun,
                    expm_skew)

B = su_basis(2)


def unit_alg(rng):
    h = sum(c * T for c, T in zip(rng.normal(size=3), B))
    return h / math.sqrt(ip(h, h))


def main():
    rep = Report("Lemma 5 / Proposition 6")
    rng = np.random.default_rng(1701)
    Y, Wq = su2_grid()
    xY = retr(Y) / 2
    wI = wII = 0.0
    ok_b, ok_m = True, True
    worst_grad = 0.0
    for beta in (0.2, 1.0):
        c = 2 * beta
        for t in (0.4, 1.0, 2.5):
            p = heat_kernel_x(t)
            pv, p1, p2 = p(xY), p.deriv(1)(xY), p.deriv(2)(xY)
            phi1, phi2 = p1 / pv, p2 / pv - (p1 / pv) ** 2
            psi = class_fn_x(np.log(heat(np.exp(2 * c * X), t)))          # W = c ReTr = 2c x on one loop
            e1, e2 = eps12(t)
            for _ in range(6):
                U = haar_sun(1, 2, rng)[0]
                h = unit_alg(rng)
                xU = np.real(np.trace(U)) / 2
                exact = psi.deriv(2)(xU) * (np.real(np.trace(U @ h)) / 2) ** 2 + psi.deriv(1)(xU) * np.real(np.trace(U @ h @ h)) / 2
                UY = U @ Y
                w = np.exp(c * retr(UY)) * pv * Wq
                nu = w / w.sum()
                f = c * retr(U @ h @ Y)
                hess = c * retr(U @ h @ h @ Y)
                formI = nu @ hess + (nu @ f ** 2 - (nu @ f) ** 2)
                trhY = retr(h @ Y) / 2
                ell = -phi1 * trhY
                m = phi2 * trhY ** 2 + phi1 * retr(h @ h @ Y) / 2
                formII = nu @ m + (nu @ ell ** 2 - (nu @ ell) ** 2)
                sc = max(1.0, abs(exact))
                wI, wII = max(wI, abs(formI - exact) / sc), max(wII, abs(formII - exact) / sc)
                ok_b &= np.abs(ell).max() <= e1 * (1 + 1e-3) and np.abs(m).max() <= e2 * (1 + 1e-3)
                g2 = 0.0
                for k in B:
                    dk = phi2 * (retr(Y @ k) / 2) * (-trhY) + phi1 * (-retr(h @ Y @ k) / 2)
                    g2 = g2 + dk ** 2
                worst_grad = max(worst_grad, np.sqrt(g2).max() / (e2 + e1 / math.sqrt(2)))
                ok_m &= abs(nu @ hess) <= c * (1 + 1e-9) and abs(nu @ m) <= e2 * (1 + 1e-3)
    rep.check("(I) Hess log z_t = E_nu[Hess W(h~,h~)] + Var_nu(f_h)", wI < 1e-6, f"max rel error {wI:.1e}")
    rep.check("(II) Hess log z_t = E_nu[sum m_e] + Var_nu(sum ell_e)", wII < 1e-6, f"max rel error {wII:.1e}")
    rep.check("(bounds) |ell| <= eps1|h|, |m| <= eps2|h|^2, |grad ell| <= (eps2 + eps1/sqrt2)|h|", ok_b and worst_grad <= 1 + 1e-3,
              f"max |grad ell|/(eps2 + eps1/sqrt2) = {worst_grad:.3f}")

    worst = 0.0
    p = heat_kernel_x(0.8)
    phi = lambda V: math.log(p(np.real(np.trace(V)) / 2))
    for _ in range(20):
        Yv = haar_sun(1, 2, rng)[0]
        h, k = unit_alg(rng), unit_alg(rng)
        hp = Yv.conj().T @ h @ Yv
        e = 1e-3
        g = lambda s, r: phi(expm_skew(-s * h) @ Yv @ expm_skew(r * k))
        mixed = (g(e, e) - g(e, -e) - g(-e, e) + g(-e, -e)) / (4 * e * e)
        d2 = lambda Z: (phi(Yv @ expm_skew(e * Z)) - 2 * phi(Yv) + phi(Yv @ expm_skew(-e * Z))) / e ** 2
        hess_kh = 0.25 * (d2(k + hp) - d2(k - hp))
        comm = k @ hp - hp @ k
        Lc = (phi(Yv @ expm_skew(e * comm)) - phi(Yv @ expm_skew(-e * comm))) / (2 * e)
        worst = max(worst, abs(mixed - (-hess_kh + 0.5 * Lc)))
    okc = all(math.sqrt(ip(k @ hh - hh @ k, k @ hh - hh @ k)) <= math.sqrt(2) * (1 + 1e-12)
              for k, hh in ((unit_alg(rng), unit_alg(rng)) for _ in range(2000)))
    rep.check("(identity) d_r d_s phi(e^{-sh} Y e^{rk}) = -Hess phi(L_k, L_h') + (1/2) L_[k,h'] phi; |[k,h']| <= sqrt2|k||h'|",
              worst < 1e-5 and okc, f"max residual {worst:.1e}")

    rep.check("(Prop 6, means) |E_nu Hess W| <= a|h|^2 and |E_nu m| <= eps2|h|^2", ok_m)

    worst = 0.0
    for _ in range(10):
        Y1, Y2 = haar_sun(2, 2, rng)
        h1, h2 = unit_alg(rng), unit_alg(rng)
        F = lambda V1, V2: np.real(np.trace(h1 @ V1)) + np.real(np.trace(h2 @ V2)) ** 2
        e = 1e-5
        grads = []
        for j in range(2):
            gj = []
            for k in B:
                if j == 0:
                    gj.append((F(Y1 @ expm_skew(e * k), Y2) - F(Y1 @ expm_skew(-e * k), Y2)) / (2 * e))
                else:
                    gj.append((F(Y1, Y2 @ expm_skew(e * k)) - F(Y1, Y2 @ expm_skew(-e * k))) / (2 * e))
            grads.append(np.array(gj))
        norm_sep = math.sqrt(sum(np.sum(g ** 2) for g in grads))
        best = 0.0
        for _ in range(400):
            v = rng.normal(size=6); v /= np.linalg.norm(v)
            k1 = sum(c * T for c, T in zip(v[:3], B)); k2 = sum(c * T for c, T in zip(v[3:], B))
            dd = (F(Y1 @ expm_skew(e * k1), Y2 @ expm_skew(e * k2)) - F(Y1 @ expm_skew(-e * k1), Y2 @ expm_skew(-e * k2))) / (2 * e)
            best = max(best, abs(dd))
            worst = max(worst, abs(dd) - norm_sep)
        v = np.concatenate(grads) / norm_sep
        k1 = sum(c * T for c, T in zip(v[:3], B)); k2 = sum(c * T for c, T in zip(v[3:], B))
        dmax = (F(Y1 @ expm_skew(e * k1), Y2 @ expm_skew(e * k2)) - F(Y1 @ expm_skew(-e * k1), Y2 @ expm_skew(-e * k2))) / (2 * e)
        worst = max(worst, abs(dmax - norm_sep))
    xs = rng.normal(size=(20000, 40)) @ rng.normal(size=(40, 40)) * 0.05
    xs = np.clip(xs, -0.3, 0.3)
    crude = (np.abs(xs).max(axis=0).sum()) ** 2
    ok = np.var(xs.sum(axis=1)) <= crude
    ok &= all((np.ones(n) * 0.3).sum() ** 2 <= 0.3 ** 2 * n * n + 1e-12 for n in (10, 100))
    rep.check("(Prop 6) |grad(l_1 + l_2)|^2 = |grad l_1|^2 + |grad l_2|^2 for observables on different links; crude bound Var(sum ell_e) <= (sum eps1|h_e|)^2",
              worst < 1e-6 and ok, f"max excess of a directional derivative over the norm: {worst:.1e}")
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
