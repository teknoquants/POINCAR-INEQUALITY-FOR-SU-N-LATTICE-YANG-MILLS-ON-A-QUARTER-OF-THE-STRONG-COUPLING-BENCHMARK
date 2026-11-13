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
"""Validation of Proposition 3 (the profile starts negative on the target interval) and every step of its proof.

What is checked
  (a) Hypothesis on L: a Z_2 edge labelling with every plaquette sum odd exists on even periodic lattices (4^2, 4^3, 4^4,
      6^2, 2x4x6) and does not exist on 3^2 and 3^3 (control: L even is needed).
  (b) U0_e = (-1)^{eta_e} 1 has every plaquette holonomy equal to -1, for SU(2) and SU(4) (N even; -1 lies in SU(N)).
  (c) ReTr(-e^X) = -N - (1/2)Tr(X^2) + O(|X|^3) for traceless anti-Hermitian X (N = 2, 4).
  (d) Hess W(A,A) = N beta sum_p |(d_1 A)_p|^2 at U0: finite differences for SU(2) (4^2, 4^3) and SU(4) (4^2).
  (e) d_1 d_0 = 0; the cochains A_i(x) = c_i (-1)^{x_1+...+x_d} with sum_i c_i = 0 are coclosed (d_0^T A = 0, hence
      horizontal) and eigenvectors of d_1^T d_1 with eigenvalue 4d; lambda_max(d_1^T d_1) = 4d on even tori, d = 2, 3, 4.
  (f) For a unit horizontal h of that form, Hess W(h,h) = 4 d N beta (SU(2): exact lattice Hessian, d = 2, 3, 4; SU(4):
      finite differences, d = 2), so k_beta(0) <= N/2 - 8 N d beta and A >= 8 N d beta - N/2.
  (g) d = 2: 8Nd = 16N(d-1) (the bound equals Corollary 2); d = 4, beta >= 1/48: N/2 - 32 N beta <= -N/6.
"""
import itertools
import math
import sys

import numpy as np

from common import (Report, cubical_box, solve_gf2, su_basis, ip, expm_skew, Lattice, exp_su2_batch, second_diff_vec)


def stagger(dims, edges, c):
    v = np.zeros(len(edges))
    for k, (x, i) in enumerate(edges):
        v[k] = c[i] * (-1) ** sum(x)
    return v / np.linalg.norm(v)


def W_suN(U, P, N, beta):
    tot = 0.0
    for bd in P:
        M = np.eye(N, dtype=complex)
        for e, o in bd:
            M = M @ (U[e] if o > 0 else U[e].conj().T)
        tot += np.real(np.trace(M))
    return N * beta * tot


def main():
    rep = Report("Proposition 3")
    rng = np.random.default_rng(1501)

    ok, det = True, []
    for dims in ((4, 4), (4, 4, 4), (4, 4, 4, 4), (6, 6), (2, 4, 6)):
        d1 = cubical_box(dims, periodic=True)[4]
        ok &= solve_gf2(np.abs(d1), np.ones(d1.shape[0])) is not None
    for dims in ((3, 3), (3, 3, 3)):
        d1 = cubical_box(dims, periodic=True)[4]
        bad = solve_gf2(np.abs(d1), np.ones(d1.shape[0])) is None
        ok &= bad
        det.append(f"{'x'.join(map(str, dims))}: no labelling")
    rep.check("(a) odd plaquette labelling exists on even tori and not on odd ones", ok, "; ".join(det))

    ok = True
    for N, dims in ((2, (4, 4)), (2, (4, 4, 4)), (4, (4, 4))):
        nv, E, P, d0, d1 = cubical_box(dims, periodic=True)
        eta = solve_gf2(np.abs(d1), np.ones(len(P)))
        U0 = [(-1) ** int(e) * np.eye(N, dtype=complex) for e in eta]
        ok &= abs(W_suN(U0, P, N, 1.0) / N + N * len(P)) < 1e-9
        ok &= abs(np.linalg.det(-np.eye(N)) - 1) < 1e-12
    rep.check("(b) U0 has every plaquette holonomy -1 (SU(2), SU(4)); -1 in SU(N) for even N", ok)

    worst = 0.0
    for N in (2, 4):
        B = su_basis(N)
        for _ in range(50):
            Xm = sum(c * T for c, T in zip(rng.normal(size=len(B)), B))
            Xm *= 1e-2 / math.sqrt(ip(Xm, Xm))
            err = np.real(np.trace(-expm_skew(Xm))) - (-N - 0.5 * np.real(np.trace(Xm @ Xm)))
            worst = max(worst, abs(err) / 1e-6)
    rep.check("(c) ReTr(-e^X) = -N - (1/2)Tr(X^2) + O(|X|^3)", worst < 1.0, f"max |remainder|/|X|^3 = {worst:.3f}")

    worst = 0.0
    for dims in ((4, 4), (4, 4, 4)):
        L = Lattice(dims)
        U0 = L.all_minus_one()
        H = L.hessian(U0)
        for _ in range(5):
            v = rng.normal(size=3 * L.nE)
            A = v.reshape(-1, 3)
            curl2 = sum(np.sum((sum(o * A[e] for e, o in zip(L.P[p], (1, 1, -1, -1)))) ** 2) for p in range(len(L.P)))
            fd = second_diff_vec(lambda s: L.action(U0 @ exp_su2_batch(s * v)))
            worst = max(worst, abs(fd - curl2 / 2) / curl2, abs(v @ H @ v - curl2 / 2) / curl2)   # sum x = sum ReTr/2: Hess = (1/2) sum |d1 A|^2
    nv, E, P, d0, d1 = cubical_box((4, 4), periodic=True)
    eta = solve_gf2(np.abs(d1), np.ones(len(P)))
    U0 = [(-1) ** int(e) * np.eye(4, dtype=complex) for e in eta]
    B4 = su_basis(4)
    for _ in range(2):
        Ae = [sum(c * T for c, T in zip(rng.normal(size=15), B4)) for _ in E]
        curl2 = sum(ip(Y, Y) for Y in (sum(o * Ae[e] for e, o in bd) for bd in P))
        fd = second_diff_vec(lambda s: W_suN([expm_skew(s * Ae[e]) @ U0[e] for e in range(len(E))], P, 4, 1.0))
        worst = max(worst, abs(fd - 4 * 1.0 * curl2) / curl2)
    rep.check("(d) Hess W(A,A) = N beta sum_p |(d_1 A)_p|^2 at U0 (SU(2), SU(4))", worst < 1e-6, f"max rel error {worst:.1e}")

    ok, det = True, []
    for dims in ((4, 4), (4, 4, 4), (4, 4, 4, 4)):
        nv, E, P, d0, d1 = cubical_box(dims, periodic=True)
        d = len(dims)
        ok &= np.abs(d1 @ d0).max() == 0
        c = rng.normal(size=d); c -= c.mean()
        a = stagger(dims, E, c)
        ok &= np.abs(d0.T @ a).max() < 1e-12 and np.abs(d1.T @ (d1 @ a) - 4 * d * a).max() < 1e-12
        if len(E) <= 300:
            lam = np.linalg.eigvalsh(d1.T @ d1)[-1]
        else:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import eigsh
            S = csr_matrix(d1); lam = eigsh(S.T @ S, k=1, which="LA")[0][0]
        ok &= abs(lam - 4 * d) < 1e-8
        det.append(f"d={d}: lambda_max = {lam:.6f}")
    rep.check("(e) d_1 d_0 = 0; staggered coclosed cochains are horizontal eigenvectors; lambda_max = 4d", ok, "; ".join(det))

    ok, det = True, []
    for dims in ((4, 4), (4, 4, 4), (4, 4, 4, 4)):
        L = Lattice(dims)
        d = L.d
        U0 = L.all_minus_one()
        c = rng.normal(size=d); c -= c.mean()
        a = stagger(dims, L.E, c)
        X0 = rng.normal(size=3); X0 /= np.linalg.norm(X0)
        v = np.kron(a, X0)
        H = L.hessian(U0)
        val = v @ H @ v * 4                      # Hess W = N beta * 2 * Hess(sum x) with N = 2, beta = 1
        ok &= abs(val - 4 * d * 2) < 1e-9
        det.append(f"d={d}: Hess W(h,h)/beta = {val:.6f} = 4dN")
    nv, E, P, d0, d1 = cubical_box((4, 4), periodic=True)
    a = stagger((4, 4), E, np.array([1.0, -1.0]))
    X0 = su_basis(4)[3]
    Ae = [a[e] * X0 for e in range(len(E))]
    fd = second_diff_vec(lambda s: W_suN([expm_skew(s * Ae[e]) @ U0_ for e, U0_ in enumerate(
        [(-1) ** int(x) * np.eye(4, dtype=complex) for x in solve_gf2(np.abs(d1), np.ones(len(P)))])], P, 4, 1.0))
    ok &= abs(fd - 4 * 2 * 4) < 1e-5
    det.append(f"SU(4), d=2: {fd:.6f} = 4dN = 32")
    rep.check("(f) Hess W(h,h) = 4 d N beta on a unit horizontal vector at U0", ok, "; ".join(det))

    ok = all(8 * 2 == 16 * (2 - 1) for _ in [0])
    for N in (2, 4, 6):
        for beta in np.linspace(1 / 48, 1 / 47, 11):
            ok &= N / 2 - 32 * N * beta <= -N / 6 + 1e-12
    rep.check("(g) d = 2: 8Nd = 16N(d-1); d = 4 and beta >= 1/48: k_beta(0) <= N/2 - 32 N beta <= -N/6", ok)
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
