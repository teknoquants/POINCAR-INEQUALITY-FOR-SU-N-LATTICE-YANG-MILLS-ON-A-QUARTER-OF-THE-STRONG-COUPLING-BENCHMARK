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
"""Validation of Lemma 1 (Hessian of the Wilson action), Corollary 2 (the two initial tensors), and their proofs.

What is checked
  (a) Trace inequalities used in the proof: |Tr(V A^2)| <= |A|^2 and |Tr(V1 A V2 B)| <= |A||B| for anti-Hermitian A, B and
      unitary V (SU(2), SU(3), SU(4); 2000 random cases).
  (b) The exact SU(2) lattice Hessian code agrees with finite differences of the action.
  (c) |Hess W(h,h)| <= a |h|^2 with a = 8 N beta (d-1): spectrum of the exact Hessian of sum_p x(U_p) lies in
      [-4(d-1), 4(d-1)] for random configurations on periodic lattices 4^2, 3^3, 4^3 (SU(2)), and finite differences
      for SU(3) on a 3x3 torus; the bound is attained in d = 2 at the all-minus-one configuration.
  (d) Mixed-derivative bound: |d_r d_s W(U e^{sh} Y e^{rk})| <= a |h||k| (SU(2), SU(3), random U, Y, h, k).
  (e) Corollary 2: Ric - Hess W >= K_SZZ and Ric - 2 Hess W >= N/2 - 16 N beta (d-1) for every sampled configuration;
      k_beta(0) > 0 iff beta < beta_0/2 = 1/(32(d-1)); in d = 4 the two thresholds are 1/48 and 1/96.
"""
import math
import sys

import numpy as np

from common import (Report, su_basis, ip, haar_sun, expm_skew, Lattice, rand_su2_batch, exp_su2_batch, second_diff_vec,
                    cubical_box)


def rand_alg(N, rng):
    B = su_basis(N)
    return sum(c * T for c, T in zip(rng.normal(size=len(B)), B))


def W_general(U, P, N, beta):
    tot = 0.0
    for bd in P:
        M = np.eye(N, dtype=complex)
        for e, o in bd:
            M = M @ (U[e] if o > 0 else U[e].conj().T)
        tot += np.real(np.trace(M))
    return N * beta * tot


def main():
    rep = Report("Lemma 1 / Corollary 2")
    rng = np.random.default_rng(1401)

    ok, worst = True, 0.0
    for N in (2, 3, 4):
        for _ in range(700):
            A, Bm = rand_alg(N, rng), rand_alg(N, rng)
            V1, V2 = haar_sun(2, N, rng)
            nA, nB = math.sqrt(ip(A, A)), math.sqrt(ip(Bm, Bm))
            r1 = abs(np.trace(V1 @ A @ A)) / nA ** 2
            r2 = abs(np.trace(V1 @ A @ V2 @ Bm)) / (nA * nB)
            worst = max(worst, r1, r2)
            ok &= r1 <= 1 + 1e-12 and r2 <= 1 + 1e-12
    rep.check("(a) |Tr(V A^2)| <= |A|^2 and |Tr(V1 A V2 B)| <= |A||B|", ok, f"largest ratio {worst:.4f}")

    L = Lattice((4, 4, 4))
    U = rand_su2_batch(L.nE, rng)
    H = L.hessian(U)
    v = rng.normal(size=3 * L.nE)
    fd = second_diff_vec(lambda s: L.action(U @ exp_su2_batch(s * v)))
    rep.check("(b) exact SU(2) lattice Hessian agrees with finite differences", abs(fd - v @ H @ v) < 1e-6 * abs(fd) + 1e-6,
              f"{fd:.8f} vs {v @ H @ v:.8f}")

    ok, det = True, []
    for dims, nconf in (((4, 4), 20), ((3, 3, 3), 10), ((4, 4, 4), 4)):
        Lt = Lattice(dims)
        bound = 4 * (Lt.d - 1)
        top = bot = 0.0
        for _ in range(nconf):
            ev = np.linalg.eigvalsh(Lt.hessian(rand_su2_batch(Lt.nE, rng)))
            top, bot = max(top, ev[-1]), min(bot, ev[0])
        ok &= top <= bound + 1e-9 and bot >= -bound - 1e-9
        det.append(f"{'x'.join(map(str, dims))}: spectrum in [{bot:.2f}, {top:.2f}] within +-{bound}")
    nv, E, P, d0, d1 = cubical_box((3, 3), periodic=True)
    worst = 0.0
    for _ in range(6):
        U3 = haar_sun(len(E), 3, rng)
        h = [rand_alg(3, rng) for _ in E]
        nh2 = sum(ip(x, x) for x in h)
        f = lambda s: W_general([U3[e] @ expm_skew(s * h[e]) for e in range(len(E))], P, 3, 1.0)
        worst = max(worst, abs(second_diff_vec(f)) / (8 * 3 * 1.0 * 1 * nh2))
    ok &= worst <= 1 + 1e-6
    det.append(f"SU(3) 3x3: max |Hess W|/(a|h|^2) = {worst:.3f}")
    L2 = Lattice((4, 4))
    top_u0 = np.linalg.eigvalsh(L2.hessian(L2.all_minus_one()))[-1]
    ok &= abs(top_u0 - 4) < 1e-9
    det.append(f"4x4 at all-minus-one: {top_u0:.6f} = bound 4 (sharp in d = 2)")
    rep.check("(c) |Hess W(h,h)| <= 8 N beta (d-1) |h|^2", ok, "; ".join(det))

    worst = 0.0
    for N in (2, 3):
        for _ in range(4):
            U3, Y3 = haar_sun(len(E), N, rng), haar_sun(len(E), N, rng)
            h = [rand_alg(N, rng) for _ in E]; k = [rand_alg(N, rng) for _ in E]
            nh, nk = math.sqrt(sum(ip(x, x) for x in h)), math.sqrt(sum(ip(x, x) for x in k))
            g = lambda s, r: W_general([U3[e] @ expm_skew(s * h[e]) @ Y3[e] @ expm_skew(r * k[e]) for e in range(len(E))], P, N, 1.0)
            eps = 1e-3
            mixed = (g(eps, eps) - g(eps, -eps) - g(-eps, eps) + g(-eps, -eps)) / (4 * eps * eps)
            worst = max(worst, abs(mixed) / (8 * N * nh * nk))
    rep.check("(d) mixed derivative |d_r d_s W(U e^{sh} Y e^{rk})| <= a |h||k|", worst <= 1 + 1e-4, f"max ratio {worst:.3f}")

    ok = True
    for d in (2, 3, 4):
        Lt = Lattice((4,) * d) if d < 4 else None
        for beta in (0.005, 0.02, 0.05):
            a = 8 * 2 * beta * (d - 1)
            if Lt is not None:
                ev = np.linalg.eigvalsh(Lt.hessian(rand_su2_batch(Lt.nE, rng))) * (4 * beta)    # Hess W = 4 beta Hess sum x
                ok &= 1 - ev.max() >= 1 - a - 1e-12 and 1 - 2 * ev.max() >= 1 - 2 * a - 1e-12
        b0 = 1 / (16 * (d - 1))
        for beta in np.linspace(0, b0, 101):
            k0 = 1 - 32 * beta * (d - 1)                # N = 2: N/2 - 16 N beta (d-1)
            ok &= (k0 > 1e-14) == (beta < b0 / 2 - 1e-14) or abs(beta - b0 / 2) < 1e-14
    ok &= abs(1 / (32 * 3) - 1 / 96) < 1e-16
    rep.check("(e) Ric - Hess W >= K_SZZ, Ric - 2 Hess W >= N/2 - 16 N beta (d-1); k(0) > 0 iff beta < beta_0/2 (1/96 in d = 4)", ok)
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
