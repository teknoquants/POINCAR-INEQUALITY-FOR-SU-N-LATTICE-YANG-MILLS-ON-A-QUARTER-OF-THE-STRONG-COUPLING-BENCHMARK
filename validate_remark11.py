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
"""Validation of Remark 11 (numerical continuation for SU(2), d = 4).

Remark 11 is numerical, so validation means independent re-computation with different discretizations.
What is checked
  (a) the closing threshold of the closed equation with the heat-flow entries only: 1/beta = 109.5 (within 1%),
      located by bisection with two time steps and two eps tables;
  (b) the threshold with the bound of Proposition 7 as a third entry: 1/beta = 85.8 (within 1%);
  (c) the table entries A, T and C (C within 5%, or 25% for the two entries next to a threshold), and the benchmark
      column 1/K_SZZ = 1/(1 - 48 beta);
  (d) the statements in the text: closure stops at 25% (proved, 1/192), about 44% (1/109.5) and about 56% (1/85.8) of the
      benchmark endpoint 1/48; every constant in the table exceeds the benchmark constant.
"""
import math
import sys

from common import Report, EpsTable, solve_closed

TABLE = [  # 1/beta, uses Prop 7, A, T, C   (as printed)
    (192, False, 0.0, 0.0, 2.28), (150, False, 0.16, 1.74, 3.84), (120, False, 1.16, 1.98, 14.9),
    (110, False, 2.24, 2.99, 499.0), (100, True, 3.89, 1.76, 91.2), (90, True, 16.1, 1.79, 6.6e3)]


def threshold(eps, hybrid, lo, hi, dt):
    for _ in range(8):                      # closes for small beta, i.e. for large 1/beta
        mid = 0.5 * (lo + hi)
        a = 48 / mid
        if solve_closed(a, eps, K=(1 - a) if hybrid else None, dt=dt)["closed"]:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def main():
    rep = Report("Remark 11")
    e1 = EpsTable()
    e2 = EpsTable(tmin=0.12, tmax=50.0, n=900)
    th = [threshold(e1, False, 105, 115, 1e-3), threshold(e2, False, 105, 115, 5e-4)]
    rep.check("(a) heat-flow entries only: closes iff 1/beta > 109.5", all(abs(x - 109.5) / 109.5 < 0.01 for x in th),
              ", ".join(f"{x:.2f}" for x in th))
    th2 = [threshold(e1, True, 82, 90, 1e-3), threshold(e2, True, 82, 90, 5e-4)]
    rep.check("(b) with Proposition 7 as a third entry: closes iff 1/beta > 85.8", all(abs(x - 85.8) / 85.8 < 0.01 for x in th2),
              ", ".join(f"{x:.2f}" for x in th2))

    ok, det = True, []
    for inv, hyb, A, T, C in TABLE:
        a = 48 / inv
        s = solve_closed(a, e2, K=(1 - a) if hyb else None, dt=2e-4)
        tolC = 0.25 if inv in (110, 90) else 0.05
        good = s["closed"] and abs(s["A"] - A) <= max(0.02, 0.03 * A) and abs(s["T"] - T) <= max(0.02, 0.03 * T) \
            and abs(s["C"] - C) / C < tolC and s["C"] > 1 / (1 - a)
        ok &= good
        det.append(f"1/{inv}: A={s['A']:.2f} T={s['T']:.2f} C={s['C']:.4g} (benchmark {1 / (1 - a):.2f})")
    rep.check("(c) table entries A, T, C and the benchmark column", ok, "; ".join(det))
    fr = [48 / 192, 48 / 109.5, 48 / 85.8]
    rep.check("(d) closure stops at 25%, ~44% and ~56% of the benchmark endpoint 1/48",
              abs(fr[0] - 0.25) < 1e-12 and abs(fr[1] - 0.44) < 0.005 and abs(fr[2] - 0.56) < 0.005,
              ", ".join(f"{100 * x:.1f}%" for x in fr))
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
