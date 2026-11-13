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
"""Run every validation script for the white paper
"A Volume-Uniform Poincare Inequality for SU(N) Lattice Yang-Mills on a Quarter of the Strong-Coupling Benchmark".

Usage:  python run_all.py      (exit code 0 if every check passes)
"""
import sys
import time

import validate_definitions
import validate_lemma1_corollary2
import validate_proposition3
import validate_lemma4
import validate_lemma5_proposition6
import validate_proposition7
import validate_lemma8_theorem9_corollary10
import validate_remark11
import validate_proposition12
import validate_problem_R2

SUITES = [
    ("Definitions 1-7", validate_definitions),
    ("Lemma 1 and Corollary 2 - Hessian of the Wilson action; the two initial tensors", validate_lemma1_corollary2),
    ("Proposition 3 - the profile starts negative on the target interval", validate_proposition3),
    ("Lemma 4 - from a profile to a constant", validate_lemma4),
    ("Lemma 5 and Proposition 6 - two Hessian formulas; the term responsible", validate_lemma5_proposition6),
    ("Proposition 7 - large-time inequality below the benchmark", validate_proposition7),
    ("Lemma 8, Theorem 9, Corollary 10 - the closed inequality and the quarter of the benchmark", validate_lemma8_theorem9_corollary10),
    ("Remark 11 - numerical continuation for SU(2), d = 4", validate_remark11),
    ("Proposition 12 - why the method stops", validate_proposition12),
    ("Problem R2 - the computations behind the open question", validate_problem_R2),
]


def main():
    results = []
    for title, mod in SUITES:
        print(f"\n=== {title} ===")
        t = time.time()
        ok = mod.main()
        results.append((title, ok, time.time() - t))
    print("\n=== Summary ===")
    for title, ok, dt in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {title}  ({dt:.1f} s)")
    print("\nNot validated: any inequality for beta >= 1/(16(d-1)) (none is claimed); Problems R1-R4; the uniformity in L of\n"
          "statements about lattices, which is proved and not testable by computation. Inequalities that involve the true\n"
          "curvature profile are tested on one SU(2) loop, where it can be computed; lattice statements are tested on\n"
          "periodic lattices up to 4^4 for SU(2) and small lattices for SU(3) and SU(4). The endpoints 1/109.5 and 1/85.8\n"
          "and the table of Remark 11 are numerical. Passing these checks supports the proofs; it does not replace them.")
    return all(ok for _, ok, _ in results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
