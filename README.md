# Validation scripts: A Volume-Uniform Poincaré Inequality for SU(N) Lattice Yang–Mills on a Quarter of the Strong-Coupling Benchmark

Companion code to the white paper by R. Cooperman and A.E. Petrunina V.V. (XISOSIS, Vienna, 13 November 2026).
Released under the MIT License (see `LICENSE`; the licence is also reproduced at the top of every script).

GitHub: https://github.com/teknoquants/POINCAR-INEQUALITY-FOR-SU-N-LATTICE-YANG-MILLS-ON-A-QUARTER-OF-THE-STRONG-COUPLING-BENCHMARK · DOI: https://doi.org/10.5281/zenodo.24203367

## Run

```
pip install numpy scipy sympy mpmath
python run_all.py
```

Python 3.10+, NumPy ≥ 2.0, SciPy ≥ 1.12, SymPy. Runtime is about 3 minutes.
`run_log.txt` is the output of a full run: 56 checks, all passing.

## Coverage

| Script | Paper item | Checks |
|---|---|---|
| `validate_definitions.py` | Definitions 1–7 | Lattice incidence (four edges per plaquette, 2(d−1) plaquettes per edge); Ric = N/2; the benchmark algebra; the profile k_β computed two independent ways; SU(2) heat kernel positivity, centrality, normalization and the decay ε₂(t) ≈ 2e^(−3t/2); the fluctuation measures |
| `validate_lemma1_corollary2.py` | Lemma 1, Corollary 2 | The two trace inequalities (SU(2)–SU(4)); **exact lattice Hessian vs finite differences**; \|∇²W\| ≤ 8Nβ(d−1) on random configurations (4², 3³, 4³ for SU(2); SU(3) on 3²), sharp in d = 2; the mixed-derivative bound; thresholds β₀ and β₀/2 |
| `validate_proposition3.py` | Proposition 3, all proof steps | The odd plaquette labelling exists on even tori, **and not on odd ones**; all plaquettes −𝟙 for SU(2), SU(4); the expansion of ReTr(−e^X); the Hessian at U⁰ by finite differences (SU(2), SU(4)); staggered coclosed eigenvectors and λ_max = 4d (d = 2, 3, 4); ∇²W(h,h) = 4dNβ; the bound −N/6 on the target interval |
| `validate_lemma4.py` | Lemma 4 | The closed form against quadrature; the extremal profile attains it and 500 admissible profiles stay below; monotonicity; r ≤ N/2; the lower bound (2/N)e^(NT/3) |
| `validate_lemma5_proposition6.py` | Lemma 5, Proposition 6 | **Both Hessian formulas against the exact Hessian of log z_t to 10⁻¹⁰** (deterministic quadrature on SU(2)); the bounds on ℓ, m and ∇ℓ at every node; the Lie-derivative identity in the proof; the mean bounds; orthogonality of single-link gradients |
| `validate_proposition7.py` | Proposition 7 and its table | The Bakry–Émery condition for the fluctuation measures; their Poincaré inequality on non-class functions; **the large-time bound against exact Hessians**; the T(β) table recomputed to 2%; logarithmic growth of T(β) |
| `validate_lemma8_theorem9_corollary10.py` | Lemma 8, Theorem 9, Corollary 10 | The variance identity; the gradient estimate on all vectors; **Lemma 8 on non-class functions**; the true profile above G(t, c(t)); monotonicity of G; **the comparison principle (solution below the truth)**; symbolic first integral; r₀ and breakdown at a = N/8; the endpoint 1/(64(d−1)); the benchmark constant is better |
| `validate_remark11.py` | Remark 11 | The two closing thresholds (1/109.5, 1/85.8) by bisection with two discretizations; every entry of the table; the percentages in the text |
| `validate_proposition12.py` | Proposition 12 | The bound on c(t) for profiles ≥ −A; the A = 3/2 dichotomy and decay rate 3/2; Proposition 7 needs β < β₀; the large-time limit of the fluctuation measures |
| `validate_problem_R2.py` | Problem R2 | Lattice Hessian vs finite differences; top eigenvalue 2d at U⁰ against the bound 4(d−1); the threshold 1/(8d); the hill-climbing search, reported as an observation only |

`common.py` holds all shared tools (character basis and flow, su(N) algebra, cubical complexes, GF(2) solver, SU(2)
heat kernel and ε₁, ε₂, SU(2) quadrature, exact lattice Hessian, solver for the closed inequality).

## What is not validated

No inequality is claimed for β ≥ 1/(16(d−1)), and none is tested. Problems R1–R4 are open. Uniformity in the lattice
size is proved, and computation cannot test it; inequalities that involve the true curvature profile are therefore tested
on one SU(2) loop, where that profile can be computed, and lattice identities on periodic lattices up to 4⁴ (SU(2)) and
small lattices for SU(3) and SU(4). The endpoints 1/109.5 and 1/85.8 and the table of Remark 11 are numerical. The
search in Problem R2 is not evidence for a global bound. Passing these checks supports the proofs; it does not replace
them, and it says nothing about a continuum limit or a mass gap.
