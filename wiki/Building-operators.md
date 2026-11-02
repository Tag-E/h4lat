An `Operator` is one $`H(4)`$-irreducible lattice operator $`\bar\psi \, \Gamma \, D \cdots D \, \psi`$, given by its
Dirac structure and its Clebsch-Gordan coefficients. The physics is explained on the theory page
[Lattice operators](https://github.com/Tag-E/h4lat/wiki/Lattice-operators). Most of the time you will load operators
from the bundled [operator database](https://github.com/Tag-E/h4lat/wiki/Operator-database); this page shows how to
build one by hand and what it contains.

## Constructing an operator

```python
import numpy as np
from h4lat import cg_calc, cg_remapping, Operator, rep_label_list

cg = cg_calc((4, 1), (4, 1))

# Take the first CG matrix for the (4,1) irrep
irep_idx = list(cg.cg_dict.keys())[0]
block = cg.cg_dict[irep_idx][0]
col   = np.round(block[:, 0], decimals=15)

cgmat = cg_remapping(col, n=2)          # reshape to (4,4) tensor

op = Operator(cgmat=cgmat, id=1, X='V',
              irrep=rep_label_list[irep_idx],
              block=1, index_block=1)

print(op)                   # LaTeX expression
print("K =", op.latex_K)   # Kinematic factor
print("C =", op.C)         # C-parity
print("tr =", op.tr)       # Trace condition
print("symm =", op.symm)   # Index symmetry
```

The arguments of `Operator` are:

| Argument | Meaning |
|----------|---------|
| `cgmat` | the Clebsch-Gordan coefficients, an array of shape `(4,)*n` with one index per Lorentz index: those of $`\Gamma`$ (none for S and P, one for V and A, two for T) and one per derivative. Its shape is `()` for $`\bar\psi\psi`$ and $`\bar\psi\gamma_5\psi`$. For T, the first two indices are those of $`\sigma_{\mu\nu}`$, stored only for $`\mu < \nu`$ |
| `X` | the Dirac structure: `'S'` ($`\mathbf{1}`$), `'P'` ($`\gamma_5`$), `'V'` ($`\gamma_\mu`$), `'A'` ($`\gamma_\mu\gamma_5`$) or `'T'` ($`\sigma_{\mu\nu}`$) |
| `id` | an identifier, e.g. the operator ID in the database |
| `irrep`, `block`, `index_block` | the irrep label, the copy of the irrep (from 1) and the column of the CG block (from 1) |
| `gamma_pol_index` | the polarisation projector of the kinematic factor (0–5, default 0) |

## What an operator contains

| Attribute | Content |
|-----------|---------|
| `op.C` | the C-parity, $`+1`$ or $`-1`$ (`'mixed'` for a combination without definite C-parity) |
| `op.tr` | `'= 0'` if all the traces over pairs of indices vanish, else `'!= 0'` |
| `op.symm` | the symmetry under permutations of the indices, e.g. `'Symmetric'` |
| `op.O`, `op.latex_O` | the operator as a SymPy expression and as a LaTeX string (also `str(op)`; `op.display()` renders it in Jupyter) |
| `op.K`, `op.latex_K` | the [kinematic factor](https://github.com/Tag-E/h4lat/wiki/Lattice-operators#kinematic-factor), as a SymPy expression in `m_N`, `E(p)`, `p_1`, `p_2`, `p_3` and as a LaTeX string |
| `op.nder`, `op.n` | the number of derivatives and of Lorentz indices |

`C`, `tr` and `symm` are computed when the operator is created. The symbolic `O` and `K` take most of the time, so they
are computed on first access and then cached: loading the whole database takes about a second.

When there are fewer than two indices to permute (S and P with up to one derivative, V and A without derivatives,
T without derivatives), `op.symm` is `"Symmetric"` (vacuously).

## Kinematic factor

```python
# Evaluate K numerically at given kinematics
K_val = op.evaluate_K(m_value=0.939, E_value=1.0, p1_value=0.0, p2_value=0.0, p3_value=0.3)

# The polarisation matrix used to compute K can be changed (indices 0–5, default 0 = γ₁γ₂):
op.set_polarization_matrix(2)   # switch to Γ_pol = ½(1 + γ₄)(1 − i γ₂γ₃)
```

`evaluate_K` returns the complex value of K. `evaluate_K_real` returns the real number that normalises the
three-point function, which includes the factor $`i^n`$ of the $`n`$ derivatives, and `evaluate_K_gvar` does the same
for `gvar` inputs, propagating their uncertainties. The six polarisation projectors
$`\Gamma_{\text{pol}} = \tfrac{1}{2}(1 + \gamma_4)(1 - i\gamma_a\gamma_b)`$ are, by index:

| Index | 0 | 1 | 2 | 3 | 4 | 5 |
|-------|---|---|---|---|---|---|
| $`(a, b)`$ | (1, 2) | (1, 3) | (2, 3) | (1, 4) | (2, 4) | (3, 4) |

Their matrices are in `Gamma_pol_list`, and the denominators of K in `den_K_list`.

## Linear combinations

Operators can be added, subtracted, and multiplied or divided by numbers, which gives the linear combinations used in
the literature:

```python
from h4lat import Operator_from_database

op = Operator_from_database(26) - Operator_from_database(27)    # vector operators with one derivative, irrep (3,1)
print(op.irrep, op.C)
```

The operands must have the same Dirac structure, number of indices and polarisation projector. The sum of operators
of different irreps (or copies) has `irrep` (or `block`) set to `None`, and a sum has no `id`. The notebook
[`operator_algebra.ipynb`](https://github.com/Tag-E/h4lat/blob/main/examples/operator_algebra.ipynb) shows the
details.

## Lower-level functions

The attributes above are computed by functions that can also be called directly on a `cgmat`: `C_parity`,
`trace_symm`, `index_symm` (and `index_symm_2exchange`, `index_symm_index_fixed`), `symO_from_Cgmat`,
`diracO_from_cgmat` (the Dirac matrix $`M(p)`$), `Kfactor_from_diracO` and `latexO_from_diracO`. See the
[API overview](https://github.com/Tag-E/h4lat/wiki/API-overview).
