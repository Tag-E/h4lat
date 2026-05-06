# h4lat

**Clebsch-Gordan coefficients and lattice operators for the hypercubic group H(4)**

---

## Physics background

Lattice QCD calculations of nucleon structure (parton distribution functions, form
factors, moments of PDFs) require constructing operators that transform irreducibly
under the discrete hypercubic group **H(4)** — the symmetry group of a four-dimensional
hypercubic lattice.  H(4) has order 384 (= 4! × 2⁴) and 20 irreducible representations
(irreps) with dimensions 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 6, 6, 6, 6, 8, 8.

The irreps are labelled `(k, l)` where `k` is the dimension and `l` distinguishes
inequivalent irreps of the same dimension:

| Label | Dim | Phys. Structure |   | Label | Dim |  Phys. Structure     |
|-------|-----|-----------------|---|-------|-----|----------------------|
| (1,1) |  1  | Scalar          |   | (4,1) |  4  | Vector (Fundamental) |
| (1,2) |  1  |                 |   | (4,2) |  4  |                      |
| (1,3) |  1  |                 |   | (4,3) |  4  |                      |
| (1,4) |  1  | Pseudoscalar    |   | (4,4) |  4  | Pseudovector         |
| (2,1) |  2  |                 |   | (6,1) |  6  | Tensor               |
| (2,2) |  2  |                 |   | (6,2) |  6  |                      |
| (3,1) |  3  |                 |   | (6,3) |  6  |                      |
| (3,2) |  3  |                 |   | (6,4) |  6  |                      |
| (3,3) |  3  |                 |   | (8,1) |  8  |                      |
| (3,4) |  3  |                 |   | (8,2) |  8  |                      |

**Clebsch-Gordan (CG) coefficients** for H(4) are the change-of-basis matrices that
decompose a tensor product of irreps into a direct sum of irreps.  They are needed to
project lattice correlation functions onto states of definite irrep, which in turn map
onto specific moments of parton distributions via the OPE.

This library provides:

* A pre-computed **CG database** (`.npy` files, bundled with the package) for all
  tensor products encountered in standard moments of PDFs calculations.
* A **calculator** (`cg_calc`) that loads from the database or computes new CG
  coefficients on demand using the projection formula of
  [Sakata (1974)](https://doi.org/10.1063/1.1666528).
* An **Operator** class that assembles lattice operators from CG matrices,
  computes their kinematic factors, C-parity, index symmetry, and provides LaTeX output.

---

## Installation

```bash
pip install h4lat                   # core (numpy + sympy + tqdm only)
pip install "h4lat[full]"           # + pylatex, IPython, matplotlib, gvar, h5py, pandas
pip install "h4lat[notebook]"       # + IPython
pip install "h4lat[operators]"      # + pandas, gvar, h5py, IPython
```

For development:

```bash
git clone <repo-url>
cd h4lat
pip install -e ".[full]"
```

---

## Quick start

### Tensor-product decomposition

```python
from h4lat import get_multiplicities, latex_print_multiplicities, rep_label_list

# Decompose (4,1) ⊗ (4,1)
muls = get_multiplicities((4, 1), (4, 1))
for i, m in enumerate(muls):
    if m > 0:
        print(f"  {rep_label_list[i]}  ×{m}")

# Get a LaTeX string
print(latex_print_multiplicities((4, 1), (4, 1)))
```

### Loading CG coefficients

```python
from h4lat import cg_calc

# Load from the bundled database (no computation)
cg = cg_calc((4, 1), (4, 1))

# cg.cg_dict   : dict[irrep_index -> list[ndarray]]
# cg.mul_list  : list of multiplicities (length 20)

for irep_idx, matrices in cg.cg_dict.items():
    from h4lat import rep_label_list
    print(f"Irrep {rep_label_list[irep_idx]}:  {len(matrices)} matrix/matrices, "
          f"shape {matrices[0].shape}")
```

### Computing new CG coefficients

```python
# Compute the (6,1) ⊗ (4,1) ⊗ (4,1) tensor product (tensor operator, with two derivatives)
cg = cg_calc((6, 1), (4, 1), (4, 1))
```

Pass `cgdatabase="/path/to/custom/db"` to store results outside the package.

### Constructing lattice operators

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

### Building an operator database

```python
from h4lat import make_operator_database

make_operator_database(
    operator_folder="my_operators",
    max_n=3,        # V and A up to 3 indices; T up to 4 indices
    verbose=True,
)
```

---

## Module overview

| Module | Contents |
|--------|----------|
| `h4lat.cg_calculator` | `cg_calc`, `get_multiplicities`, group constants, symmetry helpers |
| `h4lat.moments_operator` | `Operator`, database I/O, kinematic factor utilities |
| `h4lat.kinematic_data` | Dirac gamma matrices, symbolic momenta, polarisation matrix |
| `h4lat.utilities` | Permutation parity, perfect-square test, all-equal check |

---

## References

* Baake et al. (1982) — H(4) irreps, character table, generator matrices:
  *J. Math. Phys.* **23**, 944.  <https://doi.org/10.1063/1.525461>
* Sakata (1974) — CG coefficient projection formula:
  *J. Math. Phys.* **15**, 1702.  <https://doi.org/10.1063/1.1666528>
* Göckeler et al. (1996) — lattice operators:
  *Phys. Rev. D* **54**, 5705.  <https://doi.org/10.1103/PhysRevD.54.5705>
  arXiv: [hep-lat/9602029](https://arxiv.org/abs/hep-lat/9602029)

---

## Acknowledgements

This library was developed with the assistance of [Claude](https://www.anthropic.com/claude)
(Anthropic), an AI assistant, which helped with code generation, testing, and documentation.

---

## License

GNU General Public License v3 — see [LICENSE](LICENSE).
All derivative works must be distributed under the same open-source terms.
