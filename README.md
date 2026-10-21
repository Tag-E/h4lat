# h4lat

**Clebsch-Gordan coefficients and lattice operators for the hypercubic group H(4)**

---

## Contents

- [Physics background](#physics-background)
- [Installation](#installation)
- [Examples](#examples)
- [Quick start](#quick-start)
  - [Tensor-product decomposition](#tensor-product-decomposition)
  - [Loading CG coefficients](#loading-cg-coefficients)
  - [Computing new CG coefficients](#computing-new-cg-coefficients)
  - [Constructing lattice operators](#constructing-lattice-operators)
  - [Loading operators from the bundled database](#loading-operators-from-the-bundled-database)
  - [Operator catalogue (PDF)](#operator-catalogue-pdf)
  - [Operators used in the literature](#operators-used-in-the-literature)
  - [Building an operator database](#building-an-operator-database)
  - [Saving and reloading operators (HDF5)](#saving-and-reloading-operators-hdf5)
- [Module overview](#module-overview)
- [References](#references)
- [Acknowledgements](#acknowledgements)
- [How to cite](#how-to-cite)
- [License](#license)

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
  tensor products encountered in standard moments of PDFs calculations: the operators
  ψ̄ Γ D⋯D ψ with Γ = 𝟙, γ₅, γ_μ, γ_μγ₅, σ_μν (S, P, V, A, T) and zero to three
  derivatives.
* A **calculator** (`cg_calc`) that loads from the database or computes new CG
  coefficients on demand using the projection formula of
  [Sakata (1974)](https://doi.org/10.1063/1.1666528).
* An **Operator** class that assembles lattice operators from CG matrices,
  computes their kinematic factors, C-parity, index symmetry, and provides LaTeX output.

**C-parity of the CG copies.**  When an irrep appears more than once in a tensor product
whose factors after the first are all the same irrep (the derivative indices of an
operator), its copies are chosen with a definite C-parity: each copy is symmetric or
antisymmetric under reversal of the derivative indices, the symmetric ones first.  This is
always possible.  Charge conjugation reverses the derivative indices, which commutes with
H(4), so it maps the copies of an irrep into each other; since it squares to one, they split
into even and odd ones.  Every operator in the bundled database therefore has C = ±1.
(Until October 2026 the copies were chosen without this condition, so some operators had
mixed C-parity, and the operator IDs of irreps with multiplicity > 1 referred to other copies.)

---

## Installation

```bash
pip install h4lat                   # core (numpy + sympy + tqdm only)
pip install "h4lat[full]"           # + pylatex, IPython, matplotlib, gvar, h5py, pandas
pip install "h4lat[notebook]"       # + IPython
pip install "h4lat[operators]"      # + pandas, gvar, h5py, IPython
pip install "h4lat[parallel]"       # + numba, to compute new CG coefficients on all cores
```

For development:

```bash
git clone https://github.com/Tag-E/h4lat.git
cd h4lat
pip install -e ".[full]"
```

---

## Examples

Jupyter notebooks with worked examples are provided in the [`examples/`](examples/) folder:

| Notebook | Description |
|----------|-------------|
| [`quickstart.ipynb`](examples/quickstart.ipynb) | Mirrors this README — tensor-product decomposition, CG coefficients, operator construction, and the bundled database |
| [`operator_algebra.ipynb`](examples/operator_algebra.ipynb) | Linear combinations of operators: addition, subtraction, scalar multiplication, and mixed-irrep behaviour |
| [`operator_catalogue.ipynb`](examples/operator_catalogue.ipynb) | Listing of the most relevant operators of the bundled database — V, A and T with one and two derivatives (IDs 25–80 and 113–336) — organised by number of derivatives, Dirac structure, and H(4) irrep; all the operators are in the [PDF catalogue](#operator-catalogue-pdf) |
| [`paper_operators.ipynb`](examples/paper_operators.ipynb) | Operators used in published works, selected with `get_paper_operators` by INSPIRE texkey or DOI, and the code that builds them from the operators of the bundled database |

A Google Colab notebook with some minimal working examples can be found [here](https://colab.research.google.com/drive/1KFnE0hi4XDNyo2Kg8iDKSSEqY1YQkPGZ?usp=sharing).

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

The bundled database holds the products of X = (1,1), (1,4), (4,1), (4,4) or (6,1)
(scalar, pseudoscalar, vector, axial, tensor) with zero, one, two or three (4,1) factors
(the derivatives).  With no derivatives the "product" is the irrep of X alone, e.g.
`cg_calc((6, 1))`, whose CG matrix is the identity.  The raw symbolic blocks from which
the CG matrices are derived (`cg.raw_cg`, read on first access, needed by
`prescription_changed=True`) are bundled only for products of up to three irreps (up to
two derivatives): for the products of four irreps they would take tens of MB, so
recompute them with `force_computation=True` (about a minute) if you need them.

### Computing new CG coefficients

```python
# Compute the (6,1) ⊗ (4,1) ⊗ (4,1) tensor product (tensor operator, with two derivatives)
cg = cg_calc((6, 1), (4, 1), (4, 1))

# Products of four irreps take about a minute; parallel=True sums over the group on all cores
cg = cg_calc((6, 1), (4, 1), (4, 1), (4, 1), parallel=True)

# ... or on 8 threads only (giving n_threads implies parallel=True)
cg = cg_calc((6, 1), (4, 1), (4, 1), (4, 1), n_threads=8)
```

The projection formula sums over the 384 group elements numerically, on the coefficients
of the free symbols, so a product of four irreps such as (6,1) ⊗ (4,1)³ takes about a minute
(it took days while the sum was done symbolically, until October 2026; the result is the same
to the last bit).  With `parallel=True` the sum runs on all CPU cores with
[numba](https://numba.pydata.org) (`pip install "h4lat[parallel]"`); use fewer threads with
`n_threads=` (between 1 and the number of cores, or the environment variable `NUMBA_NUM_THREADS`
if set).  The result does not depend on `parallel` or on the number of threads.  By default (`parallel=None`) products of three or more irreps are computed in
parallel when numba is installed.  Most of the remaining time goes into the symbolic raw blocks
saved in `{product}_raw/`, which are built serially.

New results are written to `./cg_database` in the working directory (H(4) matrices
regenerated with `force_h4gen=True` go to `./h4_ele`), never into the installed package.
Later calls read the bundled data first, then `./cg_database`.  To update the bundled
data, copy these folders into `src/h4lat/data/`, or regenerate them with
`python scripts/regenerate_data.py` (see `--help`), then refresh the PDF catalogue with
[`h4lat-catalogue`](#operator-catalogue-pdf).  Pass `cgdatabase="/path/to/custom/db"`
to read and write a custom database instead.

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

### Loading operators from the bundled database

The package ships with a pre-built operator database of 1360 operators ψ̄ Γ D⋯D ψ: every
Dirac structure S (Γ = 𝟙), P (γ₅), V (γ_μ), A (γ_μγ₅) and T (σ_μν), with zero to three
derivatives.  The IDs run over the number of derivatives and, for each, over S, P, V, A, T
(the 16 operators without derivatives first, then 64 with one, 256 with two, 1024 with
three):

| Derivatives | S | P | V | A | T |
|-------------|---|---|---|---|---|
| 0 | 1 | 2 | 3–6 | 7–10 | 11–16 |
| 1 | 17–20 | 21–24 | 25–40 | 41–56 | 57–80 |
| 2 | 81–96 | 97–112 | 113–176 | 177–240 | 241–336 |
| 3 | 337–400 | 401–464 | 465–720 | 721–976 | 977–1360 |

The two convenience getters below load it without any path configuration.  The symbolic
expression `op.O`, the kinematic factor `op.K` and their LaTeX forms are computed when
first accessed, so loading the whole database takes about a second:

```python
from h4lat import get_OperatorList, get_OperatorDict

# --- Flat list, sorted by operator id ---
ops = get_OperatorList()

print(f"Total operators in the bundled database: {len(ops)}")

# Inspect the first operator
op = ops[0]
print(f"id={op.id}  X={op.X}  irrep={op.irrep}  block={op.block}")
print(f"Kinematic factor K = {op.latex_K}")
print(f"C-parity  = {op.C}")
print(f"Trace     = {op.tr}")
print(f"Symmetry  = {op.symm}")

# Evaluate K numerically at given kinematics
K_val = op.evaluate_K(m_value=0.939, E_value=1.0, p1_value=0.0, p2_value=0.0, p3_value=0.3)
print(f"K (numerical) = {K_val}")

# The polarisation matrix used to compute K can be changed (indices 0–5, default 0 = γ₁γ₂):
op.set_polarization_matrix(2)   # switch to Γ_pol = ½(1 + γ₄)(1 − i γ₂γ₃)

# --- A selection, by Dirac structure X and number of derivatives nder (or keys "<X><nder>") ---
local = get_OperatorList(nder=0)              # ψ̄ψ, ψ̄γ₅ψ, ψ̄γ_μψ, ψ̄γ_μγ₅ψ, ψ̄σ_μνψ (IDs 1–16)
scalars = get_OperatorList(X="S", nder=2)     # ψ̄ D_μ D_ν ψ (IDs 81–96)
some = get_OperatorList(keys="V1 A1")         # V and A with one derivative (IDs 25–56)
```

Operators without Lorentz indices (ψ̄ψ and ψ̄γ₅ψ) have a 0-index `op.cgmat`, of shape `()`.
When there are fewer than two indices to permute (S and P with up to one derivative, V and A
without derivatives, T without derivatives), `op.symm` is `"Symmetric"` (vacuously).

```python
# --- Nested dict, keyed by (n, X) then (irrep, block) ---
# n is the number of Lorentz indices: nder for S/P, nder + 1 for V/A, nder + 2 for T
d = get_OperatorDict()

# All 2-index vector operators in irrep (6,1), multiplicity block 1:
# (4,1)⊗(4,1) decomposes into (1,1)⊕(3,1)⊕(6,1)⊕(6,3); (6,1) is the antisymmetric piece.
ops_V2 = d[(2, 'V')][(6, 1), 1]
for op in ops_V2:
    print(f"  id={op.id}  symm={op.symm}  C={op.C}")

# All 3-index axial operators in irrep (8,1), block 1:
if (3, 'A') in d and ((8, 1), 1) in d[(3, 'A')]:
    ops_A3 = d[(3, 'A')][(8, 1), 1]
    print(f"Number of A3 operators in (8,1) block 1: {len(ops_A3)}")
```

Both functions are thin wrappers that call the more general
`OperatorList_from_database` / `OperatorDict_from_database` without a path,
so they always read from `OPERATOR_DATABASE` (the bundled path).  Pass an
explicit path to the underlying functions (which take the same `X`, `nder` and
`keys` selection) if you have generated a custom database with
`make_operator_database`.

**Operator IDs before October 2026.**  Until October 2026 the bundled database held only the
V, A and T operators with one to three derivatives (IDs 1–1176).  Adding the S and P
operators and the operators without derivatives shifted every ID; the operators themselves
are unchanged.  An old ID of an operator with one, two or three derivatives maps to a new
one as follows:

| Old IDs | Derivatives | New ID |
|---------|-------------|--------|
| 1–56 | 1 | old + 24 |
| 57–280 | 2 | old + 56 |
| 281–1176 | 3 | old + 184 |

At the same time the C-parity of the P operators was corrected: the pseudoscalar density
ψ̄γ₅ψ is C-even (as the π⁰), so `C_parity` now gives η (−1)^nder with η = +1 for S, P and A
and −1 for V and T.  P operators had not been bundled before, so no bundled operator changed.

### Operator catalogue (PDF)

To browse the database, write a PDF catalogue of its operators.  For each operator it
lists the expression, the kinematic factor, the C-parity, the trace condition and the
index symmetry, grouped by Dirac structure, number of indices and H(4) irrep.

Computing the kinematic factors takes most of the time, so by default the catalogue holds
the S, P, V, A and T operators with zero, one and two derivatives (IDs 1–336: about 30 s on
a laptop, 20 s with `-j`; the whole database takes about a minute with `-j`).  Select other
operators by Dirac structure (`-X`) and number of derivatives (`--nder`), or with keys made
of the Dirac structure followed by the number of derivatives (`--keys V2 V3 A3`: `V2` is the
vector operators with two derivatives, i.e. three Lorentz indices).  `-j` computes the
kinematic factors on several processes (`-j` alone: all cores):

```bash
h4lat-catalogue                           # S, P, V, A, T with 0, 1, 2 derivatives -> ./operator_catalogue_SPVAT_nder012.pdf
h4lat-catalogue -X V --nder 1 2 3         # vector operators with 1 to 3 derivatives -> operator_catalogue_V_nder123.pdf
h4lat-catalogue --nder 0                  # the local bilinears ψ̄ Γ ψ -> operator_catalogue_SPVAT_nder0.pdf
h4lat-catalogue --keys V2 V3 A3 -j        # a selection, on all cores -> operator_catalogue_V2_V3_A3.pdf
h4lat-catalogue --nder 0 1 2 3 -j         # the whole bundled database
h4lat-catalogue -o docs/ops.pdf --open    # custom path, open the PDF when done
h4lat-catalogue --database my_operators   # catalogue of a custom operator database
```

The operator IDs are those of the database, so a partial catalogue skips some of them; the
sections always come in database order.  The default file name and title state the
selection.  The command prints the selection and progress bars over the operators (`-q` to
silence them).

The command is installed with the package.  It needs `pylatex` (`pip install "h4lat[full]"`)
and a LaTeX installation that provides `latexmk` or `pdflatex` (e.g. TeX Live).  See
`h4lat-catalogue --help` for all options.  From Python, call `operator_show`, which takes the
same selection and returns the path of the PDF:

```python
from h4lat import operator_show

operator_show(show=False)                                    # S, P, V, A, T with 0, 1, 2 derivatives
operator_show(X="V", nder=[1, 2, 3], show=False)             # vector operators with 1 to 3 derivatives
operator_show(keys=["V2", "V3", "A3"], n_jobs=None, show=False)  # n_jobs=None: all cores
```

In a script, call it with `n_jobs > 1` under `if __name__ == "__main__":`, as usual with
multiprocessing.  If you have no LaTeX installation, the
[`operator_catalogue.ipynb`](examples/operator_catalogue.ipynb) notebook shows the most
relevant operators (V, A and T with one and two derivatives), and its `show_group` function
lists any other group.  With a development install, run `pip install -e .` again after
pulling this change so that the command is registered.

The kinematic factors are computed in exact arithmetic: the CG coefficients, computed in
floating point, are first converted to the exact numbers they approximate (fractions such as
⅓, or square roots of fractions such as 1/√2).  Until October 2026 they entered SymPy as
floats, and some kinematic factors that vanish (operators 176, 524, 527, 741 and 744 of the
current numbering) were printed as tiny numbers of order 1e-16 instead of 0.

### Operators used in the literature

`get_paper_operators` returns the H(4)-irreducible operators used in published works.
Select the paper by its INSPIRE texkey or by its DOI.  Each operator's `id` is its label:

```python
from h4lat import AVAILABLE_PAPERS, get_paper_operators

print(AVAILABLE_PAPERS)                                       # INSPIRE texkey -> DOI

ops = get_paper_operators("Rodekamp:2023wpe")                 # by INSPIRE texkey
ops = get_paper_operators(doi="10.1103/PhysRevD.109.074508")  # by DOI: the same operators

for op in ops:
    print(f"id={op.id}  X={op.X}  irrep={op.irrep}  C={op.C}")
    op.display()   # renders LaTeX in a Jupyter notebook
```

| INSPIRE texkey | Reference | Operators |
|----------------|-----------|-----------|
| `Burger:2021knd` | S. Bürger et al., [Phys. Rev. D 105, 054504 (2022)](https://doi.org/10.1103/PhysRevD.105.054504), [arXiv:2111.08306](https://arxiv.org/abs/2111.08306) | a₂: axial, irrep (4,3), 2 derivatives; d₂: axial, irrep (8,1), 2 derivatives, and tensor, irrep (8,1), 1 derivative |
| `Rodekamp:2023wpe` | M. Rodekamp et al., [Phys. Rev. D 109, 074508 (2024)](https://doi.org/10.1103/PhysRevD.109.074508), [arXiv:2401.05360](https://arxiv.org/abs/2401.05360) | 1 derivative: vector, irreps (3,1) and (6,3); axial, irrep (6,4); tensor, irreps (8,1) and (8,2) |
| `Taggi:2026skn` | E. Taggi et al., [Phys. Rev. D 114, 074507 (2026)](https://doi.org/10.1103/kknt-zxsw), [arXiv:2605.02808](https://arxiv.org/abs/2605.02808) | 2 derivatives: vector, irrep (4,2); axial, irrep (4,3); tensor, irreps (3,2), (3,3) and (6,2) |
| `Alexandrou:2026tjs` | C. Alexandrou et al., [arXiv:2607.03458](https://doi.org/10.48550/arXiv.2607.03458) (2026) | vector: 2 derivatives, irrep (4,2); 3 derivatives, irreps (1,2) and (6,2) |
| `Gockeler:2004wp` | M. Göckeler et al., [Phys. Rev. D 71, 114511 (2005)](https://doi.org/10.1103/PhysRevD.71.114511), [arXiv:hep-ph/0410187](https://arxiv.org/abs/hep-ph/0410187) | vector: 1 derivative, irreps (6,3) and (3,1); 2 derivatives, irrep (8,1); 3 derivatives, irrep (2,1) |

Each operator is a linear combination of operators of the bundled database
(`Operator_from_database(id)`), built with the operator arithmetic of
[`operator_algebra.ipynb`](examples/operator_algebra.ipynb).  The notebook
[`paper_operators.ipynb`](examples/paper_operators.ipynb) shows the code that builds the
operators of each paper.  Combining the catalogue operators differently gives the operator
basis of other works.

### Building an operator database

```python
from h4lat import make_operator_database

make_operator_database(
    operator_folder="my_operators",
    max_nder=3,     # S, P, V, A and T with 0 to 3 derivatives (as the bundled database)
    verbose=True,
)
```

The operators are numbered as in the bundled database (by number of derivatives, then
S, P, V, A, T).  The folder must not already hold operator files.  Products beyond the
bundled ones (four or more derivatives) are computed by `cg_calc`, which for four
derivatives takes a lot of memory.

### Saving and reloading operators (HDF5)

`operator_list_to_h5` / `h5_to_operator_list` save an arbitrary `list[Operator]`
to a single portable `.h5` file and reload it later, preserving every attribute
needed to reconstruct each operator exactly — including `gamma_pol_index`, so a
custom polarisation matrix set via `set_polarization_matrix()` survives the
round trip.  This is a lighter-weight alternative to `make_operator_database`
for saving an ad-hoc selection of operators (e.g. the output of
`get_paper_operators`, or operators built by hand) rather than a full database:

```python
from h4lat import get_paper_operators, operator_list_to_h5, h5_to_operator_list

ops = get_paper_operators("Rodekamp:2023wpe")  # any list of Operator objects
operator_list_to_h5(ops, "my_operators/selection.h5")

# ... later, possibly in a different session ...
reloaded = h5_to_operator_list("my_operators/selection.h5")
assert [op.id for op in reloaded] == [op.id for op in ops]
```

Requires `h5py` (`pip install "h4lat[full]"` or `"h4lat[operators]"`). Under the
hood, each operator is stored in its own group and (de)serialised with
`write_operator` / `read_operator`, which can be used directly if you need to
embed a single operator inside a larger, custom HDF5 layout of your own.

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
  Note: for the irrep (2,2), h4lat now uses the generator α = diag(1, −1), which differs
  from the one given in this reference (the identity matrix, used by h4lat until October 2026).
  We believe the reference is wrong here: with α = 𝟙 the generators do not satisfy the S(4)
  relation (αβ)³ = 1, so they do not define a representation of H(4).
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

## How to cite

If you use `h4lat` in your research, please cite the repository:

```bibtex
@misc{Taggi_h4lat,
  author  = {Taggi, Emilio},
  title   = {{h4lat}: {C}lebsch-{G}ordan coefficients and lattice operators for the hypercubic group {H(4)}},
  year    = {2026},
  url     = {https://github.com/Tag-E/h4lat}
}
```

---

## License

GNU General Public License v3 — see [LICENSE](LICENSE).
All derivative works must be distributed under the same open-source terms.
