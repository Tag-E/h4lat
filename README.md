# h4lat

[![tests](https://github.com/Tag-E/h4lat/actions/workflows/tests.yml/badge.svg)](https://github.com/Tag-E/h4lat/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/h4lat?label=PyPI)](https://pypi.org/project/h4lat/)
[![Python versions](https://img.shields.io/pypi/pyversions/h4lat?label=Python)](https://pypi.org/project/h4lat/)
[![Documentation](https://img.shields.io/badge/docs-wiki-blue)](https://github.com/Tag-E/h4lat/wiki)
[![License: GPL v3+](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](https://github.com/Tag-E/h4lat/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23598681.svg)](https://doi.org/10.5281/zenodo.23598681)

**Clebsch-Gordan coefficients and lattice operators for the hypercubic group H(4)**

Lattice QCD calculations of nucleon structure (moments of parton distribution functions, form factors) need
operators that transform irreducibly under the hypercubic group **H(4)**, the symmetry group of a four-dimensional
hypercubic lattice, of order 384 with 20 irreducible representations (irreps). h4lat computes the Clebsch-Gordan
coefficients of H(4) and builds these operators:

* **Clebsch-Gordan coefficients** of any tensor product of H(4) irreps, with the projection formula of
  [Sakata (1974)](https://doi.org/10.1063/1.1666528) (optionally on all CPU cores), and a bundled database of the
  products needed for the operators below.
* A bundled database of **1360 H(4)-irreducible operators** ψ̄ Γ D⋯D ψ, with Γ = 𝟙, γ₅, γ_μ, γ_μγ₅, σ_μν (S, P, V,
  A, T) and zero to three derivatives, each with a definite C-parity.
* For every operator: its expression, its tree-level **kinematic factor** (in exact arithmetic, with a choice of six
  polarisation projectors), its **C-parity**, trace condition and index symmetry.
* The operators used in **published works**, a **PDF catalogue** of the operators, linear combinations of operators
  and HDF5 storage.

---

## Documentation

The documentation is in the **[wiki](https://github.com/Tag-E/h4lat/wiki)**:

| | |
|---|---|
| **Getting started** | [Installation](https://github.com/Tag-E/h4lat/wiki/Installation) (optional features, install from GitHub, development install) |
| **User guide** | [Computing Clebsch-Gordan coefficients](https://github.com/Tag-E/h4lat/wiki/Computing-Clebsch-Gordan-coefficients) · [Building operators](https://github.com/Tag-E/h4lat/wiki/Building-operators) · [Operator database](https://github.com/Tag-E/h4lat/wiki/Operator-database) (operator IDs) · [Operator catalogue (PDF)](https://github.com/Tag-E/h4lat/wiki/Operator-catalogue) · [Operators from the literature](https://github.com/Tag-E/h4lat/wiki/Operators-from-the-literature) · [Saving and loading operators](https://github.com/Tag-E/h4lat/wiki/Saving-and-loading-operators) · [API overview](https://github.com/Tag-E/h4lat/wiki/API-overview) |
| **Theory** | [The hypercubic group](https://github.com/Tag-E/h4lat/wiki/The-hypercubic-group) · [Clebsch-Gordan coefficients](https://github.com/Tag-E/h4lat/wiki/Clebsch-Gordan-coefficients) · [Lattice operators](https://github.com/Tag-E/h4lat/wiki/Lattice-operators) · [References](https://github.com/Tag-E/h4lat/wiki/References) |

Every function and class also has a detailed docstring (`help(h4lat.cg_calc)`).

---

## Installation

```bash
pip install h4lat                   # core (numpy + sympy + tqdm only)
pip install "h4lat[full]"           # all the optional features (PDF catalogue, parallel computation, HDF5, …)
```

h4lat needs Python ≥ 3.11. To install a given release, or the latest code, directly from GitHub:

```bash
pip install "h4lat @ git+https://github.com/Tag-E/h4lat.git@v0.3.0"   # release 0.3.0
pip install "h4lat[full] @ git+https://github.com/Tag-E/h4lat.git"    # latest code on main
```

The [Installation](https://github.com/Tag-E/h4lat/wiki/Installation) page of the wiki describes the other optional
features and the development install.

---

## Quick start

```python
from h4lat import cg_calc, get_OperatorList, get_paper_operators, latex_print_multiplicities

# Decompose (4,1) ⊗ (4,1), the product of the vector operators with one derivative ψ̄ γ_μ D_ν ψ
print(latex_print_multiplicities((4, 1), (4, 1)))    # τ⁽¹⁾₁ ⊕ τ⁽³⁾₁ ⊕ τ⁽⁶⁾₁ ⊕ τ⁽⁶⁾₃, as a LaTeX string

# Its Clebsch-Gordan coefficients, loaded from the bundled database: one block per irrep and copy
cg = cg_calc((4, 1), (4, 1), verbose=False)

# Operators of the bundled database, selected by Dirac structure and number of derivatives ("V1")
ops = get_OperatorList(keys="V1")
op = ops[1]
print(op.id, op.irrep, op.C, op.tr, op.symm)       # ID, H(4) irrep, C-parity, trace condition, index symmetry
print(op.latex_O)                                   # the operator, as a LaTeX string
print(op.K)                                         # kinematic factor, a SymPy expression in m_N, E(p), p_1, p_2, p_3
print(op.evaluate_K(m_value=0.939, E_value=1.0, p1_value=0.0, p2_value=0.0, p3_value=0.3))

# The operators of a published work, selected by INSPIRE texkey or DOI
paper_ops = get_paper_operators("Rodekamp:2023wpe")
```

```bash
h4lat-catalogue -X V --nder 1 2      # PDF catalogue of the vector operators with 1 and 2 derivatives (needs LaTeX)
```

---

## Examples

Jupyter notebooks with worked examples are provided in the [`examples/`](https://github.com/Tag-E/h4lat/tree/main/examples) folder:

| Notebook | Description |
|----------|-------------|
| [`quickstart.ipynb`](https://github.com/Tag-E/h4lat/blob/main/examples/quickstart.ipynb) | Tensor-product decomposition, CG coefficients, operator construction, and the bundled database, in executable form |
| [`operator_algebra.ipynb`](https://github.com/Tag-E/h4lat/blob/main/examples/operator_algebra.ipynb) | Linear combinations of operators: addition, subtraction, scalar multiplication, and mixed-irrep behaviour |
| [`operator_catalogue.ipynb`](https://github.com/Tag-E/h4lat/blob/main/examples/operator_catalogue.ipynb) | Listing of the most relevant operators of the bundled database — V, A and T with one and two derivatives (IDs 25–80 and 113–336) — organised by number of derivatives, Dirac structure, and H(4) irrep; all the operators are in the [PDF catalogue](https://github.com/Tag-E/h4lat/wiki/Operator-catalogue) |
| [`paper_operators.ipynb`](https://github.com/Tag-E/h4lat/blob/main/examples/paper_operators.ipynb) | Operators used in published works, selected with `get_paper_operators` by INSPIRE texkey or DOI, and the code that builds them from the operators of the bundled database |

A Google Colab notebook with some minimal working examples can be found [here](https://colab.research.google.com/drive/1KFnE0hi4XDNyo2Kg8iDKSSEqY1YQkPGZ?usp=sharing).

---

## How to cite

If you use `h4lat` in your research, please cite it, and give the version you used (`h4lat.__version__`),
since operator IDs and conventions can change between versions (see the
[changelog](https://github.com/Tag-E/h4lat/blob/main/CHANGELOG.md)):

```bibtex
@software{Taggi_h4lat,
  author  = {Taggi, Emilio},
  title   = {{h4lat}: {C}lebsch-{G}ordan coefficients and lattice operators for the hypercubic group {H(4)}},
  year    = {2026},
  version = {0.3.0},
  doi     = {10.5281/zenodo.23598681},
  url     = {https://github.com/Tag-E/h4lat}
}
```

The DOI [10.5281/zenodo.23598681](https://doi.org/10.5281/zenodo.23598681) covers all versions of h4lat and
leads to the latest one.  Each release also has its own DOI, listed on the
[Zenodo page](https://doi.org/10.5281/zenodo.23598681).  The "Cite this repository" button on the GitHub page
gives the same reference in APA and BibTeX form (from
[`CITATION.cff`](https://github.com/Tag-E/h4lat/blob/main/CITATION.cff)).

---

## Contributing and changelog

Questions, bug reports and suggestions are welcome as
[GitHub issues](https://github.com/Tag-E/h4lat/issues).
[CONTRIBUTING.md](https://github.com/Tag-E/h4lat/blob/main/CONTRIBUTING.md) explains how to set up a
development install, run the tests and edit the wiki, and
[CHANGELOG.md](https://github.com/Tag-E/h4lat/blob/main/CHANGELOG.md) lists the changes of each release.

---

## Acknowledgements

This library was developed with the assistance of [Claude](https://www.anthropic.com/claude)
(Anthropic), an AI assistant, which helped with code generation, testing, and documentation.

---

## License

GNU General Public License v3.0 or later: see
[LICENSE](https://github.com/Tag-E/h4lat/blob/main/LICENSE).
All derivative works must be distributed under the same open-source terms.
