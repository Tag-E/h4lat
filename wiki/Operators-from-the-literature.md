`get_paper_operators` returns the $`{H(4)}`$-irreducible operators used in published works. Select the paper by its
INSPIRE texkey or by its DOI. Each operator's `id` is its label in the paper:

```python
from h4lat import AVAILABLE_PAPERS, get_paper_operators

print(AVAILABLE_PAPERS)                                       # INSPIRE texkey -> DOI

ops = get_paper_operators("Rodekamp:2023wpe")                 # by INSPIRE texkey
ops = get_paper_operators(doi="10.1103/PhysRevD.109.074508")  # by DOI: the same operators

for op in ops:
    print(f"id={op.id}  X={op.X}  irrep={op.irrep}  C={op.C}")
    op.display()   # renders LaTeX in a Jupyter notebook
```

## Available papers

| INSPIRE texkey | Reference | Operators |
|----------------|-----------|-----------|
| `Burger:2021knd` | S. Bürger et al., [Phys. Rev. D 105, 054504 (2022)](https://doi.org/10.1103/PhysRevD.105.054504), [arXiv:2111.08306](https://arxiv.org/abs/2111.08306) | a₂: axial, irrep (4,3), 2 derivatives; d₂: axial, irrep (8,1), 2 derivatives, and tensor, irrep (8,1), 1 derivative |
| `Rodekamp:2023wpe` | M. Rodekamp et al., [Phys. Rev. D 109, 074508 (2024)](https://doi.org/10.1103/PhysRevD.109.074508), [arXiv:2401.05360](https://arxiv.org/abs/2401.05360) | 1 derivative: vector, irreps (3,1) and (6,3); axial, irrep (6,4); tensor, irreps (8,1) and (8,2) |
| `Taggi:2026skn` | E. Taggi et al., [Phys. Rev. D 114, 074507 (2026)](https://doi.org/10.1103/kknt-zxsw), [arXiv:2605.02808](https://arxiv.org/abs/2605.02808) | 2 derivatives: vector, irrep (4,2); axial, irrep (4,3); tensor, irreps (3,2), (3,3) and (6,2) |
| `Alexandrou:2026tjs` | C. Alexandrou et al., [arXiv:2607.03458](https://doi.org/10.48550/arXiv.2607.03458) (2026) | vector: 2 derivatives, irrep (4,2); 3 derivatives, irreps (1,2) and (6,2) |
| `Gockeler:2004wp` | M. Göckeler et al., [Phys. Rev. D 71, 114511 (2005)](https://doi.org/10.1103/PhysRevD.71.114511), [arXiv:hep-ph/0410187](https://arxiv.org/abs/hep-ph/0410187) | vector: 1 derivative, irreps (6,3) and (3,1); 2 derivatives, irrep (8,1); 3 derivatives, irrep (2,1) |
| `LHPC:2002xzk` | D. Dolgov et al. (LHPC and SESAM), [Phys. Rev. D 66, 034506 (2002)](https://doi.org/10.1103/PhysRevD.66.034506), [arXiv:hep-lat/0201021](https://arxiv.org/abs/hep-lat/0201021) | vector: 1 derivative, irreps (6,3) and (3,1); 2 derivatives, irrep (8,1); 3 derivatives, irrep (2,1); axial: no derivative, irrep (4,4); 1 derivative, irreps (6,4) and (6,1); 2 derivatives, irreps (4,3) and (8,1); tensor: no derivative, irrep (6,1); 1 derivative, irrep (8,1) |
| `Gockeler:1995wg` | M. Göckeler et al., [Phys. Rev. D 53, 2317 (1996)](https://doi.org/10.1103/PhysRevD.53.2317), [arXiv:hep-lat/9508004](https://arxiv.org/abs/hep-lat/9508004) | vector: 1 derivative, irreps (6,3) and (3,1); 2 derivatives, irrep (8,1); 3 derivatives, irrep (2,1); axial: no derivative, irrep (4,4); 2 derivatives, irreps (4,3) and (8,1) |

The test suite checks the labels, Dirac structures, irreps and C-parities of the operators of every paper. For the
papers of Göckeler et al. and Dolgov et al., it also compares the operators with the expressions given in the papers,
and their kinematic factors with the papers' kinematics.

## How the operators are built

Each operator is a linear combination of operators of the bundled database (`Operator_from_database(id)`), built with
the [operator arithmetic](https://github.com/Tag-E/h4lat/wiki/Building-operators#linear-combinations). The notebook
[`paper_operators.ipynb`](https://github.com/Tag-E/h4lat/blob/main/examples/paper_operators.ipynb) shows the code that
builds the operators of each paper. Combining the catalogue operators differently gives the operator basis of other
works: contributions of new papers are welcome (see
[CONTRIBUTING.md](https://github.com/Tag-E/h4lat/blob/main/CONTRIBUTING.md)).

For operators free of mixing under renormalisation, see also `get_op_selection` and the theory page
[Lattice operators](https://github.com/Tag-E/h4lat/wiki/Lattice-operators#mixing-under-renormalisation).
