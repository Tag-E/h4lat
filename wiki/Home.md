**h4lat** computes the Clebsch-Gordan coefficients of the hypercubic group $`H(4)`$ and builds $`H(4)`$-irreducible
lattice operators $`\bar\psi \, \Gamma \, D \cdots D \, \psi`$, with
$`\Gamma = \mathbf{1}, \gamma_5, \gamma_\mu, \gamma_\mu\gamma_5, \sigma_{\mu\nu}`$, for lattice QCD calculations of the
moments of parton distribution functions. For every operator it gives the kinematic factor, the C-parity, the trace
condition and the index symmetry.

This wiki is the documentation of h4lat: a user guide, the group theory behind the library, and the references.

## Where to start

| If you want to… | Read |
|-----------------|------|
| install h4lat | [Installation](https://github.com/Tag-E/h4lat/wiki/Installation) |
| understand the group theory and the method | [The hypercubic group](https://github.com/Tag-E/h4lat/wiki/The-hypercubic-group) → [Clebsch-Gordan coefficients](https://github.com/Tag-E/h4lat/wiki/Clebsch-Gordan-coefficients) → [Lattice operators](https://github.com/Tag-E/h4lat/wiki/Lattice-operators) |
| decompose tensor products and get Clebsch-Gordan coefficients | [Computing Clebsch-Gordan coefficients](https://github.com/Tag-E/h4lat/wiki/Computing-Clebsch-Gordan-coefficients) |
| use the 1360 bundled operators | [Operator database](https://github.com/Tag-E/h4lat/wiki/Operator-database) and [Building operators](https://github.com/Tag-E/h4lat/wiki/Building-operators) |
| browse the operators in a PDF | [Operator catalogue](https://github.com/Tag-E/h4lat/wiki/Operator-catalogue) |
| reproduce the operators of a published paper | [Operators from the literature](https://github.com/Tag-E/h4lat/wiki/Operators-from-the-literature) |
| save a selection of operators | [Saving and loading operators](https://github.com/Tag-E/h4lat/wiki/Saving-and-loading-operators) |
| find a function | [API overview](https://github.com/Tag-E/h4lat/wiki/API-overview) |

The Jupyter notebooks in the [`examples/`](https://github.com/Tag-E/h4lat/tree/main/examples) folder show the
library at work.

## A first example

```python
from h4lat import get_OperatorList, latex_print_multiplicities

# (4,1) ⊗ (4,1): the vector operators with one derivative, ψ̄ γ_μ D_ν ψ
print(latex_print_multiplicities((4, 1), (4, 1)))

# the same operators from the bundled database, with their irrep, C-parity and kinematic factor
for op in get_OperatorList(keys="V1"):
    print(op.id, op.irrep, op.C, op.K)
```

## Versions

The wiki describes the code on the `main` branch. Features listed under "Unreleased" in the
[changelog](https://github.com/Tag-E/h4lat/blob/main/CHANGELOG.md) are not yet in the release on
[PyPI](https://pypi.org/project/h4lat/). Operator IDs and conventions can change between versions: the changelog lists
every change, and the wiki notes the ones that affect results ("Until October 2026 …").

## Citing, questions and contributions

If you use h4lat in your research, please cite it as explained in the
[README](https://github.com/Tag-E/h4lat#how-to-cite), giving the version you used. Questions, bug reports and
suggestions are welcome as [GitHub issues](https://github.com/Tag-E/h4lat/issues), and
[CONTRIBUTING.md](https://github.com/Tag-E/h4lat/blob/main/CONTRIBUTING.md) explains how to contribute, also to this
wiki.
