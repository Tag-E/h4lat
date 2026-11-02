h4lat needs Python ≥ 3.11 and is published on [PyPI](https://pypi.org/project/h4lat/).

## From PyPI

```bash
pip install h4lat                   # core (numpy + sympy + tqdm only)
pip install "h4lat[full]"           # + pylatex, IPython, matplotlib, gvar, h5py, pandas, numba
```

The core installation is enough to load the bundled Clebsch-Gordan coefficients and operators and to compute new
ones. The optional dependencies ("extras") add the other features:

| Extra | Adds | Needed for |
|-------|------|------------|
| `full` | everything below, and matplotlib | all the features |
| `parallel` | numba | computing new Clebsch-Gordan coefficients on all CPU cores, `cg_calc(..., parallel=True)` |
| `notebook` | IPython | rendering operators and decompositions in Jupyter, e.g. `op.display()` |
| `operators` | pandas, tabulate, gvar, h5py, IPython | `decomposition_analysis`, `evaluate_K_gvar`, HDF5 files |
| `hdf5` | h5py, gvar | saving and loading operators in HDF5 files |

For example, `pip install "h4lat[parallel]"`. The PDF catalogue of the operators (`h4lat-catalogue`,
`operator_show`) also needs pylatex (in `full`) and a LaTeX installation that provides `latexmk` or `pdflatex`, e.g.
TeX Live.

## From GitHub

To install a given release, or the latest code, directly from GitHub:

```bash
pip install "h4lat @ git+https://github.com/Tag-E/h4lat.git@vX.Y.Z"   # release X.Y.Z (see the Releases page)
pip install "h4lat[full] @ git+https://github.com/Tag-E/h4lat.git"    # latest code on main
```

The releases are listed on the [Releases page](https://github.com/Tag-E/h4lat/releases), and their changes in the
[changelog](https://github.com/Tag-E/h4lat/blob/main/CHANGELOG.md).

## For development

```bash
git clone https://github.com/Tag-E/h4lat.git
cd h4lat
pip install -e ".[dev]"             # editable install with all extras and pytest
```

See [CONTRIBUTING.md](https://github.com/Tag-E/h4lat/blob/main/CONTRIBUTING.md) for the tests, the code style and the
rules for the bundled data.

## Checking the installation

```bash
python -c "import h4lat; print(h4lat.__version__)"
```

When reporting a problem or citing h4lat, please give this version: operator IDs and conventions can change between
versions.
