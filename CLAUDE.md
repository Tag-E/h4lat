# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Run these inside the project's virtual environment (e.g. `.venv/`). Python ≥ 3.11.

```bash
pip install -e ".[full]" pytest     # development install ("dev" extra = pytest + full)
pip install -e .                    # re-run after changing [project.scripts] entry points

python -m pytest                                          # whole suite (tests/), ~4 min
python -m pytest -k "not symbolic_sum"                    # skips the slowest test (~2 min of the 4)
python -m pytest tests/test_basic.py::test_known_kinematic_factors
python -m pytest -k "paper_operators"
python -m pytest tests/test_cg_correctness.py             # after any change to the generators, CG algorithm or bundled data

pre-commit run --all-files     # black + ruff --fix, configured in pyproject.toml (line length 119)
```

pre-commit is not part of any extra: install it with `pip install pre-commit`. Black keeps single quotes
(`skip-string-normalization`). Ruff ignores E741 (names like `l` and `O` are standard physics notation) and
E265/E266 (the `## … ##` banner comments are intentional).

Other entry points:

```bash
h4lat-catalogue [-X V A] [--nder 1 2] [--keys V2 A3] [-j [N]] [-o out.pdf]   # PDF catalogue; needs pylatex and latexmk/pdflatex
python scripts/regenerate_data.py {h4,cg,operators,all} [--jobs N] [--from-raw] [--products "(6,1),(4,1)"] [--max-nder 3]
```

## What the package does

h4lat computes Clebsch-Gordan (CG) coefficients of the hypercubic group H(4) (order 384, 20 irreps labelled
`(dim, l)`) and builds H(4)-irreducible lattice operators ψ̄ Γ D⋯D ψ (Γ = S 𝟙, P γ₅, V γ_μ, A γ_μγ₅, T σ_μν)
for moments of PDFs, with their kinematic factor K, C-parity, trace condition and index symmetry. The README
has the physics background and the user-facing API.

## Architecture

### Three-stage bundled data pipeline (`src/h4lat/data/`, tracked in git, ~9400 `.npy` files)

Each stage is derived from the previous one. When a stage's input changes, regenerate that stage and every stage
after it:

1. **`h4_ele/(k, l)/{0..383}.npy`**: matrices of all 384 group elements in each irrep. They come from the generator
   lists `alpha_list` / `beta_list` / `gamma_list` in `cg_calculator.py` (S(4) coset enumeration, then the Z₂⁴
   reflections). The lists follow the order of `rep_label_list`.
2. **`cg_database/{product}/{irep}_{m}.npy`**: one CG block per copy `m` of irrep `irep`. Here `irep` is the
   0-based index into `rep_label_list`, and the folder name is the concatenation of `str(tuple)`, e.g.
   `(6, 1)(4, 1)(4, 1)`. `{product}_raw/` holds the symbolic sympy blocks (pickled object arrays). Raw blocks are
   bundled only for products of ≤ 3 irreps (`MAX_RAW_IRREPS`), because the larger ones take tens of MB.
3. **`operator_database/operator_{id}_{n}_{X}_{k}_{l}_{block}_{index_block}.npy`**: one file per operator, holding
   its `cgmat`. All metadata is in the **file name**: loaders parse it, and selection by X/nder never opens the
   file. `n` is the number of Lorentz indices, **not** the number of derivatives
   (`n = nder + {S,P: 0, V,A: 1, T: 2}`).

`cg_calc` **never writes into the package data**. New results go to `./cg_database` and `./h4_ele` in the
working directory, or to `cgdatabase=`. Bundled data change only through `scripts/regenerate_data.py`, which
computes into `regenerated_data/` (gitignored), copies over only the files whose content changed (beyond 1e-10
noise), and reports operator-ID shifts. Then review with `git status src/h4lat/data`.

### `cg_calculator.py`: CG coefficients

- `cg_calc(*irreps)` loads a product from the bundled DB, then from `./cg_database`. Otherwise it computes the
  product with Sakata's projection formula Σ_h T(h)·A·D(h)ᵀ, where A is a matrix of free sympy symbols.
- The sum over the group is done **numerically** on the coefficients of the A symbols (`_projection`, optionally a
  numba `prange` kernel). The result is bit-identical to the old symbolic loop and does not depend on the number
  of threads. Tests check this, so keep the summation order over `h` when touching it.
- `_raw_block` turns the coefficients back into symbolic blocks. `CGmat_from_block` then picks `multiplicity`
  independent copies deterministically, from reference rows of column 0 and a minimal-norm A via `pinv`. With
  `enforce_c_parity`, each copy is even or odd under reversal of the derivative factors, **even copies first**.
  This is why every bundled operator has C = ±1. The docstring of `CGmat_from_block` explains the algorithm.

### `moments_operator.py`: operators (the largest module)

- `Operator(cgmat, id, X, irrep, block, index_block, gamma_pol_index=0)`. `cgmat` has shape `(4,)*n`, and shape
  `()` for ψ̄ψ / ψ̄γ₅ψ. For T, the first two indices are σ_μν, stored only at μ<ν: the antisymmetry is implicit
  (`cg_remapping_T`, `_SIGMA_PAIRS`). `C`, `tr` and `symm` are computed eagerly. `O`, `K`, `latex_O`, `latex_K`
  and `p3corr_is_real` are `functools.cached_property`, computed lazily, which keeps loading the whole DB at
  about 1 s. `set_polarization_matrix` pops the K-derived entries from `__dict__`.
- Operators support `+ - * /`, which gives the linear combinations used by `get_op_selection` /
  `get_paper_operators`. Adding operators of different irreps or blocks sets those fields to `None`.
- K is computed in **exact arithmetic**: `_exact_number` maps float CG coefficients to sympy rationals or ±√(p/q),
  and `Gamma_pol_list` uses `sym.Rational(1, 2)`. Don't let floats back into `diracO_from_cgmat` /
  `Kfactor_from_diracO`, or K values that should vanish become ~1e-16.
- **Operator IDs** run by nder (0..3), then X in `_X_ORDER = ('S','P','V','A','T')`, then irrep order, copy and
  column. The bundled DB has 1360 operators (ID ranges table in the README). `make_operator_database(folder,
  max_nder)` reproduces it.
- Selection by `X` / `nder` / `keys` (`"V2"` = V with 2 **derivatives**) is shared by the loaders
  (`_operator_filter`) and the PDF catalogue (`_catalogue_selection`, defaults S–T with nder 0–2). `operator_show`
  computes K in parallel with a `ProcessPoolExecutor` (`_kfactor_worker` must stay module-level so it can be
  pickled).
- HDF5 I/O (`write_operator` / `read_operator`) stores `None` fields through `*_is_none` attributes.

### `kinematic_data.py`

Euclidean gamma matrices, p_μ = (p₁, p₂, p₃, iE), the six polarisation projectors and the K denominators. The
float→LaTeX table (2.8 M entries) is built lazily by `latex_conversion_table()` so that `import h4lat` stays fast.
Don't build it at import time.

## Cross-file invariants (easy to break)

- `get_op_selection` and `get_paper_operators` **hard-code bundled-DB operator IDs**. Any change to the ordering or
  content of the operator database shifts them. Fix them using the shift report from `regenerate_data.py`.
- `examples/paper_operators.ipynb` shows the source of `get_paper_operators`, and
  `test_paper_operators_notebook_code` checks that every ```` ```python ```` block in its markdown cells is a
  line-by-line slice of the function (indentation ignored). Edit both together.
- The basis order of the 6-dim irreps in `cg_calculator.py` (pairs 12, 13, 23, 14, 24, 34) must match
  `_SIGMA_PAIRS` in `moments_operator.py`.
- New public names must be imported in `src/h4lat/__init__.py` **and** listed in `__all__`. Tests check this.
- `_X_ORDER` and `_operator_irreps` are imported by `tests/test_cg_correctness.py` and `scripts/regenerate_data.py`.
  They define which products are bundled.
- User-visible changes usually come with updates to the README, the CLI help (`cli.py`) and the affected example
  notebooks, which are then re-executed. The README records breaking changes as dated notes ("Until October
  2026 …").

## Conventions

- **Commit messages:** a short title line, an empty line, then a list of the updates, one `- ` bullet per item
  (see `git log`).
  Changes the maintainer flags as important start the title with `IMPORTANT UPDATE: `.
- **Docstrings and comments:** every module, class and function has a docstring, and inline comments explain what
  isn't obvious. Code should be readable by someone who knows neither the libraries (numpy, sympy, numba, pylatex,
  h5py) nor the group theory: say what a call does and why it's needed.
- Module files start with a `####` banner and the GPL-3 header. Docstrings are numpy-style and use Unicode math
  (ψ̄, γ_μ, ⊗), and comments explain the physics.
- `h4lat-catalogue` writes `operator_catalogue_*.pdf` to the working directory (gitignored). `*.ipynb` files are
  gitignored except `examples/*.ipynb`, and so are `*.txt` files in the repository root (local notes).
