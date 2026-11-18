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
python -m pytest tests/test_honeycomb.py                  # the honeycomb group against the tables of arXiv:2611.21543
python -m pytest tests/test_wiki.py                       # after any change to wiki/, the README or a public name

pre-commit run --all-files     # black + ruff --fix, configured in pyproject.toml (line length 119)
```

pre-commit is not part of any extra: install it with `pip install pre-commit`. Black keeps single quotes
(`skip-string-normalization`). Ruff ignores E741 (names like `l` and `O` are standard physics notation) and
E265/E266 (the `## … ##` banner comments are intentional).

Other entry points:

```bash
h4lat-catalogue [--group honeycomb] [-X V A] [--nder 1 2] [--keys V2 A3] [-j [N]] [-o out.pdf]   # needs pylatex, LaTeX
python scripts/regenerate_data.py {elements,cg,operators,all} [--group honeycomb] [--jobs N] [--from-raw] \
    [--products "(6,1),(4,1)"] [--max-nder 3]          # stage `h4` = older name of `elements`
```

## What the package does

h4lat computes Clebsch-Gordan (CG) coefficients of the hypercubic group H(4) (order 384, 20 irreps labelled
`(dim, l)`) and of the 16-cell honeycomb group W(F₄) (order 1152, 25 irreps ϱ^(k)_l, labels of Katz and Nogradi,
arXiv:2611.21543; `group='honeycomb'`), and builds irreducible lattice operators ψ̄ Γ D⋯D ψ (Γ = S 𝟙, P γ₅, V γ_μ, A γ_μγ₅, T σ_μν)
for moments of PDFs, with their kinematic factor K, C-parity, trace condition and index symmetry. The wiki
(source in `wiki/`, see below) has the user guide and the theory (group theory, Sakata's formula, K, C-parity,
mixing); the README is a short overview that links to it.

## Architecture

### Groups (`groups.py`, `honeycomb.py`)

- `LatticeGroup` (frozen dataclass, compared by identity, pickled by name) holds the static data of a group: name
  (`'H4'` | `'honeycomb'`), order, `rep_label_list`, read-only `char_table`, `class_orders`, LaTeX symbol (`\tau` |
  `\varrho`), data-folder names, `ele_layout`, `operator_file_suffix` (`''` | `'_honeycomb'`). `get_group(group)`
  resolves `None`/aliases; every group-dependent function takes `group=None` (→ H(4), backward compatible).
- The H(4) tables live in `groups.py`; `cg_calculator` re-exports them as the historical module-level names
  (`rep_label_list`, `char_table`, …: lists/dicts, H(4) only). Never hard-code H(4) tables elsewhere: use the group.
- The honeycomb character table (from GAP, rows matched to the paper's definitions eq. 5.2) is hard-coded in
  `groups.py`; its **columns are classes in order of first appearance in the element list**.
- `honeycomb.py` (pure functions, no I/O) builds the 1152 elements as **H(4) in h4lat's order, then H(4)·A, H(4)·A²**
  (A = triality, paper eq. 3.2), the classes, and the 25 irreps (R, det R, Λ²R explicit; the others from the
  `_RECIPE` table by projection + Gram-Schmidt; entries snapped to exact rational/±√rational).

### Three-stage bundled data pipeline (`src/h4lat/data/`, tracked in git, ~12 000 `.npy` files)

Each stage is derived from the previous one, for each group. When a stage's input changes, regenerate that stage and
every stage after it (the honeycomb data derive from the H(4) generators too):

1. **`h4_ele/(k, l)/{0..383}.npy`**: matrices of all 384 group elements in each irrep. They come from the generator
   lists `alpha_list` / `beta_list` / `gamma_list` in `cg_calculator.py` (S(4) coset enumeration, then the Z₂⁴
   reflections). The lists follow the order of `rep_label_list`.
   For the honeycomb group: **`honeycomb_ele/(k, l).npy`**, one (1152, k, k) array per irrep (`ele_layout='irrep'`).
2. **`cg_database/{product}/{irep}_{m}.npy`** (honeycomb: `cg_database_honeycomb/`, never any `_raw`): one CG block
   per copy `m` of irrep `irep`. Here `irep` is the
   0-based index into `rep_label_list`, and the folder name is the concatenation of `str(tuple)`, e.g.
   `(6, 1)(4, 1)(4, 1)`. `{product}_raw/` holds the symbolic sympy blocks (pickled object arrays). Raw blocks are
   bundled only for products of ≤ 3 irreps (`MAX_RAW_IRREPS`), because the larger ones take tens of MB.
3. **`operator_database/operator_{id}_{n}_{X}_{k}_{l}_{block}_{index_block}.npy`**: one file per operator, holding
   its `cgmat` (honeycomb: `operator_database_honeycomb/…_{index_block}_honeycomb.npy`; no suffix = H(4)). All metadata is in the **file name**: loaders parse it, and selection by X/nder never opens the
   file. `n` is the number of Lorentz indices, **not** the number of derivatives
   (`n = nder + {S,P: 0, V,A: 1, T: 2}`).

`cg_calc` **never writes into the package data**. New results go to `./cg_database` and `./h4_ele` in the
working directory (`./cg_database_honeycomb`, `./honeycomb_ele` for the honeycomb group), or to `cgdatabase=`. Bundled data change only through `scripts/regenerate_data.py`, which
computes into `regenerated_data/` (gitignored), copies over only the files whose content changed (beyond 1e-10
noise), and reports operator-ID shifts. Then review with `git status src/h4lat/data`.

### `cg_calculator.py`: CG coefficients

- `cg_calc(*irreps, group=None)` loads a product from the bundled DB of the group, then from the local DB.
  Otherwise it computes the product with Sakata's projection formula Σ_h T(h)·A·D(h)ᵀ, where A is a matrix of free
  sympy symbols. Instance attributes (`rep_label_list`, `irrep_index`, …) are those of its group; the class
  attributes stay H(4). `force_group_gen` (deprecated alias `force_h4gen`), `group_mat` (alias `h4_mat`), loaded by
  the cached `_load_group_matrices(folder, group_name)`; bundled paths in `_BUNDLED_DATA[group][kind]`.
- The sum over the group is done **numerically** on the coefficients of the A symbols (`_projection`, optionally a
  numba `prange` kernel). The result is bit-identical to the old symbolic loop and does not depend on the number
  of threads. Tests check this, so keep the summation order over `h` when touching it.
- The copies are extracted **numerically**: `_linear_forms_from_coefficients` gives exactly the array
  `_linear_forms(_raw_block(...))` (same values, C-contiguous: the layout changes BLAS rounding), and
  `_cgmat_from_linear_forms` is the numerical part of `CGmat_from_block`. Bit-identical to the symbolic route (tested).
  `_raw_block` (symbolic) is only built to be saved: `save_raw=None` → True for H(4), False for honeycomb.
  `CGmat_from_block` picks `multiplicity` independent copies deterministically, from reference rows of column 0 and a minimal-norm A via `pinv`. With
  `enforce_c_parity`, each copy is even or odd under reversal of the derivative factors, **even copies first**.
  This is why every bundled operator has C = ±1. The docstring of `CGmat_from_block` explains the algorithm.

### `moments_operator.py`: operators (the largest module)

- `Operator(cgmat, id, X, irrep, block, index_block, gamma_pol_index=0, group=None)`; `op.group` is the group name
  (`'H4'` | `'honeycomb'`), the irrep is validated against it, `+`/`-` refuse mixing groups, HDF5 stores it as
  `symmetry_group` (absent → H4), file names as the suffix. `cgmat` has shape `(4,)*n`, and shape
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
  column. The bundled DB has 1360 operators (ID ranges table in `wiki/Operator-database.md`).
  `make_operator_database(folder, max_nder)` reproduces it.
- Selection by `X` / `nder` / `keys` (`"V2"` = V with 2 **derivatives**) is shared by the loaders
  (`_operator_filter`) and the PDF catalogue (`_catalogue_selection`, defaults S–T with nder 0–2). `operator_show`
  computes K in parallel with a `ProcessPoolExecutor` (`_kfactor_worker` must stay module-level so it can be
  pickled).
- HDF5 I/O (`write_operator` / `read_operator`) stores `None` fields through `*_is_none` attributes.

### `kinematic_data.py`

Euclidean gamma matrices, p_μ = (p₁, p₂, p₃, iE), the six polarisation projectors and the K denominators. The
float→LaTeX table (2.8 M entries) is built lazily by `latex_conversion_table()` so that `import h4lat` stays fast.
Don't build it at import time.

### `wiki/`: the documentation

The pages of the GitHub wiki (https://github.com/Tag-E/h4lat/wiki) are the Markdown files of `wiki/`: the file name is
the page name, and `_Sidebar.md` / `_Footer.md` are shown on every page. `.github/workflows/wiki.yml` copies the
folder to the wiki repository (`Tag-E/h4lat.wiki`) at every push to `main` that changes it, so web edits of the wiki
are overwritten. Theory pages: `The-hypercubic-group`, `Clebsch-Gordan-coefficients`, `Lattice-operators`,
`The-16-cell-honeycomb-group`, `References`; the other pages are the user guide (the former README sections). Conventions, checked by
`tests/test_wiki.py`:
- links between pages are full URLs `https://github.com/Tag-E/h4lat/wiki/Page-Name#section` (GitHub documents only
  those for Markdown wikis), and every page is linked from `_Sidebar.md`;
- inline math is `` $`{…}`$ ``, braced: GitHub's MathJax 4 otherwise lets the browser break it at top-level
  operators (=, +, ⊗, `\,`). Display math is a ```` ```math ```` block outside lists and quotes. No `\operatorname`
  (GitHub refuses it and shows "The following macros are not allowed": use `\mathrm{Tr}`; `GITHUB_BLOCKED_MACROS` in
  the test lists the others), no `\slashed`, no footnotes, no formulas inside bold text;
- every `h4lat.…` name and every `from h4lat… import …` in the pages and the README must exist;
- no version numbers in the wiki (use `vX.Y.Z`), so that a release does not have to touch it.

Every number in the theory pages (generators, γ⁽ʲ⁾ = β^{j−1} γ β^{−(j−1)}, the (4,1)⊗(4,1) blocks and their raw
factors 1, √2, √6, K = 1 for γ₄) was checked against the code: re-check the pages when the generators, the CG
normalisation or K change. The tables of `The-16-cell-honeycomb-group` (definitions of the irreps, restriction to
H(4), rest-frame channels) are those checked by `tests/test_honeycomb.py`.

## Cross-file invariants (easy to break)

- The honeycomb data rely on: elements 0–383 = h4lat's H(4) in h4lat's order; (4,1) = Cartesian matrices, (4,4) =
  det·(4,1), (6,1) = Λ² in the `_SIGMA_PAIRS` order (so `_DIRAC_IRREP`, `cg_remapping*`, K and C-parity are shared
  by both groups); the honeycomb char-table column order. `tests/data/honeycomb_reference.json` holds the paper's
  tables used by `tests/test_honeycomb.py`.
- `get_op_selection` and `get_paper_operators` **hard-code bundled-DB operator IDs**: those of the H(4) database,
  except the paper `"Katz:2026qmx"`, which reads the honeycomb one with `Operator_from_database(id, group="honeycomb")`
  (keep this exact spelling: `regenerate_data.py` finds the IDs with a regex). Any change to the ordering or content
  of an operator database shifts them. Fix them using the shift report from `regenerate_data.py --group …`.
- `examples/paper_operators.ipynb` shows the source of `get_paper_operators`, and
  `test_paper_operators_notebook_code` checks that every ```` ```python ```` block in its markdown cells is a
  line-by-line slice of the function (indentation ignored). Edit both together.
- The operator definitions of `get_op_selection` and `get_paper_operators` are hand-aligned tables inside
  `# fmt: off` / `# fmt: on`. Keep new entries inside that region: black would otherwise reformat them and break
  `test_paper_operators_notebook_code`.
- The basis order of the 6-dim irreps in `cg_calculator.py` (pairs 12, 13, 23, 14, 24, 34) must match
  `_SIGMA_PAIRS` in `moments_operator.py`.
- New public names must be imported in `src/h4lat/__init__.py` **and** listed in `__all__`. Tests check this.
- `_X_ORDER` and `_operator_irreps` are imported by `tests/test_cg_correctness.py` and `scripts/regenerate_data.py`.
  They define which products are bundled.
- User-visible changes usually come with updates to the wiki pages in `wiki/`, the README (overview, quick start),
  the CLI help (`cli.py`) and the affected example notebooks, which are then re-executed. The wiki records breaking
  changes as dated notes on the page concerned ("Until October 2026 …"), and `CHANGELOG.md` lists them under
  "Unreleased".
- The version number appears in `pyproject.toml`, `CITATION.cff` (`version`, `date-released`), `CHANGELOG.md`
  and the README (BibTeX entry and the `git+…@vX.Y.Z` install line). Change them together.

## Conventions

- **Commit messages:** a short title line, an empty line, then a list of the updates, one `- ` bullet per item
  (see `git log`).
  Changes the maintainer flags as important start the title with `IMPORTANT UPDATE: `.
- **Docstrings and comments:** every module, class and function has a docstring, and inline comments explain what
  isn't obvious. Code should be readable by someone who knows neither the libraries (numpy, sympy, numba, pylatex,
  h5py) nor the group theory: say what a call does and why it's needed.
- Module files start with a `####` banner and the GPL-3 header. Docstrings are numpy-style and use Unicode math
  (ψ̄, γ_μ, ⊗), and comments explain the physics.
- **Releases:** bump the version (see above), turn "Unreleased" in `CHANGELOG.md` into the new version, then
  publish a GitHub release with tag `vX.Y.Z`. `.github/workflows/publish.yml` checks that the tag matches
  `pyproject.toml` and uploads to PyPI (Trusted Publishing), and Zenodo archives the release.
  `.github/workflows/catalogue.yml` (on release, or by hand with a tag) installs LaTeX on the runner, runs
  `h4lat-catalogue --nder 0 1 2 3` and attaches `operator_catalogue_SPVAT_nder0123.pdf` to the release; the wiki page
  `Operator-catalogue` links to that file on the latest release, so keep its name.
  `.github/workflows/tests.yml` runs pre-commit and pytest (Python 3.11–3.14) on every push and PR.
  `.github/dependabot.yml` opens monthly grouped PRs for the GitHub Actions and the pre-commit hook revs, but not
  for the Python dependencies, which are unpinned (handling them: CONTRIBUTING.md, "Dependency updates").
- `h4lat-catalogue` writes `operator_catalogue_*.pdf` to the working directory (gitignored). `*.ipynb` files are
  gitignored except `examples/*.ipynb`, and so are `*.txt` files in the repository root (local notes).
