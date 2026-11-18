# Changelog

All notable changes to h4lat are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the version numbers follow
[Semantic Versioning](https://semver.org/) (while the version is 0.x, a minor release may change the API).
Changes marked **Breaking** can change the results or the code of existing users.

## [Unreleased]

### Added

- The symmetry group W(F₄) of the 16-cell honeycomb lattice (order 1152, 25 irreps ϱ^(k)_l labelled as in Katz and
  Nogradi, arXiv:2611.21543), next to H(4). Every function that depends on the group takes the argument `group`:
  `None` or `'H4'` (the default, so existing code is unchanged) or `'honeycomb'`. This covers `cg_calc`,
  `get_multiplicities`, `latex_print_multiplicities`, `Operator`, `make_operator_database`, the database loaders
  (`get_OperatorList`, `get_OperatorDict`, `OperatorList_from_database`, `OperatorDict_from_database`,
  `Operator_from_database`), `operator_show`, `decomposition_analysis` and `irrep_count_table`.
- `get_group`, `LatticeGroup` and `AVAILABLE_GROUPS` (module `h4lat.groups`): the irrep labels, character table and
  data folders of each group. The module `h4lat.honeycomb` builds the 1152 elements (h4lat's H(4) and its two
  cosets under the triality rotation of the paper) and the matrices of the 25 irreps, without external programs.
- Bundled data of the honeycomb group: the matrices of its elements (`honeycomb_ele/`), the CG coefficients of the
  20 operator products (`cg_database_honeycomb/`, without raw blocks) and a database of 1360 operators sorted by its
  irreps (`operator_database_honeycomb/`), with the same ID ranges per Dirac structure and number of derivatives.
- `Operator.group` (`'H4'` or `'honeycomb'`). It is stored in the file names of the honeycomb operators (ending in
  `_honeycomb`) and in HDF5 files (attribute `symmetry_group`); files without it are read as H(4) operators.
- `h4lat-catalogue --group honeycomb` and `scripts/regenerate_data.py --group honeycomb`; the release workflow also
  attaches `operator_catalogue_honeycomb_SPVAT_nder0123.pdf`.
- `cg_calc(save_raw=…)`: whether to build and save the raw symbolic blocks (default: True for H(4), False for the
  honeycomb group).
- Wiki page "The 16-cell honeycomb group", and the notebook `examples/honeycomb.ipynb`.

### Changed

- `h4lat-catalogue -j 0` is now an error: the number of processes must be positive, as for
  `operator_show(n_jobs=…)`. It used to mean all cores, which was not documented; `-j` alone still uses all cores.
- `cg_calc` extracts the CG blocks directly from the numerical coefficients of the projection, without the symbolic
  raw blocks, with the same result to the last bit. Without the raw blocks (`save_raw=False`) the H(4) product
  (6,1) ⊗ (4,1)³ takes about 15 s on 12 cores (35 s serially) instead of about 80 s.
- `cg_calc(force_h4gen=…)` is now called `force_group_gen` (it regenerates the matrices of either group);
  `force_h4gen` still works, with a `DeprecationWarning`. `cg_calc.h4_mat` is now also `cg_calc.group_mat`: one
  read-only array of shape (number of elements, d, d) per irrep (`h4_mat[ir][e]` still is the matrix of element `e`).
- An irrep that the group does not have (in `cg_calc`, `get_multiplicities`, `Operator`) raises a `ValueError`
  that lists the irreps of the group, instead of a `KeyError`.

### Fixed

- `irrep_count_table(split=True)` no longer drops the last irrep when the number of irreps is odd (as for the 25
  irreps of the honeycomb group).

## [0.3.0] - 2026-11-02

### Added

- `get_paper_operators`: the operators of Dolgov et al. (LHPC and SESAM, 2002), `"LHPC:2002xzk"`, and of
  Göckeler et al. (1996), `"Gockeler:1995wg"`. The axial operators of Göckeler et al. (1996) use the polarisation
  projector `gamma_pol_index=1`, which polarises the nucleon in the 2-direction as in the paper.
- [Wiki](https://github.com/Tag-E/h4lat/wiki) with the documentation: a user guide, and the theory behind the
  library (the group H(4), the Clebsch-Gordan coefficients from Sakata's formula, the operators, their kinematic
  factor, C-parity and mixing). Its pages are in `wiki/` and are published by a GitHub Actions workflow.

### Changed

- The README is a short overview, with links to the wiki, which now holds the detailed usage and the notes on past
  changes of results ("Until October 2026 …"). PyPI lists the wiki as the documentation.
- `cg_calc` reads the H(4) matrices (`cg_calc.h4_mat`) only when it computes new CG coefficients, and then once
  per session and folder: loading a product from a database no longer reads 7680 files (0.15 s → under 1 ms).
  The arrays in `h4_mat` are shared between instances and read-only.

### Fixed

- `Operator_from_file` reads Windows paths whose folder names contain `_`, and also accepts a `Path`.
- `Operator.save` refuses operators without `id`, `irrep`, `block` or `index_block` (those built with `+` or `-`)
  with a `ValueError` pointing to `operator_list_to_h5`. It used to raise a `TypeError`, or to write a file that
  `Operator_from_file` could not read.
- `operator_show(show=True)` (the default) and `h4lat-catalogue --open` open the PDF on macOS and Windows too, not
  only with `xdg-open`. If no viewer can be started, a warning replaces the error, and the PDF is kept, even with
  `remove_pdf=True`.
- The docstring of `Kfactor_from_diracO` gave the denominator of K with an extra factor 1/(2E), and said that K = 1
  for the scalar operator. The computed K is unchanged: K = 1 for ψ̄γ₄ψ (and m_N/E for ψ̄ψ).

## [0.2.0] - 2026-10-23

First public release.

### Added

- Scalar and pseudoscalar operators (`X = "S"`, `"P"`, Γ = 𝟙, γ₅) and operators without derivatives,
  including those without Lorentz indices (ψ̄ψ, ψ̄γ₅ψ).
- Bundled operator database of 1360 operators ψ̄ Γ D⋯D ψ: every Dirac structure S, P, V, A, T with zero to
  three derivatives, together with the CG coefficients they are built from.
- `get_OperatorList` / `get_OperatorDict` load the bundled database without any path. They and the
  `*_from_database` loaders select operators by Dirac structure (`X`), number of derivatives (`nder`) or keys
  such as `"V2"`.
- `get_paper_operators` and `AVAILABLE_PAPERS`: the operators of Bürger et al. (2022), Rodekamp et al. (2024),
  Taggi et al. (2026), Alexandrou et al. (2026) and Göckeler et al. (2005), selected by INSPIRE texkey or DOI.
- `get_op_selection`: operators free of mixing for one to three derivatives.
- `h4lat-catalogue` command and `operator_show`: a PDF catalogue of the operators, with selection by `X`,
  `nder` or keys and kinematic factors computed on several processes.
- Selectable polarisation matrix for the kinematic factor (`Gamma_pol_list`,
  `Operator.set_polarization_matrix`).
- `operator_list_to_h5` / `h5_to_operator_list`: save any list of operators to one HDF5 file.
- `cg_calc(..., parallel=True, n_threads=...)`: sum over the group on several threads with numba (new
  `parallel` extra). The result does not depend on the number of threads.
- `irrep_count_table` in `decomposition_analysis`.
- `scripts/regenerate_data.py` to regenerate the bundled data, with a report of shifted operator IDs.
- Example notebooks (`quickstart`, `operator_algebra`, `operator_catalogue`, `paper_operators`) and a
  correctness test suite for the group, the CG coefficients and the operators.

### Changed

- **Breaking:** operator IDs are ordered by number of derivatives, then S, P, V, A, T. The README has the table
  that maps old IDs to new ones; the operators themselves are unchanged.
- **Breaking:** when an irrep appears more than once in a tensor product, its copies are chosen with a definite
  C-parity (even copies first), so every bundled operator has C = ±1. Operators in such irreps can refer to
  different copies than before.
- **Breaking:** `CGmat_from_block` picks the copies of an irrep by linear algebra on the coefficients of the free
  symbols (new signature), and `make_operator_database(folder, max_nder)` takes the maximum number of
  derivatives instead of `max_n` and refuses a folder that already holds operators.
- The CG coefficients are computed numerically on the coefficients of the free symbols: (6,1) ⊗ (4,1)³ takes
  about a minute instead of days, with results identical to the last bit.
- Kinematic factors are computed in exact arithmetic, so the ones that vanish are exactly 0 instead of ~1e-16.
- `O`, `K` and their LaTeX forms are computed on first access: loading the whole database takes about a second,
  and `import h4lat` about half a second.
- `cg_calc` never writes into the installed package: new results go to `./cg_database` and `./h4_ele`.
- Python ≥ 3.11 is required.
- The package metadata declare the licence as GPL-3.0-or-later, as the source files always did.

### Fixed

- H(4) generators: the 6-dimensional β and the (2,2) α = diag(1, −1) are corrected, so all 20 irreps are
  representations. For (2,2) this differs from Baake et al. (1982) (see the README). The bundled data were
  regenerated.
- The trace condition `Operator.tr`.
- `write_operator` / `read_operator`: operators with `id=None` can be written, and irreps are read back as plain
  integers.
- Tensor products with a one-dimensional factor (S and P operators) no longer crash, and CG files are loaded in
  copy order.

### Removed

- The `force_symmetry_*` helpers and the `enforce_symmetry` option of `cg_calc`: an exact projection already
  has the symmetries.

## [0.1.0] - 2026-05-06

Initial version: the CG calculator for H(4) (projection formula of Sakata), a bundled database of CG
coefficients, and the `Operator` class with kinematic factor, C-parity, trace condition, index symmetry and
LaTeX output.

[Unreleased]: https://github.com/Tag-E/h4lat/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Tag-E/h4lat/releases/tag/v0.3.0
[0.2.0]: https://github.com/Tag-E/h4lat/releases/tag/v0.2.0
[0.1.0]: https://github.com/Tag-E/h4lat/commit/ddbd177
