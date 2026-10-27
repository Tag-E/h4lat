# Changelog

All notable changes to h4lat are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the version numbers follow
[Semantic Versioning](https://semver.org/) (while the version is 0.x, a minor release may change the API).
Changes marked **Breaking** can change the results or the code of existing users.

## [Unreleased]

### Changed

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

[Unreleased]: https://github.com/Tag-E/h4lat/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Tag-E/h4lat/releases/tag/v0.2.0
[0.1.0]: https://github.com/Tag-E/h4lat/commit/ddbd177
