# Contributing to h4lat

Thank you for your interest in h4lat. Questions, bug reports and suggestions are welcome, also about the
physics and the group theory behind the package.

## Questions and bug reports

Open an [issue](https://github.com/Tag-E/h4lat/issues/new/choose). The forms ask for what is needed to
reproduce a problem:

- the h4lat version (`python -c "import h4lat; print(h4lat.__version__)"`), the Python version and the
  operating system;
- a short piece of code that shows the problem, with the output you get and the one you expected.

If you think a CG coefficient, an operator, a kinematic factor or a C-parity is wrong, please say how you checked
it (a reference, a hand calculation, another code). This is the most useful information for physics issues.

## Development setup

```bash
git clone https://github.com/Tag-E/h4lat.git
cd h4lat
python -m venv .venv && source .venv/bin/activate   # Python ≥ 3.11
pip install -e ".[dev]"                               # the package in editable mode, all extras and pytest
pip install pre-commit && pre-commit install          # format and lint at every commit
```

## Tests and code style

```bash
python -m pytest                              # whole suite, about 4 minutes
python -m pytest -k "not symbolic_sum"        # skips the slowest test (about 2 of the 4 minutes)
python -m pytest tests/test_cg_correctness.py # after any change to the generators, the CG algorithm or the data
pre-commit run --all-files                    # black + ruff, configured in pyproject.toml
```

The tests run on GitHub for every push and pull request (Python 3.11 to 3.14). Tests that need LaTeX are
skipped when `latexmk` / `pdflatex` are not installed.

The code is formatted with black (line length 119, single quotes kept) and checked with ruff. Every module,
class and function has a numpy-style docstring, and comments explain the physics and anything that is not
obvious. The code should be readable by someone who knows neither the libraries nor the group theory.

## Things that are easy to break

- **The bundled data** (`src/h4lat/data/`) are never edited by hand. They are regenerated with
  `python scripts/regenerate_data.py {h4,cg,operators,all}`, which computes into `regenerated_data/`, copies
  only the files whose content changed, and reports shifted operator IDs. When one stage changes, regenerate
  every stage after it (H(4) matrices → CG coefficients → operators).
- **Operator IDs** are hard-coded in `get_op_selection` and `get_paper_operators`. Any change to the content or
  the order of the operator database must update them (use the shift report of `regenerate_data.py`), and the
  README must note the change for existing users.
- **`examples/paper_operators.ipynb`** shows the source of `get_paper_operators`, and a test checks that they
  match. Edit both together, and keep the operator tables inside their `# fmt: off` / `# fmt: on` region.
- **New public names** must be imported in `src/h4lat/__init__.py` and listed in `__all__`.

`CLAUDE.md` describes the architecture and these invariants in more detail.

## Pull requests

For anything larger than a small fix, please open an issue first, so we can agree on the approach. A pull
request should pass the tests and pre-commit, and update the README, the CLI help, the example notebooks
and `CHANGELOG.md` (section "Unreleased") when it changes what users see.

Commit messages have a short title line, an empty line, then one `- ` bullet per change.

By contributing, you agree that your contribution is licensed under the
[GNU General Public License v3.0 or later](LICENSE), like the rest of h4lat.

## Making a release (maintainer)

1. Set the new `version` in `pyproject.toml`, `version` and `date-released` in `CITATION.cff`, and the
   version in the README (BibTeX entry and the `git+…@vX.Y.Z` install line).
2. In `CHANGELOG.md`, rename "Unreleased" to the new version with the date, add an empty "Unreleased"
   section above it, and update the links at the bottom.
3. Commit, push, and check that the tests pass on GitHub.
4. On GitHub, go to *Releases → Draft a new release*. Create the tag `vX.Y.Z` on `main`, use the title
   `h4lat X.Y.Z`, paste the changelog section as the notes, and publish.

Publishing the release runs `.github/workflows/publish.yml`. The workflow checks that the tag matches the
version, builds the package, uploads it to [PyPI](https://pypi.org/project/h4lat/) and attaches it to the
release. Zenodo archives the release and gives it a DOI.
