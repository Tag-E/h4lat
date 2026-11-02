To browse the [operator database](https://github.com/Tag-E/h4lat/wiki/Operator-database), write a PDF catalogue of
its operators. For each operator it lists the expression, the kinematic factor, the C-parity, the trace condition and
the index symmetry, grouped by Dirac structure, number of indices and $`{H(4)}`$ irrep.

## From the command line

Computing the kinematic factors takes most of the time, so by default the catalogue holds the S, P, V, A and T
operators with zero, one and two derivatives (IDs 1–336: about 30 s on a laptop, 20 s with `-j`; the whole database
takes about a minute with `-j`). Select other operators by Dirac structure (`-X`) and number of derivatives
(`--nder`), or with keys made of the Dirac structure followed by the number of derivatives (`--keys V2 V3 A3`: `V2` is
the vector operators with two derivatives, i.e. three Lorentz indices). `-j` computes the kinematic factors on several
processes (`-j` alone: all cores):

```bash
h4lat-catalogue                           # S, P, V, A, T with 0, 1, 2 derivatives -> ./operator_catalogue_SPVAT_nder012.pdf
h4lat-catalogue -X V --nder 1 2 3         # vector operators with 1 to 3 derivatives -> operator_catalogue_V_nder123.pdf
h4lat-catalogue --nder 0                  # the local bilinears ψ̄ Γ ψ -> operator_catalogue_SPVAT_nder0.pdf
h4lat-catalogue --keys V2 V3 A3 -j        # a selection, on all cores -> operator_catalogue_V2_V3_A3.pdf
h4lat-catalogue --nder 0 1 2 3 -j         # the whole bundled database
h4lat-catalogue -o docs/ops.pdf --open    # custom path, open the PDF when done
h4lat-catalogue --database my_operators   # catalogue of a custom operator database
```

The operator IDs are those of the database, so a partial catalogue skips some of them; the sections always come in
database order. The default file name and title state the selection. The command prints the selection and progress
bars over the operators (`-q` to silence them).

The command is installed with the package. It needs `pylatex` (`pip install "h4lat[full]"`) and a LaTeX installation
that provides `latexmk` or `pdflatex` (e.g. TeX Live). See `h4lat-catalogue --help` for all options. With a
development install, run `pip install -e .` again if the command is not found.

## From Python

`operator_show` takes the same selection and returns the path of the PDF:

```python
from h4lat import operator_show

operator_show(show=False)                                    # S, P, V, A, T with 0, 1, 2 derivatives
operator_show(X="V", nder=[1, 2, 3], show=False)             # vector operators with 1 to 3 derivatives
operator_show(keys=["V2", "V3", "A3"], n_jobs=None, show=False)  # n_jobs=None: all cores
```

In a script, call it with `n_jobs > 1` under `if __name__ == "__main__":`, as usual with multiprocessing.

## Without LaTeX

If you have no LaTeX installation, the
[`operator_catalogue.ipynb`](https://github.com/Tag-E/h4lat/blob/main/examples/operator_catalogue.ipynb) notebook
shows the most relevant operators (V, A and T with one and two derivatives), and its `show_group` function lists any
other group.

## Exact kinematic factors

The kinematic factors are computed in exact arithmetic: the Clebsch-Gordan coefficients, computed in floating point,
are first converted to the exact numbers they approximate (fractions such as ⅓, or square roots of fractions such as
$`{1/\sqrt{2}}`$). Until October 2026 they entered SymPy as floats, and some kinematic factors that vanish (operators
176, 524, 527, 741 and 744 of the current numbering) were printed as tiny numbers of order 1e-16 instead of 0.
