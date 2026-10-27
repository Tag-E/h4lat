"""Command-line tools installed with h4lat.

h4lat-catalogue
    Write the PDF catalogue of a selection of the operators of an operator database (by
    default the bundled one), with the expression, kinematic factor, C-parity, trace
    condition and index symmetry of every operator.  By default it holds the S, P, V, A and
    T operators with zero, one and two derivatives; choose others with -X / --nder or --keys.
    Wraps :func:`h4lat.operator_show`; needs pylatex (``pip install "h4lat[full]"``) and a
    LaTeX installation (latexmk or pdflatex).

Examples::

    h4lat-catalogue                           # S, P, V, A, T, 0-2 derivatives -> operator_catalogue_SPVAT_nder012.pdf
    h4lat-catalogue -X V --nder 1 2 3         # vector operators with 1 to 3 derivatives
    h4lat-catalogue --nder 0                  # the local bilinears ψ̄ Γ ψ
    h4lat-catalogue --keys V2 V3 A3 -j        # a selection (X + number of derivatives), on all cores
    h4lat-catalogue -o docs/ops.pdf --open    # custom path, open the PDF when done
    h4lat-catalogue --database my_operators   # catalogue of a custom operator database
"""

import argparse
import os
import sys
from pathlib import Path

# Shown at the end of --help.
_EPILOG = """\
selection:
  The operators are selected by Dirac structure X (S, P, V, A, T) and number of
  derivatives, either as a product (-X V A --nder 1 2 3) or as keys made of X followed
  by the number of derivatives (--keys V2 V3 A3: 'V2' is the vector operators with two
  derivatives, i.e. three Lorentz indices).  Operator IDs are those of the database: a
  partial catalogue skips some of them.  The default file name and title state the
  selection, e.g. operator_catalogue_SPVAT_nder012.pdf or operator_catalogue_V2_V3_A3.pdf.

examples:
  h4lat-catalogue                           # S, P, V, A, T with 0, 1, 2 derivatives
  h4lat-catalogue -X V --nder 1 2 3         # vector operators with 1 to 3 derivatives
  h4lat-catalogue --keys S0 P0 V2 -j        # a selection, kinematic factors on all cores
  h4lat-catalogue --nder 0 1 2 3 -j 8       # the whole bundled database, on 8 processes
"""


def _split_commas(values: list[str] | None) -> list[str] | None:
    """Split the items of a list option at commas, so that '-X V,A' works like '-X V A'."""
    if values is None:
        return None
    return [item for value in values for item in value.split(",") if item]


def catalogue_main(argv=None) -> int:
    """Entry point of ``h4lat-catalogue``; returns the exit status."""
    parser = argparse.ArgumentParser(
        prog="h4lat-catalogue",
        description="Write the PDF catalogue of (a selection of) the h4lat operator database.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="PDF file to write (default: operator_catalogue_<selection>.pdf in the working directory)",
    )
    parser.add_argument(
        "--database",
        type=Path,
        help="operator database made by make_operator_database (default: the bundled one)",
    )

    # Which operators to write: -X / --nder, or --keys (not both; checked below).
    selection = parser.add_argument_group("operator selection")
    selection.add_argument(
        "-X",
        nargs="+",
        metavar="X",
        help="Dirac structures among S P V A T (default: all of them)",
    )
    selection.add_argument(
        "--nder",
        nargs="+",
        metavar="N",
        help="numbers of derivatives (default: 0 1 2)",
    )
    selection.add_argument(
        "--keys",
        nargs="+",
        metavar="KEY",
        help="groups given as X followed by the number of derivatives, e.g. S0 V2 A3 (not with -X/--nder)",
    )

    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        nargs="?",
        const=0,
        default=1,
        metavar="N",
        help="compute the kinematic factors on N processes (-j alone: all cores; default: 1)",
    )
    parser.add_argument("--title", help="title of the first page (default: states the selected operators)")
    parser.add_argument("--author", default="E.T.", help="author on the first page (default: E.T.)")
    parser.add_argument("--open", action="store_true", help="open the PDF with the default viewer when done")
    parser.add_argument("--keep-tex", action="store_true", help="keep the generated .tex file")
    parser.add_argument("-q", "--quiet", action="store_true", help="only report errors (no progress bars)")
    args = parser.parse_args(argv)

    if args.database is not None and not args.database.is_dir():
        parser.error(f"no operator database at {args.database}")
    if args.keys is not None and (args.X is not None or args.nder is not None):
        parser.error("select the operators either with --keys or with -X/--nder, not both")
    if args.jobs < 0:
        parser.error("--jobs must be a positive number of processes (or -j alone for all cores)")

    # Numbers of derivatives: integers, also accepted as comma-separated lists (--nder 1,2).
    nder = _split_commas(args.nder)
    if nder is not None:
        try:
            nder = [int(d) for d in nder]
        except ValueError:
            parser.error(f"--nder takes integers, got {' '.join(args.nder)}")

    try:
        from pylatex.errors import CompilerError
    except ImportError:
        print('h4lat-catalogue: pylatex is required; install it with: pip install "h4lat[full]"', file=sys.stderr)
        return 1

    from .moments_operator import operator_show

    # operator_show takes the file name without extension, relative to the working
    # directory; without -o it names the file after the selection.
    doc_name = None
    if args.output is not None:
        output = args.output.resolve()
        if output.suffix == ".pdf":
            output = output.with_suffix("")
        output.parent.mkdir(parents=True, exist_ok=True)
        doc_name = str(output)

    try:
        operator_show(
            operator_database=None if args.database is None else str(args.database),
            title=args.title,
            author=args.author,
            doc_name=doc_name,
            # operator_show prints the selection, progress bars over the operators and the result
            verbose=not args.quiet,
            show=args.open,
            clean_tex=not args.keep_tex,
            X=_split_commas(args.X),
            nder=nder,
            keys=_split_commas(args.keys),
            # -j alone (0) means all cores
            n_jobs=args.jobs if args.jobs > 0 else (os.cpu_count() or 1),
        )
    except CompilerError as err:
        # the message also covers a missing compiler: "No LaTex compiler was found ..."
        print(f"h4lat-catalogue: LaTeX compilation failed: {err}", file=sys.stderr)
        return 1
    except ValueError as err:
        # an invalid selection, e.g. an unknown key or operators missing from the database
        print(f"h4lat-catalogue: {err}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(catalogue_main())
