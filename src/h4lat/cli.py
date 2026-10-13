"""Command-line tools installed with h4lat.

h4lat-catalogue
    Write the PDF catalogue of an operator database (by default the bundled one), with the
    expression, kinematic factor, C-parity, trace condition and index symmetry of every
    operator.  Wraps :func:`h4lat.operator_show`; needs pylatex (``pip install "h4lat[full]"``)
    and a LaTeX installation (latexmk or pdflatex).

Examples::

    h4lat-catalogue                           # writes ./operator_catalogue.pdf
    h4lat-catalogue -o docs/ops.pdf --open    # custom path, open the PDF when done
    h4lat-catalogue --database my_operators   # catalogue of a custom operator database
"""

import argparse
import sys
from pathlib import Path


def catalogue_main(argv=None) -> int:
    """Entry point of ``h4lat-catalogue``; returns the exit status."""
    parser = argparse.ArgumentParser(
        prog="h4lat-catalogue",
        description="Write the PDF catalogue of the h4lat operator database.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("operator_catalogue.pdf"),
        help="PDF file to write (default: operator_catalogue.pdf)",
    )
    parser.add_argument(
        "--database",
        type=Path,
        help="operator database made by make_operator_database (default: the bundled one)",
    )
    parser.add_argument("--title", help="title of the first page (default: states the maximum number of derivatives)")
    parser.add_argument("--author", default="E.T.", help="author on the first page (default: E.T.)")
    parser.add_argument("--open", action="store_true", help="open the PDF with xdg-open when done")
    parser.add_argument("--keep-tex", action="store_true", help="keep the generated .tex file")
    parser.add_argument("-q", "--quiet", action="store_true", help="only report errors")
    args = parser.parse_args(argv)

    if args.database is not None and not args.database.is_dir():
        parser.error(f"no operator database at {args.database}")

    try:
        from pylatex.errors import CompilerError
    except ImportError:
        print('h4lat-catalogue: pylatex is required; install it with: pip install "h4lat[full]"', file=sys.stderr)
        return 1

    from .moments_operator import operator_show

    # operator_show takes the file name without extension, relative to the working directory
    output = args.output.resolve()
    if output.suffix == ".pdf":
        output = output.with_suffix("")
    output.parent.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print("Writing the operator catalogue ...")
    try:
        operator_show(
            operator_database=None if args.database is None else str(args.database),
            title=args.title,
            author=args.author,
            doc_name=str(output),
            show=args.open,
            clean_tex=not args.keep_tex,
        )
    except CompilerError as err:
        # the message also covers a missing compiler: "No LaTex compiler was found ..."
        print(f"h4lat-catalogue: LaTeX compilation failed: {err}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Operator catalogue written to {output}.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(catalogue_main())
