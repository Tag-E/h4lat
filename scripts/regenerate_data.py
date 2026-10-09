"""Regenerate the data bundled with h4lat (src/h4lat/data/).

The bundled data come in three stages, each derived from the previous one:

  h4         matrices of the 384 elements of H(4) in its 20 irreps, built
             from the generators in cg_calculator.py               -> h4_ele/
  cg         Clebsch-Gordan coefficients of the bundled products    -> cg_database/
  operators  lattice operators built from the CG coefficients       -> operator_database/

Each stage is computed in a work directory; the files whose content changed (beyond
floating-point noise) are then copied over the bundled data, which git tracks: review
the result with `git status src/h4lat/data` and throw it away with
`git checkout -- src/h4lat/data`.  CG coefficients are always computed from the
generators in the code (force_h4gen), never from stale bundled matrices.

Examples (from the repository root, with h4lat installed with `pip install -e .`):

  python scripts/regenerate_data.py all --jobs 6
  python scripts/regenerate_data.py h4
  python scripts/regenerate_data.py cg --products "(6,1),(4,1)" "(4,1),(4,1)"
  python scripts/regenerate_data.py operators

Then run `python -m pytest tests/test_cg_correctness.py` (and the full suite) and
commit src/h4lat/data.
"""

import argparse
import ast
import contextlib
import inspect
import logging
import multiprocessing
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
STAGES = ("h4", "cg", "operators")
NOISE = 1e-10  # smaller differences between old and new data are floating-point noise, not changes

log = logging.getLogger("regenerate_data")


######################## Helpers ########################################


def product_name(product):
    """Folder name of a tensor product in a CG database, e.g. '(6, 1)(4, 1)'."""
    return "".join(str(irrep) for irrep in product)


def parse_product(text):
    """Parse a command-line product such as '(6,1),(4,1)' into ((6, 1), (4, 1))."""
    try:
        product = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        product = None
    if not (
        isinstance(product, tuple)
        and len(product) >= 2
        and all(isinstance(irrep, tuple) and len(irrep) == 2 for irrep in product)
    ):
        raise argparse.ArgumentTypeError(f"not a product of at least two irreps: {text!r}")
    return product


def bundled_products(data):
    """Tensor products in the CG database of *data*."""
    names = sorted(p.name for p in (data / "cg_database").iterdir() if p.is_dir() and p.name.startswith("("))
    return [
        tuple((int(a), int(b)) for a, b in re.findall(r"\((\d+), (\d+)\)", name))
        for name in names
        if not name.endswith("_raw")
    ]


def operator_products(max_n):
    """Products that make_operator_database needs for V, A and T operators up to *max_n* indices."""
    leading = {"V": (4, 1), "A": (4, 4), "T": (6, 1)}
    return [(leading[X],) + ((4, 1),) * (n - 1) for n in range(2, max_n + 1) for X in "VAT"]


def duration(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def id_ranges(ids):
    """Compact form of sorted integers: [1, 2, 3, 7] -> '1-3, 7'."""
    ranges = []
    for i in ids:
        if ranges and i == ranges[-1][1] + 1:
            ranges[-1][1] = i
        else:
            ranges.append([i, i])
    return ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in ranges)


def same_content(old, new):
    """Whether two .npy files hold the same array, up to floating-point noise (< NOISE)."""
    if old.read_bytes() == new.read_bytes():
        return True
    a, b = np.load(old, allow_pickle=True), np.load(new, allow_pickle=True)
    if a.shape != b.shape:
        return False
    if a.dtype != object and b.dtype != object:
        return np.allclose(a, b, rtol=0, atol=NOISE)

    import sympy as sym

    # raw CG blocks: linear combinations of the symbols A[i, j] with float coefficients
    return all(
        all(abs(c) < NOISE for c in sym.expand(x - y).as_coefficients_dict().values()) for x, y in zip(a.flat, b.flat)
    )


def changed_files(old, new):
    """Relative paths of the files whose content differs between folders *old* and *new*."""

    def files(folder):
        return {p.relative_to(folder): p for p in folder.rglob("*") if p.is_file()} if folder.exists() else {}

    old_files, new_files = files(old), files(new)
    return sorted(
        p
        for p in old_files.keys() | new_files.keys()
        if p not in old_files or p not in new_files or not same_content(old_files[p], new_files[p])
    )


def install(new, target):
    """Update the bundled folder *target* to *new* and log what changed.

    Only files whose content really changed are rewritten: files that differ by floating-point
    noise alone keep their bytes, so that `git status` shows the actual changes.
    """
    changed = changed_files(target, new)
    for rel in changed:
        if (new / rel).exists():
            (target / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(new / rel, target / rel)
        else:
            (target / rel).unlink()

    folders = sorted({p.parts[0] for p in changed if len(p.parts) > 1})
    if folders:
        where = " in " + ", ".join(folders)
    elif 0 < len(changed) <= 8:
        where = ": " + ", ".join(p.name for p in changed)
    else:
        where = ""
    log.info("installed %s: %d files changed%s", target.name, len(changed), where)


######################## Stages #########################################


def regenerate_h4(data, workdir):
    """Build the H(4) matrices from the generators and install them."""
    from h4lat import cg_calc

    log.info("h4: building the H(4) matrices from the generators")
    shutil.rmtree(workdir / "h4_ele", ignore_errors=True)
    with contextlib.chdir(workdir):
        # force_h4gen writes the matrices to ./h4_ele; the (4,1)x(4,1) CGs are only loaded
        cg_calc((4, 1), (4, 1), force_h4gen=True, verbose=False)
    install(workdir / "h4_ele", data / "h4_ele")


def compute_product(product, cg_database, scratch):
    """Compute the CG coefficients of one product into *cg_database* (runs in a worker process)."""
    from h4lat import cg_calc

    scratch.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with contextlib.chdir(scratch):
        # force_h4gen: matrices from the generators in the code, written to the scratch ./h4_ele
        cg_calc(*product, cgdatabase=str(cg_database), force_computation=True, force_h4gen=True, verbose=False)
    shutil.rmtree(scratch)
    return time.time() - start


def regenerate_cg(data, workdir, products, jobs):
    """Recompute the CGs of *products*, installing each one as soon as it is done.

    Returns the products that failed.
    """
    cg_database = workdir / "cg_database"
    if jobs > 1:
        os.environ["TQDM_DISABLE"] = "1"  # no interleaved progress bars: tqdm reads it when a worker imports it

    # largest products first, so that the longest computation starts right away
    todo = sorted(products, key=lambda p: -int(np.prod([irrep[0] for irrep in p])))
    failed = []
    with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {}
        for p in todo:
            futures[pool.submit(compute_product, p, cg_database, workdir / "scratch" / product_name(p))] = p
            log.info("cg: queued %s", product_name(p))
        for future in as_completed(futures):
            p = futures[future]
            try:
                elapsed = future.result()
            except Exception:
                log.exception("cg: %s failed", product_name(p))
                failed.append(p)
                continue
            log.info("cg: %s computed in %s", product_name(p), duration(elapsed))
            for suffix in ("", "_raw"):
                install(cg_database / (product_name(p) + suffix), data / "cg_database" / (product_name(p) + suffix))
    return failed


def report_operator_changes(old_folder, new_folder):
    """Log which operator IDs changed, in particular those hard-coded in get_op_selection."""
    from h4lat import get_op_selection

    def load(folder):
        ops = {}
        for f in folder.glob("*.npy"):
            _, i, *meta = f.stem.split("_")
            ops[int(i)] = ("_".join(meta), np.load(f))
        return ops

    old, new = load(old_folder), load(new_folder)

    def status(i):
        if i not in old or i not in new:
            return "added" if i in new else "removed"
        (meta_old, a), (meta_new, b) = old[i], new[i]
        if meta_old != meta_new or a.shape != b.shape:
            return f"now {meta_new} (was {meta_old})"
        if np.allclose(a, b):
            return "unchanged"
        return "sign flipped" if np.allclose(a, -b) else "changed"

    statuses = {i: status(i) for i in sorted(old.keys() | new.keys())}
    changed = [i for i, s in statuses.items() if s != "unchanged"]
    log.info("operators: %d of %d unchanged", len(statuses) - len(changed), len(statuses))
    if changed:
        log.info("operators: changed IDs: %s", id_ranges(changed))

    source = inspect.getsource(get_op_selection)
    used = sorted({int(i) for i in re.findall(r"Operator_from_database\((\d+)\)", source)})
    affected = [i for i in used if statuses.get(i) != "unchanged"]
    if not affected:
        log.info("operators: the %d IDs used by get_op_selection are unchanged", len(used))
    for i in affected:
        now_at = []
        if i in old:
            tensor = old[i][1]
            now_at = [j for j, (_, t) in sorted(new.items()) if t.shape == tensor.shape and np.allclose(t, tensor)]
        log.warning(
            "operators: get_op_selection uses ID %d: %s; its old tensor is now at IDs %s", i, statuses.get(i), now_at
        )


def regenerate_operators(data, workdir, max_n):
    """Rebuild the operator database from the bundled CGs, report changed IDs, install."""
    from h4lat import make_operator_database

    missing = [p for p in operator_products(max_n) if not (data / "cg_database" / product_name(p)).is_dir()]
    if missing:
        products = " ".join(f'"{",".join(str(irrep).replace(" ", "") for irrep in p)}"' for p in missing)
        raise SystemExit(
            f"operators: CG products missing for max_n = {max_n}; compute them first with\n"
            f"  python scripts/regenerate_data.py cg --products {products}"
        )

    log.info("operators: building the operator database up to max_n = %d", max_n)
    new = workdir / "operator_database"
    shutil.rmtree(new, ignore_errors=True)
    make_operator_database(str(new), max_n=max_n, verbose=True, cg_database=str(data / "cg_database"))
    report_operator_changes(data / "operator_database", new)
    install(new, data / "operator_database")


######################## Command line ###################################


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="+", choices=STAGES + ("all",), help="what to regenerate")
    parser.add_argument(
        "--products",
        nargs="+",
        type=parse_product,
        metavar="PRODUCT",
        help='CG products for the cg stage, e.g. "(6,1),(4,1)" (default: all bundled products)',
    )
    parser.add_argument("--jobs", type=int, default=1, help="CG products computed in parallel (default: 1)")
    parser.add_argument("--max-n", type=int, default=3, help="max_n of the operator database (default: 3)")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO / "src" / "h4lat" / "data",
        help="data folder to update (default: src/h4lat/data of this repository)",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=REPO / "regenerated_data",
        help="where the stages compute before installing (default: regenerated_data/)",
    )
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.max_n < 2:
        parser.error("--jobs must be at least 1 and --max-n at least 2")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    import h4lat

    imported_from = Path(h4lat.__file__).resolve().parent
    if imported_from != REPO / "src" / "h4lat":
        parser.error(f"h4lat is imported from {imported_from}, not from this repository: run `pip install -e .`")
    data = args.data_dir.resolve()
    products = args.products or bundled_products(data)
    unknown = sorted({irrep for p in products for irrep in p if irrep not in h4lat.rep_label_list})
    if unknown:
        parser.error(f"unknown irreps: {unknown}")
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    if "h4" in args.stages or "all" in args.stages:
        regenerate_h4(data, workdir)
    if "cg" in args.stages or "all" in args.stages:
        failed = regenerate_cg(data, workdir, products, args.jobs)
        if failed:
            log.error("cg: failed: %s; rerun them with --products", ", ".join(product_name(p) for p in failed))
            return 1
    if "operators" in args.stages or "all" in args.stages:
        regenerate_operators(data, workdir, args.max_n)

    log.info("done in %s", duration(time.time() - start))
    log.info("next: python -m pytest tests/test_cg_correctness.py, the full suite, then commit %s", data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
