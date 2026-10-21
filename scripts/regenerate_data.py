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
generators in the code (force_h4gen), never from stale bundled matrices.  With
--from-raw, the cg stage instead only re-applies CGmat_from_block to the bundled raw
blocks (`{product}_raw/`), which takes seconds: use it when CGmat_from_block changed
but the generators did not.  Raw blocks are bundled only for products of up to
MAX_RAW_IRREPS irreps, i.e. for the operators with up to two derivatives: those of larger
products take tens of MB and long to load, and computing such a product takes about a
minute, so --from-raw recomputes them instead.

The operator database holds the operators of every Dirac structure (S, P, V, A, T) with
0 to --max-nder derivatives, so it needs the products irrep(X) ⊗ (4,1)^k for k = 0 (the
irrep of X alone), 1, ..., max_nder.

Examples (from the repository root, with h4lat installed with `pip install -e .`):

  python scripts/regenerate_data.py all --jobs 6
  python scripts/regenerate_data.py h4
  python scripts/regenerate_data.py cg --products "(6,1),(4,1)" "(4,1),(4,1)"
  python scripts/regenerate_data.py cg --products "(1,4)" "(1,4),(4,1)"   # a single irrep: no derivatives
  python scripts/regenerate_data.py cg --from-raw
  python scripts/regenerate_data.py cg --products "(6,1),(4,1),(4,1),(4,1)"
  python scripts/regenerate_data.py operators            # up to --max-nder 3 derivatives

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
MAX_RAW_IRREPS = 3  # raw CG blocks are bundled only for products of up to this many irreps

log = logging.getLogger("regenerate_data")


######################## Helpers ########################################


def product_name(product):
    """Folder name of a tensor product in a CG database, e.g. '(6, 1)(4, 1)'."""
    return "".join(str(irrep) for irrep in product)


def is_irrep(value):
    """Whether *value* is an irrep label, a pair of integers such as (4, 1)."""
    return isinstance(value, tuple) and len(value) == 2 and all(isinstance(x, int) for x in value)


def parse_product(text):
    """Parse a command-line product such as '(6,1),(4,1)' into ((6, 1), (4, 1)).

    A single irrep, '(4,1)', is the product ((4, 1),) of one factor (an operator without
    derivatives).
    """
    try:
        product = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        product = None
    if is_irrep(product):
        product = (product,)
    if not (isinstance(product, tuple) and product and all(is_irrep(irrep) for irrep in product)):
        raise argparse.ArgumentTypeError(f"not an irrep or a product of irreps: {text!r}")
    return product


def bundled_products(data):
    """Tensor products in the CG database of *data*."""
    names = sorted(p.name for p in (data / "cg_database").iterdir() if p.is_dir() and p.name.startswith("("))
    return [
        tuple((int(a), int(b)) for a, b in re.findall(r"\((\d+), (\d+)\)", name))
        for name in names
        if not name.endswith("_raw")
    ]


def operator_products(max_nder):
    """Products that make_operator_database needs for the operators with up to *max_nder* derivatives."""
    from h4lat.moments_operator import _X_ORDER, _operator_irreps

    return [_operator_irreps(X, nder) for nder in range(max_nder + 1) for X in _X_ORDER]


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


def compute_product(product, cg_database, scratch, verbose, raw_source=None):
    """Compute the CG coefficients of one product into *cg_database* (runs in a worker process).

    With *raw_source* (a CG database), the raw blocks of the product are taken from there
    and only CGmat_from_block is re-applied to them.
    """
    from h4lat import cg_calc

    scratch.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with contextlib.chdir(scratch):
        if raw_source is not None:
            name = product_name(product)
            shutil.rmtree(cg_database / name, ignore_errors=True)
            shutil.rmtree(cg_database / (name + "_raw"), ignore_errors=True)
            shutil.copytree(raw_source / (name + "_raw"), cg_database / (name + "_raw"))
            (cg_database / name).mkdir(parents=True)
            # prescription_changed: the final blocks are rebuilt from the raw ones
            cg_calc(*product, cgdatabase=str(cg_database), prescription_changed=True, verbose=verbose)
        else:
            # force_h4gen: matrices from the generators in the code, written to the scratch ./h4_ele
            cg_calc(*product, cgdatabase=str(cg_database), force_computation=True, force_h4gen=True, verbose=verbose)
    shutil.rmtree(scratch)
    return time.time() - start


def regenerate_cg(data, workdir, products, jobs, from_raw=False):
    """Recompute the CGs of *products*, installing each one as soon as it is done.

    With *from_raw*, only the final blocks are rebuilt, from the bundled raw blocks; the
    products without bundled raw blocks (more than MAX_RAW_IRREPS irreps) are recomputed.
    Returns the products that failed.
    """
    raw_source = data / "cg_database" if from_raw else None
    if from_raw:
        missing = [p for p in products if not (raw_source / (product_name(p) + "_raw")).is_dir()]
        if missing:
            log.info("cg: no bundled raw blocks for %s: recomputed", ", ".join(product_name(p) for p in missing))
    cg_database = workdir / "cg_database"
    if jobs > 1:
        os.environ["TQDM_DISABLE"] = "1"  # no interleaved progress bars: tqdm reads it when a worker imports it

    # largest products first, so that the longest computation starts right away
    todo = sorted(products, key=lambda p: -int(np.prod([irrep[0] for irrep in p])))
    failed = []
    with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {}
        for p in todo:
            # with one job, cg_calc output and progress bar are shown (otherwise they would interleave)
            scratch = workdir / "scratch" / product_name(p)
            source = raw_source if from_raw and (raw_source / (product_name(p) + "_raw")).is_dir() else None
            futures[pool.submit(compute_product, p, cg_database, scratch, jobs == 1, source)] = p
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
            # the raw blocks of products of more than MAX_RAW_IRREPS irreps are not bundled
            for suffix in ("", "_raw") if len(p) <= MAX_RAW_IRREPS else ("",):
                install(cg_database / (product_name(p) + suffix), data / "cg_database" / (product_name(p) + suffix))
    return failed


def report_operator_changes(old_folder, new_folder):
    """Log how the operator database changed, in particular at the IDs used in get_op_selection and get_paper_operators.

    The operators of the two databases are matched by their metadata (number of indices, Dirac
    structure, irrep, copy and component, read from the file name), which identifies an operator
    also when its ID changes.  The log states the operators added and removed, those whose tensor
    changed, and how the IDs of the others moved.
    """
    from h4lat import get_op_selection, get_paper_operators

    def load(folder):
        """{ID: (metadata, tensor)} of the operator files operator_{ID}_{metadata}.npy in *folder*."""
        ops = {}
        for f in folder.glob("*.npy"):
            _, i, *meta = f.stem.split("_")
            ops[int(i)] = ("_".join(meta), np.load(f))
        return ops

    def same_tensor(a, b):
        return a.shape == b.shape and np.allclose(a, b)

    old, new = load(old_folder), load(new_folder)

    # --- by operator (metadata): added, removed, changed tensors, moved IDs ---
    old_by_meta = {meta: (i, tensor) for i, (meta, tensor) in old.items()}
    new_by_meta = {meta: (i, tensor) for i, (meta, tensor) in new.items()}
    kept = old_by_meta.keys() & new_by_meta.keys()
    added = sorted(new_by_meta[meta][0] for meta in new_by_meta.keys() - kept)
    removed = sorted(old_by_meta[meta][0] for meta in old_by_meta.keys() - kept)
    changed = sorted(new_by_meta[m][0] for m in kept if not same_tensor(old_by_meta[m][1], new_by_meta[m][1]))
    log.info(
        "operators: %d before, %d now; %d kept, %d of them with a changed tensor",
        len(old),
        len(new),
        len(kept),
        len(changed),
    )
    if added:
        log.info("operators: added, at the new IDs %s", id_ranges(added))
    if removed:
        log.info("operators: removed, from the old IDs %s", id_ranges(removed))
    if changed:
        log.info("operators: changed tensors, at the new IDs %s", id_ranges(changed))
    # old ID -> new ID of the same operator, grouped by the shift (new ID - old ID)
    shifts = {}
    for meta in sorted(kept, key=lambda m: old_by_meta[m][0]):
        old_id, new_id = old_by_meta[meta][0], new_by_meta[meta][0]
        if new_id != old_id:
            shifts.setdefault(new_id - old_id, []).append(old_id)
    for shift, ids in sorted(shifts.items(), key=lambda item: item[1][0]):
        log.warning("operators: renumbered: the operators of the old IDs %s now have ID %+d", id_ranges(ids), shift)

    # --- by ID: what the IDs used in the code refer to now ---
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

    for function in (get_op_selection, get_paper_operators):
        source = inspect.getsource(function)
        used = sorted({int(i) for i in re.findall(r"Operator_from_database\((\d+)\)", source)})
        affected = [i for i in used if statuses.get(i) != "unchanged"]
        if not affected:
            log.info("operators: the %d IDs used by %s are unchanged", len(used), function.__name__)
        for i in affected:
            now_at = []
            if i in old:
                tensor = old[i][1]
                now_at = [j for j, (_, t) in sorted(new.items()) if t.shape == tensor.shape and np.allclose(t, tensor)]
            log.warning(
                "operators: %s uses ID %d: %s; its old tensor is now at IDs %s",
                function.__name__,
                i,
                statuses.get(i),
                now_at,
            )


def regenerate_operators(data, workdir, max_nder):
    """Rebuild the operator database from the bundled CGs, report changed IDs, install."""
    from h4lat import make_operator_database

    missing = [p for p in operator_products(max_nder) if not (data / "cg_database" / product_name(p)).is_dir()]
    if missing:
        products = " ".join(f'"{",".join(str(irrep).replace(" ", "") for irrep in p)}"' for p in missing)
        raise SystemExit(
            f"operators: CG products missing for max_nder = {max_nder}; compute them first with\n"
            f"  python scripts/regenerate_data.py cg --products {products}"
        )

    log.info("operators: building the operator database up to max_nder = %d derivatives", max_nder)
    new = workdir / "operator_database"
    shutil.rmtree(new, ignore_errors=True)
    make_operator_database(str(new), max_nder=max_nder, verbose=True, cg_database=str(data / "cg_database"))
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
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help="cg stage: only rebuild the final blocks from the bundled raw blocks (no projection)",
    )
    parser.add_argument("--jobs", type=int, default=1, help="CG products computed in parallel (default: 1)")
    parser.add_argument(
        "--max-nder",
        type=int,
        default=3,
        help="operator database: the operators with 0 to MAX_NDER derivatives (default: 3)",
    )
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
    if args.jobs < 1 or args.max_nder < 0:
        parser.error("--jobs must be at least 1 and --max-nder at least 0")

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
        failed = regenerate_cg(data, workdir, products, args.jobs, from_raw=args.from_raw)
        if failed:
            log.error("cg: failed: %s; rerun them with --products", ", ".join(product_name(p) for p in failed))
            return 1
    if "operators" in args.stages or "all" in args.stages:
        regenerate_operators(data, workdir, args.max_nder)

    log.info("done in %s", duration(time.time() - start))
    log.info("next: python -m pytest tests/test_cg_correctness.py, the full suite, then commit %s", data)
    if "operators" in args.stages or "all" in args.stages:
        # The default catalogue holds only the operators with up to two derivatives.
        nders = " ".join(str(d) for d in range(args.max_nder + 1))
        log.info("then refresh the operator catalogue PDF with: h4lat-catalogue --nder %s -j", nders)
    return 0


if __name__ == "__main__":
    sys.exit(main())
