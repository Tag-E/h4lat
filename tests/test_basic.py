"""Basic tests for the h4lat package."""

import inspect
import itertools as it
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import sympy as sym

import h4lat
from h4lat import (
    AVAILABLE_PAPERS,
    OPERATOR_DATABASE,
    Gamma_pol_list,
    Operator_from_database,
    OperatorDict_from_database,
    OperatorList_from_database,
    cg_calc,
    cg_remapping,
    cg_remapping_T,
    char_table,
    class_orders,
    decomposition_analysis,
    get_multiplicities,
    get_op_selection,
    get_OperatorDict,
    get_OperatorList,
    get_paper_operators,
    index_symm,
    index_symm_index_fixed,
    irrep_dim,
    irrep_index,
    latex_print_multiplicities,
    make_operator_database,
    n_ele_h4,
    n_rep,
    operator_show,
    rep_dim_list,
    rep_label_list,
)
from h4lat.cli import catalogue_main
from h4lat.kinematic_data import E, I, mN, p1, p2, p3
from h4lat.moments_operator import _catalogue_selection, _compute_kinematic_factors, _exact_number, trace_symm

# ---------------------------------------------------------------------------
# Group constants
# ---------------------------------------------------------------------------


def test_n_rep():
    assert n_rep == 20


def test_n_ele_h4():
    assert n_ele_h4 == 384


def test_rep_label_list_length():
    assert len(rep_label_list) == 20


def test_rep_dim_list():
    assert rep_dim_list == [1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 6, 6, 6, 6, 8, 8]


def test_irrep_index_roundtrip():
    for i, label in enumerate(rep_label_list):
        assert irrep_index[label] == i


def test_irrep_dim_values():
    for label in rep_label_list:
        assert irrep_dim[label] == label[0]


def test_char_table_shape():
    assert char_table.shape == (20, 20)


def test_class_orders_sum():
    # Sum of class orders = group order = 384
    assert sum(class_orders) == 384


# ---------------------------------------------------------------------------
# Multiplicity formula
# ---------------------------------------------------------------------------


def test_multiplicities_trivial():
    """(1,1) ⊗ any irrep = that irrep (trivial rep tensored with any irrep)."""
    for label in rep_label_list:
        muls = get_multiplicities((1, 1), label)
        idx = irrep_index[label]
        assert muls[idx] == 1
        assert sum(muls) == 1, f"trivial tensor {label} should give multiplicity sum 1, got {muls}"


def test_multiplicities_fund_x_fund():
    """(4,1) ⊗ (4,1) — total dimension must be 4×4 = 16."""
    muls = get_multiplicities((4, 1), (4, 1))
    total_dim = sum(m * rep_dim_list[i] for i, m in enumerate(muls))
    assert total_dim == 16


def test_multiplicities_nonnegative():
    """All multiplicities must be non-negative integers."""
    for l1 in [(1, 1), (4, 1), (6, 1)]:
        for l2 in [(4, 1), (4, 4)]:
            muls = get_multiplicities(l1, l2)
            assert all(m >= 0 for m in muls)
            assert all(isinstance(m, int) for m in muls)


def test_multiplicities_dimension_conservation():
    """Sum of (multiplicity × dim) must equal the product of input dims."""
    cases = [
        ((4, 1), (4, 1)),
        ((4, 4), (4, 1)),
        ((6, 1), (4, 1)),
        ((4, 1), (4, 1), (4, 1)),
    ]
    for irreps in cases:
        muls = get_multiplicities(*irreps)
        expected_dim = 1
        for irep in irreps:
            expected_dim *= irep[0]
        total_dim = sum(m * rep_dim_list[i] for i, m in enumerate(muls))
        assert total_dim == expected_dim, f"{irreps}: expected dim {expected_dim}, got {total_dim}"


# ---------------------------------------------------------------------------
# latex_print_multiplicities
# ---------------------------------------------------------------------------


def test_latex_string_starts_and_ends():
    s = latex_print_multiplicities((4, 1), (4, 1))
    assert s.startswith("$")
    assert s.endswith("$")
    assert r"\otimes" in s
    assert r"\oplus" in s


# ---------------------------------------------------------------------------
# cg_calc — load from bundled database
# ---------------------------------------------------------------------------


def test_cg_calc_loads_fund_x_fund():
    """Load (4,1)⊗(4,1) CG database without recomputing."""
    cg = cg_calc((4, 1), (4, 1), verbose=False)
    assert hasattr(cg, "cg_dict")
    assert len(cg.cg_dict) > 0


def test_cg_calc_mul_list_consistent():
    """mul_list returned by cg_calc must match get_multiplicities."""
    cg = cg_calc((4, 1), (4, 1), verbose=False)
    expected = get_multiplicities((4, 1), (4, 1))
    assert cg.mul_list == expected


def test_cg_calc_matrix_shapes():
    """Each CG matrix column should have length = product of input irrep dims."""
    cg = cg_calc((4, 1), (4, 1), verbose=False)
    expected_rows = 4 * 4  # both irreps are 4-dimensional
    for irep_idx, matrices in cg.cg_dict.items():
        for mat in matrices:
            assert (
                mat.shape[0] == expected_rows
            ), f"Irrep {rep_label_list[irep_idx]}: expected {expected_rows} rows, got {mat.shape[0]}"
            assert (
                mat.shape[1] == rep_dim_list[irep_idx]
            ), f"Irrep {rep_label_list[irep_idx]}: expected {rep_dim_list[irep_idx]} cols"


def test_cg_calc_matrix_dtype():
    """CG matrices should be real float arrays."""
    cg = cg_calc((4, 1), (4, 1), verbose=False)
    for matrices in cg.cg_dict.values():
        for mat in matrices:
            assert np.issubdtype(mat.dtype, np.floating)


def test_cg_calc_axial():
    """(4,4)⊗(4,1) should also load correctly."""
    cg = cg_calc((4, 4), (4, 1), verbose=False)
    assert len(cg.cg_dict) > 0


def test_cg_calc_tensor():
    """(6,1)⊗(4,1) should also load correctly."""
    cg = cg_calc((6, 1), (4, 1), verbose=False)
    assert len(cg.cg_dict) > 0


def test_cg_calc_three_irreps():
    """(4,1)⊗(4,1)⊗(4,1) — verify dimension conservation."""
    cg = cg_calc((4, 1), (4, 1), (4, 1), verbose=False)
    total_dim = sum(m * rep_dim_list[i] for i, m in enumerate(cg.mul_list))
    assert total_dim == 4 * 4 * 4


def test_cg_calc_get_multiplicities_method():
    cg = cg_calc((4, 1), (4, 1), verbose=False)
    assert cg.get_multiplicities() == cg.mul_list


# ---------------------------------------------------------------------------
# Package-level imports
# ---------------------------------------------------------------------------


def test_package_version():
    assert hasattr(h4lat, "__version__")
    assert isinstance(h4lat.__version__, str)


def test_public_api_present():
    assert all(hasattr(h4lat, name) for name in h4lat.__all__)


# ---------------------------------------------------------------------------
# h4lat-catalogue — command-line wrapper of operator_show
# ---------------------------------------------------------------------------


def _run_catalogue(monkeypatch, argv):
    """Run h4lat-catalogue with operator_show replaced by a recorder; return its kwargs."""
    pytest.importorskip("pylatex")
    import h4lat.moments_operator

    calls = []
    monkeypatch.setattr(h4lat.moments_operator, "operator_show", lambda **kwargs: calls.append(kwargs))
    assert catalogue_main(argv) == 0
    assert len(calls) == 1
    return calls[0]


def test_operator_show_in_public_api():
    assert "operator_show" in h4lat.__all__


def test_catalogue_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    kwargs = _run_catalogue(monkeypatch, [])
    # Without -o, operator_show names the file after the selection, in the working directory.
    assert kwargs["doc_name"] is None
    assert kwargs["operator_database"] is None
    assert kwargs["show"] is False
    assert kwargs["clean_tex"] is True
    # Verbose (progress bars), the default selection (V, A, T with 1, 2 derivatives), serial.
    assert kwargs["verbose"] is True
    assert kwargs["X"] is None and kwargs["nder"] is None and kwargs["keys"] is None
    assert kwargs["n_jobs"] == 1


def test_catalogue_options(monkeypatch, tmp_path):
    output = tmp_path / "sub" / "dir" / "ops.pdf"
    kwargs = _run_catalogue(
        monkeypatch,
        ["-o", str(output), "--database", OPERATOR_DATABASE, "--open", "--keep-tex", "-q"]
        + ["-X", "V,A", "T", "--nder", "1", "3", "-j", "3"],
    )
    assert kwargs["doc_name"] == str(output.with_suffix(""))
    assert output.parent.is_dir()
    assert kwargs["operator_database"] == OPERATOR_DATABASE
    assert kwargs["show"] is True
    assert kwargs["clean_tex"] is False
    assert kwargs["verbose"] is False
    # Comma-separated and space-separated values are both accepted.
    assert kwargs["X"] == ["V", "A", "T"]
    assert kwargs["nder"] == [1, 3]
    assert kwargs["keys"] is None
    assert kwargs["n_jobs"] == 3


def test_catalogue_keys_and_all_cores(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    kwargs = _run_catalogue(monkeypatch, ["--keys", "V2,V3", "A3", "-j"])
    assert kwargs["keys"] == ["V2", "V3", "A3"]
    assert kwargs["X"] is None and kwargs["nder"] is None
    # -j without a number: all cores
    assert kwargs["n_jobs"] == (os.cpu_count() or 1)


@pytest.mark.parametrize(
    "argv",
    [
        ["--keys", "V2", "-X", "A"],  # keys and X/nder are mutually exclusive
        ["--keys", "V2", "--nder", "2"],
        ["--nder", "x"],  # not an integer
        ["-j", "-2"],  # negative number of processes
    ],
)
def test_catalogue_invalid_options(argv):
    with pytest.raises(SystemExit):
        catalogue_main(argv)


def test_catalogue_missing_database(tmp_path):
    with pytest.raises(SystemExit):
        catalogue_main(["--database", str(tmp_path / "missing")])


# ---------------------------------------------------------------------------
# operator_show — selection of the operators written to the catalogue
# ---------------------------------------------------------------------------


def test_catalogue_selection_default():
    """By default: every Dirac structure (S, P, V, A, T) with zero, one and two derivatives."""
    selection, title, name = _catalogue_selection()
    assert selection == [(x, d) for x in "SPVAT" for d in (0, 1, 2)]
    assert title == "Available Operators: X = S, P, V, A, T with 0, 1, 2 derivatives"
    assert name == "operator_catalogue_SPVAT_nder012"


@pytest.mark.parametrize(
    "kwargs, title, name",
    [
        (
            dict(X="V", nder=[1, 2, 3]),
            "Available Operators: X = V with 1, 2, 3 derivatives",
            "operator_catalogue_V_nder123",
        ),
        (dict(X=["A"], nder=1), "Available Operators: X = A with 1 derivative", "operator_catalogue_A_nder1"),
        # no derivatives; the structures are listed in the order S, P, V, A, T
        (dict(keys="P0 s0"), "Available Operators: X = S, P with 0 derivatives", "operator_catalogue_SP_nder0"),
        (dict(X="TV", nder=0), "Available Operators: X = V, T with 0 derivatives", "operator_catalogue_VT_nder0"),
        (
            dict(keys=["V2", "V3", "A3"]),
            "Available Operators: V with 2, 3 derivatives; A with 3 derivatives",
            "operator_catalogue_V2_V3_A3",
        ),
        # The same selection gets the same title and name, however it is written:
        # keys in any order and case, or X and nder (with repetitions).
        (
            dict(keys="a2, v1 A1,V2"),
            "Available Operators: X = V, A with 1, 2 derivatives",
            "operator_catalogue_VA_nder12",
        ),
        (
            dict(X="AV", nder=(2, 1, 2)),
            "Available Operators: X = V, A with 1, 2 derivatives",
            "operator_catalogue_VA_nder12",
        ),
    ],
)
def test_catalogue_selection_titles_and_names(kwargs, title, name):
    assert _catalogue_selection(**kwargs)[1:] == (title, name)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(keys=["V2"], X="V"),  # keys cannot be combined with X / nder
        dict(keys=["V2"], nder=2),
        dict(keys=["Q2"]),  # unknown Dirac structure
        dict(keys=["V-1"]),  # a non-negative number of derivatives
        dict(keys=["V"]),  # no number of derivatives
        dict(keys=[]),
        dict(X="VZ"),
        dict(X=""),
        dict(nder=-1),
        dict(nder=[1, 2.5]),
        dict(nder=True),
    ],
)
def test_catalogue_selection_errors(kwargs):
    with pytest.raises(ValueError):
        _catalogue_selection(**kwargs)


def _capture_catalogue(monkeypatch, tmp_path, **kwargs):
    """Run operator_show without compiling: return the PDF path and the LaTeX source it would compile."""
    pylatex = pytest.importorskip("pylatex")
    sources = []
    # Replace the LaTeX compilation by a recorder of the document source.
    monkeypatch.setattr(pylatex.Document, "generate_pdf", lambda self, *args, **kw: sources.append(self.dumps()))
    monkeypatch.chdir(tmp_path)
    path = operator_show(show=False, verbose=False, **kwargs)
    assert len(sources) == 1
    return path, sources[0]


def _catalogue_ids(tex):
    """Operator IDs printed in a catalogue, in order of appearance."""
    return [int(i) for i in re.findall(r"\\text\{Operator (\d+)\}", tex)]


def test_operator_show_writes_only_the_selection(monkeypatch, tmp_path):
    path, tex = _capture_catalogue(monkeypatch, tmp_path, keys=["V1"])
    # The V operators with one derivative, with their database IDs.
    assert path == Path("operator_catalogue_V_nder1.pdf")
    assert _catalogue_ids(tex) == list(range(25, 41))
    assert "Available Operators: X = V with 1 derivative" in tex
    assert "X=V, n=2 (1 derivative)" in tex


def test_operator_show_keeps_the_database_order(monkeypatch, tmp_path):
    """Sections come in database order (increasing ID) whatever the order of the keys."""
    _, tex = _capture_catalogue(monkeypatch, tmp_path, keys=["T1", "A1"])
    assert _catalogue_ids(tex) == list(range(41, 81))  # A1: 41-56, T1: 57-80


def test_operator_show_in_parallel_gives_the_same_catalogue(monkeypatch, tmp_path):
    _, serial = _capture_catalogue(monkeypatch, tmp_path, keys=["V1"])
    _, parallel = _capture_catalogue(monkeypatch, tmp_path, keys=["V1"], n_jobs=2)
    assert parallel == serial


def test_operator_show_missing_operators(monkeypatch, tmp_path):
    # Groups missing from the database are left out with a warning ...
    with pytest.warns(UserWarning, match="V4"):
        _, tex = _capture_catalogue(monkeypatch, tmp_path, keys=["V1", "V4"])
    assert _catalogue_ids(tex) == list(range(25, 41))
    # ... and if nothing is left, it is an error.
    with pytest.raises(ValueError, match="None of the selected operators"):
        _capture_catalogue(monkeypatch, tmp_path, keys=["V4"])


def test_operator_show_invalid_n_jobs():
    pytest.importorskip("pylatex")
    with pytest.raises(ValueError, match="n_jobs"):
        operator_show(keys=["V1"], n_jobs=0, show=False, verbose=False)


@pytest.mark.skipif(
    shutil.which("latexmk") is None and shutil.which("pdflatex") is None, reason="needs a LaTeX installation"
)
def test_operator_show_compiles_a_small_catalogue(monkeypatch, tmp_path):
    pytest.importorskip("pylatex")
    monkeypatch.chdir(tmp_path)
    path = operator_show(keys="V1", show=False, verbose=False)
    assert path == Path("operator_catalogue_V_nder1.pdf")
    assert (tmp_path / path).is_file()
    # clean_tex=True (default): the LaTeX source is removed
    assert not (tmp_path / "operator_catalogue_V_nder1.tex").exists()


def test_parallel_kinematic_factors_match_the_serial_ones():
    """K computed in worker processes equals K computed on access."""
    ids = [1, 26, 65, 176, 471]
    serial = [Operator_from_database(i) for i in ids]
    parallel = [Operator_from_database(i) for i in ids]
    _compute_kinematic_factors(serial, n_jobs=1, verbose=False)
    _compute_kinematic_factors(parallel, n_jobs=2, verbose=False)
    for a, b in zip(serial, parallel):
        assert a.K == b.K
        assert a.latex_K == b.latex_K


# ---------------------------------------------------------------------------
# get_OperatorList — bundled database convenience getter
# ---------------------------------------------------------------------------


def test_get_OperatorList_returns_nonempty_list():
    """get_OperatorList() must return a non-empty list of Operator objects."""
    ops = get_OperatorList()
    # The bundled database ships with at least one operator.
    assert isinstance(ops, list)
    assert len(ops) > 0


def test_get_OperatorList_sorted_by_id():
    """Operators returned by get_OperatorList() must be sorted by ascending id."""
    ops = get_OperatorList()
    ids = [op.id for op in ops]
    # Each id must be a positive integer and the list must be non-decreasing.
    assert ids == sorted(ids)
    assert all(isinstance(i, int) and i >= 1 for i in ids)


def test_get_OperatorList_matches_OperatorList_from_database():
    """get_OperatorList() and OperatorList_from_database() must return the same operators."""
    ops_getter = get_OperatorList()
    # Call the general function with no path so it also falls back to the bundled DB.
    ops_direct = OperatorList_from_database()
    assert len(ops_getter) == len(ops_direct)
    # Check that id, Dirac structure, and irrep agree for every operator.
    for a, b in zip(ops_getter, ops_direct):
        assert a.id == b.id
        assert a.X == b.X
        assert a.irrep == b.irrep


def test_get_OperatorList_matches_explicit_path():
    """get_OperatorList() must agree with OperatorList_from_database(OPERATOR_DATABASE)."""
    ops_explicit = OperatorList_from_database(OPERATOR_DATABASE)
    ops_getter = get_OperatorList()
    # Both should resolve to exactly the same set of operators.
    assert len(ops_explicit) == len(ops_getter)
    for a, b in zip(ops_explicit, ops_getter):
        assert a.id == b.id


def test_get_OperatorList_operator_attributes():
    """Every operator returned by get_OperatorList() must have well-formed attributes."""
    ops = get_OperatorList()
    for op in ops:
        # Dirac structure must be one of the five supported types.
        assert op.X in ('S', 'P', 'V', 'A', 'T'), f"Unexpected X={op.X!r} for operator {op.id}"
        # irrep must be a 2-tuple of positive integers.
        assert isinstance(op.irrep, tuple) and len(op.irrep) == 2
        assert all(isinstance(k, int) and k >= 1 for k in op.irrep)
        # Block and column indices are 1-based positive integers.
        assert isinstance(op.block, int) and op.block >= 1
        assert isinstance(op.index_block, int) and op.index_block >= 1
        # n counts the indices of the Dirac structure (0 for S/P, 1 for V/A, 2 for T) and of the derivatives.
        assert op.n == op.nder + {'S': 0, 'P': 0, 'V': 1, 'A': 1, 'T': 2}[op.X]
        assert op.nder >= 0


def test_bundled_operators_have_definite_c_parity():
    """Copies of an irrep are chosen with definite C-parity, so no bundled operator is 'mixed'."""
    mixed = [op.id for op in get_OperatorList() if op.C not in (1, -1)]
    assert mixed == [], f"operators without definite C-parity: {mixed}"


def test_op_selection_has_definite_c_parity():
    """Every selected operator has a definite irrep and C-parity; the (6,2) tensors are block 1, C = -1."""
    for n_der in (1, 2, 3):
        for op in get_op_selection(n_der):
            assert op.irrep is not None and op.C in (1, -1), f"n_der={n_der}, operator {op.id}"
    for op in get_op_selection(2)[11:]:
        assert (op.irrep, op.block, op.C) == ((6, 2), 1, -1), f"operator {op.id}"


def test_op_selection_labels():
    """Every selected operator carries its label as id."""
    expected = {
        1: ["V1", "V2", "V3", "A1", "A2", "T1", "T2", "T3", "T4"],
        2: ["V1", "V2", "V3", "V4", "A1", "A2", "A3", "A4", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"],
        3: ["V", "A", "T1", "T2", "T1'", "T2'"],
    }
    for n_der, ids in expected.items():
        assert [op.id for op in get_op_selection(n_der)] == ids, f"n_der={n_der}"


def test_op_selection_three_derivatives():
    """3 derivatives: V and A are {1,2,3,4}; the (4,2) tensors have K != 0 and traceless differences T - T'."""
    opV, opA, opT1, opT2, opT1p, opT2p = get_op_selection(3)
    # the completely symmetric combination of 4 indices
    symmetric = np.zeros((4, 4, 4, 4))
    symmetric[tuple(np.array(list(it.permutations(range(4)))).T)] = 1.0
    assert (opV.X, opV.irrep, opV.C) == ("V", (1, 2), 1) and np.array_equal(opV.cgmat, symmetric)
    assert (opA.X, opA.irrep, opA.C) == ("A", (1, 3), -1) and np.array_equal(opA.cgmat, symmetric)
    for op in (opT1, opT2, opT1p, opT2p):
        assert (op.X, op.irrep, op.C) == ("T", (4, 2), 1), f"operator {op.id}"
        assert sym.simplify(op.K) != 0, f"operator {op.id}"
    assert trace_symm((opT1 - opT1p).cgmat, "T") == "= 0"
    assert trace_symm((opT2 - opT2p).cgmat, "T") == "= 0"


# ---------------------------------------------------------------------------
# get_paper_operators — operators used in published works
# ---------------------------------------------------------------------------


def test_paper_operators_labels():
    """Every paper returns its operators with their labels as id."""
    expected = {
        "Burger:2021knd": ["O^5_{234}", "O^5_{134}", "O^5_{124}", "O^5_{123}"]
        + [f"O^{i}_1" for i in range(1, 9)]
        + [f"O^{i}_2" for i in range(1, 9)],
        "Rodekamp:2023wpe": ["V1", "V2", "V3", "A1", "A2", "T1", "T2", "T3", "T4"],
        "Taggi:2026skn": ["V1", "V2", "V3", "V4", "A1", "A2", "A3", "A4"] + [f"T{i}" for i in range(1, 10)],
        "Alexandrou:2026tjs": ["{2,3,4}", "{1,3,4}", "{1,2,4}", "{1,2,3}", "{1,2,3,4}"]
        + ["{4,4,2,3} - {1,1,2,3}", "{4,4,1,3} - {2,2,1,3}", "{4,4,1,2} - {3,3,1,2}"],
        "Gockeler:2004wp": ["v_2a", "v_2b", "v_3", "v_4"],
    }
    assert list(expected) == list(AVAILABLE_PAPERS)
    for paper, ids in expected.items():
        assert [op.id for op in get_paper_operators(paper)] == ids, paper


def test_paper_operators_irreps():
    """Dirac structure, number of indices and irrep of every operator; all have a definite C-parity."""
    expected = {
        "Burger:2021knd": [("A", 3, (4, 3))] * 4 + [("A", 3, (8, 1))] * 8 + [("T", 3, (8, 1))] * 8,
        "Rodekamp:2023wpe": [("V", 2, (3, 1))] * 2
        + [("V", 2, (6, 3))]
        + [("A", 2, (6, 4))] * 2
        + [("T", 3, (8, 1))] * 2
        + [("T", 3, (8, 2))] * 2,
        "Taggi:2026skn": [("V", 3, (4, 2))] * 4
        + [("A", 3, (4, 3))] * 4
        + [("T", 4, (3, 2))]
        + [("T", 4, (3, 3))] * 2
        + [("T", 4, (6, 2))] * 6,
        "Alexandrou:2026tjs": [("V", 3, (4, 2))] * 4 + [("V", 4, (1, 2))] + [("V", 4, (6, 2))] * 3,
        "Gockeler:2004wp": [("V", 2, (6, 3)), ("V", 2, (3, 1)), ("V", 3, (8, 1)), ("V", 4, (2, 1))],
    }
    for paper, structure in expected.items():
        ops = get_paper_operators(paper)
        assert [(op.X, op.n, op.irrep) for op in ops] == structure, paper
        assert all(op.C in (1, -1) for op in ops), paper


def test_paper_operators_selection():
    """A paper is selected by texkey, by DOI (with or without prefix, in any case) or by both."""
    key, doi = "Taggi:2026skn", "10.1103/kknt-zxsw"
    reference = [op.cgmat for op in get_paper_operators(paper=key)]
    for selection in (
        dict(doi=doi),
        dict(doi=f"https://doi.org/{doi}"),
        dict(doi=f"doi:{doi.upper()}"),
        dict(paper=key, doi=f"http://dx.doi.org/{doi}"),
    ):
        ops = get_paper_operators(**selection)
        assert len(ops) == len(reference), selection
        assert all(np.array_equal(op.cgmat, cgmat) for op, cgmat in zip(ops, reference)), selection
    with pytest.raises(ValueError, match="INSPIRE texkey"):
        get_paper_operators()
    with pytest.raises(ValueError, match="paper 'Nobody:2000abc'"):
        get_paper_operators("Nobody:2000abc")
    with pytest.raises(ValueError, match="DOI '10.1103/none'"):
        get_paper_operators(doi="10.1103/none")
    with pytest.raises(ValueError, match="different papers"):
        get_paper_operators(paper="Rodekamp:2023wpe", doi=doi)


def test_paper_operators_explicit():
    """Two operators written out: a sum over (6,2) blocks of Alexandrou et al. and a tensor of Bürger et al."""
    # {4,4,2,3} - {1,1,2,3}: every ordering of the indices, completely symmetric and traceless
    expected = np.zeros((4, 4, 4, 4))
    for indices in set(it.permutations((3, 3, 1, 2))):
        expected[indices] += 1
    for indices in set(it.permutations((0, 0, 1, 2))):
        expected[indices] -= 1
    op = get_paper_operators("Alexandrou:2026tjs")[5]
    assert np.allclose(op.cgmat, expected)
    assert (op.irrep, op.block, op.symm, op.tr) == ((6, 2), None, "Symmetric", "= 0")
    # O^1_2 = (σ_31 D_1 - σ_34 D_4) / (2√2); the cgmat holds σ_μν with μ < ν only
    expected = np.zeros((4, 4, 4))
    expected[0, 2, 0] = expected[2, 3, 3] = -1 / (2 * np.sqrt(2))
    assert np.allclose(get_paper_operators("Burger:2021knd")[12].cgmat, expected)


def test_paper_operators_gockeler():
    """The operators of Göckeler et al. are those of Eq. (31) of the paper, with the C-parities of its Table 2."""

    def braces(*indices):
        """{i,j,...}: every distinct ordering of the (1-based) indices, with coefficient 1."""
        tensor = np.zeros((4,) * len(indices))
        for ordering in set(it.permutations(i - 1 for i in indices)):
            tensor[ordering] += 1
        return tensor

    # Eq. (31), written out index by index
    expected = [
        braces(1, 4),
        braces(4, 4) - 1 / 3 * (braces(1, 1) + braces(2, 2) + braces(3, 3)),
        braces(4, 4, 1) - 1 / 2 * (braces(2, 2, 1) + braces(3, 3, 1)),
        braces(1, 1, 4, 4)
        + braces(2, 2, 3, 3)
        - 1 / 2 * (braces(1, 1, 3, 3) + braces(1, 1, 2, 2) + braces(2, 2, 4, 4) + braces(3, 3, 4, 4)),
    ]
    ops = get_paper_operators(doi="https://doi.org/10.1103/PhysRevD.71.114511")
    assert len(ops) == len(expected)
    for op, cgmat in zip(ops, expected):
        assert np.allclose(op.cgmat, cgmat), op.id
    assert [op.C for op in ops] == [1, 1, -1, 1]
    # v_3 and v_4 are sums over blocks 1 and 2 of their irrep: completely symmetric, but in no single block
    assert [op.block for op in ops] == [1, 1, None, None]
    assert all(op.symm == "Symmetric" and op.tr == "= 0" for op in ops)


def test_paper_operators_gockeler_kinematic_factors():
    """At p = (p_1, 0, 0), K of each operator of Göckeler et al. is a constant times its ratio in Eq. (39)."""
    # the ratios of Eq. (39), for v_2a, v_2b, v_3 and v_4 (without the moments v_n)
    ratios = [I * p1, -(E**2 + p1**2 / 3) / E, -I * p1 * E, p1**2 * E]
    factors = []
    for op, ratio in zip(get_paper_operators("Gockeler:2004wp"), ratios):
        # K depends on m_N only through m_N^2 = E^2 - p_1^2
        K = op.K.subs({p2: 0, p3: 0}).subs(mN**2, E**2 - p1**2)
        factors.append(sym.simplify(K / ratio))
    # the factors come from the normalisation of the operators and of K
    assert factors == [-2 * I, -I, -3, 6 * I]


def test_paper_operators_notebook_code():
    """The code shown in examples/paper_operators.ipynb is that of get_paper_operators."""
    notebook = Path(__file__).parents[1] / "examples" / "paper_operators.ipynb"
    if not notebook.is_file():
        pytest.skip("the example notebooks are not available")
    source = [line.strip() for line in inspect.getsource(get_paper_operators).splitlines() if line.strip()]
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    markdown = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown"]
    blocks = [block for text in markdown for block in re.findall(r"```python\n(.*?)```", text, re.DOTALL)]
    assert len(blocks) == len(AVAILABLE_PAPERS)
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        assert any(source[i : i + len(lines)] == lines for i in range(len(source))), lines[0]


def test_get_paper_operators_in_public_api():
    """get_paper_operators and AVAILABLE_PAPERS must appear in h4lat.__all__."""
    assert {"get_paper_operators", "AVAILABLE_PAPERS"} <= set(h4lat.__all__)


def test_get_OperatorList_in_public_api():
    """get_OperatorList must appear in h4lat.__all__."""
    assert "get_OperatorList" in h4lat.__all__


# ---------------------------------------------------------------------------
# get_OperatorDict — bundled database convenience getter
# ---------------------------------------------------------------------------


def test_get_OperatorDict_returns_nonempty_dict():
    """get_OperatorDict() must return a non-empty dict."""
    d = get_OperatorDict()
    assert isinstance(d, dict)
    assert len(d) > 0


def test_get_OperatorDict_key_structure():
    """Top-level keys of get_OperatorDict() must be (n, X) pairs with valid values."""
    d = get_OperatorDict()
    for key in d:
        n, X = key
        # n is the total number of Lorentz indices (0 for ψ̄ψ and ψ̄γ₅ψ).
        assert isinstance(n, int) and n >= 0, f"Unexpected n={n} in key {key}"
        assert X in ('S', 'P', 'V', 'A', 'T'), f"Unexpected X={X!r} in key {key}"


def test_get_OperatorDict_matches_OperatorDict_from_database():
    """get_OperatorDict() must produce the same key structure as OperatorDict_from_database()."""
    d_getter = get_OperatorDict()
    d_direct = OperatorDict_from_database()
    # Same top-level keys.
    assert set(d_getter.keys()) == set(d_direct.keys())
    # Same inner keys for every (n, X) group.
    for key in d_getter:
        assert set(d_getter[key].keys()) == set(d_direct[key].keys())


def test_get_OperatorDict_contains_vector_operators():
    """The bundled dict must contain 2-index vector operators, including the scalar channel."""
    d = get_OperatorDict()
    # (2, 'V') corresponds to the simplest V-type operators with one derivative.
    assert (2, 'V') in d, "Missing (2, 'V') entry in operator dict"
    # The product (4,1)⊗(4,1) decomposes into (1,1), (3,1), (6,1), (6,3).
    # Check that the scalar channel (1,1) is present with block 1.
    assert ((1, 1), 1) in d[(2, 'V')], "Missing (1,1), block 1 in V 2-index dict"
    ops = d[(2, 'V')][(1, 1), 1]
    assert isinstance(ops, list) and len(ops) > 0


def test_get_OperatorDict_in_public_api():
    """get_OperatorDict must appear in h4lat.__all__."""
    assert "get_OperatorDict" in h4lat.__all__


# ---------------------------------------------------------------------------
# Bundled operator database: content
# ---------------------------------------------------------------------------


def test_bundled_operator_database_covers_three_derivatives():
    """S, P, V, A and T operators with 0 to 3 derivatives, one per basis vector of each product, IDs 1-1360."""
    d = get_OperatorDict()
    counts = {key: sum(len(ops) for ops in group.values()) for key, group in d.items()}
    # keys (n, X), with n = nder for S/P, nder + 1 for V/A, nder + 2 for T, in database order
    # fmt: off
    assert list(counts.items()) == [
        ((0, 'S'), 1), ((0, 'P'), 1), ((1, 'V'), 4), ((1, 'A'), 4), ((2, 'T'), 6),
        ((1, 'S'), 4), ((1, 'P'), 4), ((2, 'V'), 16), ((2, 'A'), 16), ((3, 'T'), 24),
        ((2, 'S'), 16), ((2, 'P'), 16), ((3, 'V'), 64), ((3, 'A'), 64), ((4, 'T'), 96),
        ((3, 'S'), 64), ((3, 'P'), 64), ((4, 'V'), 256), ((4, 'A'), 256), ((5, 'T'), 384),
    ]
    # fmt: on
    ids = sorted(op.id for group in d.values() for ops in group.values() for op in ops)
    assert ids == list(range(1, 1361))


def test_bundled_operator_ids_are_ordered_by_derivatives_then_dirac_structure():
    """IDs run over the number of derivatives 0, 1, 2, 3 and, for each, over X = S, P, V, A, T."""
    ops = get_OperatorList()
    keys = [(op.nder, "SPVAT".index(op.X)) for op in ops]
    assert keys == sorted(keys)
    # first ID of each group: 16 · 4^nder operators per number of derivatives
    first = {}
    for op in ops:
        first.setdefault(f"{op.X}{op.nder}", op.id)
    # fmt: off
    assert first == {
        "S0": 1, "P0": 2, "V0": 3, "A0": 7, "T0": 11,
        "S1": 17, "P1": 21, "V1": 25, "A1": 41, "T1": 57,
        "S2": 81, "P2": 97, "V2": 113, "A2": 177, "T2": 241,
        "S3": 337, "P3": 401, "V3": 465, "A3": 721, "T3": 977,
    }
    # fmt: on


def test_local_bilinears():
    """The operators without derivatives ψ̄ Γ ψ: their cgmat, C-parity and tree-level kinematic factor."""
    ops = get_OperatorList(nder=0)
    assert [op.id for op in ops] == list(range(1, 17))
    assert [op.X for op in ops] == ["S", "P"] + ["V"] * 4 + ["A"] * 4 + ["T"] * 6
    S, P, V, A, T = ops[0], ops[1], ops[2:6], ops[6:10], ops[10:]
    # ψ̄ψ and ψ̄γ₅ψ carry no Lorentz index: a 0-index cgmat
    for op in (S, P):
        assert op.cgmat.shape == () and op.cgmat == 1 and op.n == 0
    assert S.latex_O == "O^{S}" and P.latex_O == "O^{P}"
    # γ_μ, γ_μγ₅: one component per μ; σ_μν: one component per pair μ < ν (12, 13, 23, 14, 24, 34)
    for group in (V, A):
        assert np.array_equal(np.array([op.cgmat for op in group]), np.eye(4))
    pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
    for op, (mu, nu) in zip(T, pairs):
        expected = np.zeros((4, 4))
        expected[mu, nu] = 1
        assert np.array_equal(op.cgmat, expected)
    # C-parity of the local bilinears: + for 𝟙, γ₅, γ_μγ₅; - for γ_μ, σ_μν
    assert [op.C for op in ops] == [1, 1] + [-1] * 4 + [1] * 4 + [-1] * 6
    # nothing to permute: (vacuously) symmetric; no trace
    assert all(op.symm == "Symmetric" and op.tr == "= 0" for op in ops)
    # forward matrix elements: ū u = 2m (K = m/E), ū γ₅ u = 0, ū γ_μ u = 2 p_μ (K = -i p_μ / E, the charge K = 1)
    assert S.K == mN / E
    assert P.K == 0
    p = [p1, p2, p3, I * E]
    assert [sym.simplify(op.K + I * p_mu / E) for op, p_mu in zip(V, p)] == [0] * 4
    assert V[3].K == 1


def test_operators_with_derivatives_of_S_and_P():
    """S and P with derivatives: n = nder indices, all of them derivatives, and C = (-1)^nder."""
    for X in "SP":
        for nder in (1, 2, 3):
            ops = get_OperatorList(X=X, nder=nder)
            assert len(ops) == 4**nder
            assert all(op.n == nder and op.cgmat.shape == (4,) * nder for op in ops)
            assert all(op.C == (-1) ** nder for op in ops if op.symm == "Symmetric"), (X, nder)
            assert all(op.C in (1, -1) for op in ops)
    # with one derivative there is one index: ψ̄ D_μ ψ in the irrep (4,1), ψ̄ γ₅ D_μ ψ in (4,4)
    assert {op.irrep for op in get_OperatorList(keys="S1")} == {(4, 1)}
    assert {op.irrep for op in get_OperatorList(keys="P1")} == {(4, 4)}


def test_operator_getters_select_by_dirac_structure_and_derivatives():
    """get_OperatorList / get_OperatorDict select operators with X, nder or keys, as operator_show."""
    assert [op.id for op in get_OperatorList(X="S", nder=2)] == list(range(81, 97))
    assert [op.id for op in get_OperatorList(keys="V0 A0")] == list(range(3, 11))
    assert [op.id for op in get_OperatorList(X="SP", nder=[0, 1])] == [1, 2] + list(range(17, 25))
    assert len(get_OperatorList(X="T")) == 6 + 24 + 96 + 384
    assert len(get_OperatorList(nder=3)) == 1024
    assert list(get_OperatorDict(X="S", nder=2)) == [(2, "S")]
    # σ_μν without derivatives and γ_μ D_ν both have n = 2 indices; T0 (IDs 11-16) comes first
    assert list(get_OperatorDict(keys=["T0", "V1"])) == [(2, "T"), (2, "V")]
    assert OperatorList_from_database(OPERATOR_DATABASE, X="P", nder=0)[0].id == 2
    for kwargs in (dict(keys="V1", X="V"), dict(X="Q"), dict(nder=-1), dict(keys="V")):
        with pytest.raises(ValueError):
            get_OperatorList(**kwargs)


def test_make_operator_database_reproduces_the_bundled_database(tmp_path):
    """make_operator_database(folder, max_nder) writes the first operators of the bundled database, file by file."""
    folder = tmp_path / "operators"
    make_operator_database(str(folder), max_nder=1)
    bundled = {f.name: f for f in Path(OPERATOR_DATABASE).glob("*.npy")}
    written = sorted(folder.glob("*.npy"), key=lambda f: int(f.name.split("_")[1]))
    assert [int(f.name.split("_")[1]) for f in written] == list(range(1, 81))
    for f in written:
        assert f.name in bundled
        assert np.array_equal(np.load(f), np.load(bundled[f.name])), f.name
    # a second database in the same folder would mix two sets of IDs
    with pytest.raises(FileExistsError):
        make_operator_database(str(folder), max_nder=0)
    for max_nder in (-1, 1.5, True):
        with pytest.raises(ValueError, match="max_nder"):
            make_operator_database(str(tmp_path / "other"), max_nder=max_nder)


def test_cg_remapping():
    """A CG column is the C-order flattening of the cgmat; σ_μν is stored for μ < ν only."""
    assert cg_remapping(np.array([2.0]), 0).shape == ()
    assert cg_remapping(np.array([2.0]), 0) == 2
    assert np.array_equal(cg_remapping(np.arange(4.0), 1), np.arange(4.0))
    two = cg_remapping(np.arange(16.0), 2)
    assert all(two[i, j] == 4 * i + j for i in range(4) for j in range(4))
    pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
    sigma = cg_remapping_T(np.arange(1.0, 7.0), 1)  # σ_μν without derivatives
    expected = np.zeros((4, 4))
    for k, (mu, nu) in enumerate(pairs):
        expected[mu, nu] = k + 1
    assert np.array_equal(sigma, expected)
    sigma_D = cg_remapping_T(np.arange(24.0), 2)  # σ_μν D_ρ
    assert all(sigma_D[mu, nu, rho] == 4 * k + rho for k, (mu, nu) in enumerate(pairs) for rho in range(4))
    assert np.count_nonzero(sigma_D) == 23  # every entry with μ < ν but the one equal to 0


def test_index_symmetry_without_indices_to_permute():
    """With fewer than two indices to permute, the symmetry is (vacuously) 'Symmetric'."""
    assert index_symm(np.array(1.0)) == "Symmetric"
    assert index_symm(np.array([1.0, 0, 0, 0])) == "Symmetric"
    assert index_symm_index_fixed(np.eye(4), 0) == "Symmetric"
    # with two indices nothing changes
    assert index_symm(np.eye(4)) == "Symmetric"
    assert index_symm(np.triu(np.ones((4, 4)), 1)) == "Mixed Symmetry"


def test_operator_without_indices_arithmetic_and_hdf5(tmp_path):
    """An operator with a 0-index cgmat supports the operator arithmetic and the HDF5 round trip."""
    S = Operator_from_database(1)
    combined = 2 * S + S - S / 2
    assert combined.cgmat.shape == () and combined.cgmat == 2.5
    assert (combined.X, combined.n, combined.C) == ("S", 0, 1)
    assert sym.simplify(combined.K - sym.Rational(5, 2) * mN / E) == 0
    pytest.importorskip("h5py")
    from h4lat import h5_to_operator_list, operator_list_to_h5

    ops = get_OperatorList(nder=0)
    operator_list_to_h5(ops, str(tmp_path / "local.h5"))
    reloaded = h5_to_operator_list(str(tmp_path / "local.h5"))
    for op, op_r in zip(ops, reloaded):
        assert (op_r.id, op_r.X, op_r.irrep, op_r.n) == (op.id, op.X, op.irrep, op.n)
        assert op_r.cgmat.shape == op.cgmat.shape and np.array_equal(op_r.cgmat, op.cgmat)


def test_decomposition_analysis_with_zero_and_one_derivative():
    """Without derivatives the product is the irrep of X alone, mixing-safe; ψ̄ D_μ ψ can mix with ψ̄ γ_μ ψ."""
    pytest.importorskip("pandas")
    df = decomposition_analysis("V", 0, operator_dict=get_OperatorDict(nder=0))
    assert df.values.tolist() == [[(4, 1), "Y", 1, "S"]]
    df = decomposition_analysis("S", 1)
    assert df.values.tolist() == [[(4, 1), "N", 1]]
    with pytest.raises(ValueError, match="n_der"):
        decomposition_analysis("S", -1)


def test_operator_symbolic_attributes_are_computed_on_access():
    """O, K and their LaTeX forms are computed on first access; set_polarization_matrix resets K."""
    from h4lat.moments_operator import Kfactor_from_diracO, Operator_from_database, diracO_from_cgmat

    op = Operator_from_database(25)
    assert not {"O", "K", "latex_O", "latex_K", "p3corr_is_real"} & set(op.__dict__)
    assert op.K == Kfactor_from_diracO(diracO_from_cgmat(op.cgmat, op.X), 0)
    latex_K = op.latex_K
    assert {"K", "latex_K"} <= set(op.__dict__)

    op.set_polarization_matrix(2)
    assert not {"K", "latex_K", "p3corr_is_real"} & set(op.__dict__)
    assert op.K == Kfactor_from_diracO(diracO_from_cgmat(op.cgmat, op.X), 2)
    op.set_polarization_matrix(0)
    assert op.latex_K == latex_K

    with pytest.raises(ValueError, match="gamma_pol_index"):
        type(op)(cgmat=op.cgmat, id=1, X=op.X, irrep=op.irrep, block=1, index_block=1, gamma_pol_index=6)


# ---------------------------------------------------------------------------
# Kinematic factor: exact arithmetic
# ---------------------------------------------------------------------------


def test_exact_number():
    """Floating-point CG coefficients are mapped to the exact numbers they approximate."""
    # noise of the numerical CG computation, as found in the bundled data
    assert _exact_number(2.000000000000001) == 2
    assert _exact_number(-2.999999999999989) == -3
    assert _exact_number(1 / 3) == sym.Rational(1, 3)
    # ±√(p/q), as in operators built with 1/np.sqrt(2)
    assert _exact_number(1 / np.sqrt(2)) == sym.sqrt(2) / 2
    assert _exact_number(-2 / np.sqrt(6)) == -sym.sqrt(6) / 3
    assert _exact_number(0.0) == 0
    # neither p/q nor ±√(p/q) with a small q: kept as a float
    for x in (np.pi, 1 + np.sqrt(2), 0.1234567):
        assert isinstance(_exact_number(float(x)), sym.Float)


def test_polarisation_projectors_are_exact():
    """No float in the projectors: a 0.5 would bring floating-point residues back into K."""
    assert not any(entry.atoms(sym.Float) for projector in Gamma_pol_list for entry in projector)


@pytest.mark.parametrize("operator_id", [176, 524, 527, 741, 744])
def test_kinematic_factors_that_vanish_are_exactly_zero(operator_id):
    """With the float CG coefficients these K came out as ~1e-16, e.g. -7105427357601 i p1 p2 p3 / (8·10²⁷ E)."""
    op = Operator_from_database(operator_id)
    assert op.K == 0
    assert op.latex_K == "0"


def test_known_kinematic_factors():
    """K of a few operators (V with 1 and 3 derivatives, T with 1): the same as before exact arithmetic."""
    expected = {
        26: I * (-3 * mN**2 - 4 * p1**2 - 4 * p2**2 - 4 * p3**2) / E,
        65: -3 * p2 * (E * mN + mN**2 + p1**2 + p2**2) / (2 * E * (E + mN)),
        471: 2
        * I
        * (
            -(mN**2) * p1**2
            - mN**2 * p2**2
            + 2 * mN**2 * p3**2
            - p1**4
            - 4 * p1**2 * p2**2
            + 2 * p1**2 * p3**2
            - p2**4
            + 2 * p2**2 * p3**2
            + 2 * p3**4
        )
        / E,
    }
    for operator_id, K in expected.items():
        op = Operator_from_database(operator_id)
        assert sym.simplify(op.K - K) == 0, f"operator {operator_id}"
        assert not op.K.atoms(sym.Float)
    # the operator expression is printed as before (only non-zero coefficients are summed)
    assert (
        Operator_from_database(65).latex_O == r"O^{T}_{1, 2, 2}-\frac{1}{2}O^{T}_{1, 3, 3}-\frac{1}{2}O^{T}_{1, 4, 4}"
    )


def test_latex_K_with_square_roots():
    """Exact square roots in K (here 1/√2 · operator 38) are written as LaTeX roots."""
    op = get_op_selection(1)[2]
    assert op.K == sym.sqrt(2) * p1
    assert op.latex_K == r"\sqrt{2}p_1"


def test_latex_conversion_table():
    """The float → LaTeX table is built on first use, and still reachable as numerics_to_latex_conv."""
    # In a fresh interpreter, importing h4lat does not build the table (it takes a few seconds).
    code = "import h4lat.kinematic_data as k; print(k.latex_conversion_table.cache_info().currsize)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "0"

    from h4lat import kinematic_data

    table = kinematic_data.latex_conversion_table()
    assert kinematic_data.numerics_to_latex_conv is table
    assert len(table) == 2799010
    assert table[1] == "" and table[3] == "3"
    assert table[0.5] == r"\frac{1}{2}"
    assert table[np.round(2 / 3, decimals=13)] == r"\frac{2}{3}"
    assert table[1 / np.sqrt(2)] == r"\frac{1}{\sqrt{2}}"
    # the first (num, den) giving a value wins: √2 = 2/√2 is written with den = 2
    assert table[2 / np.sqrt(2)] == r"\frac{2}{\sqrt{2}}"
