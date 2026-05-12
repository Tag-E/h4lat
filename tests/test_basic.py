"""Basic tests for the h4lat package."""

import numpy as np

import h4lat
from h4lat import (
    OPERATOR_DATABASE,
    OperatorDict_from_database,
    OperatorList_from_database,
    cg_calc,
    char_table,
    class_orders,
    get_OperatorDict,
    get_OperatorList,
    get_multiplicities,
    irrep_dim,
    irrep_index,
    latex_print_multiplicities,
    n_ele_h4,
    n_rep,
    rep_dim_list,
    rep_label_list,
)

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
    pass


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
        # Dirac structure must be one of the three supported types.
        assert op.X in ('V', 'A', 'T'), f"Unexpected X={op.X!r} for operator {op.id}"
        # irrep must be a 2-tuple of positive integers.
        assert isinstance(op.irrep, tuple) and len(op.irrep) == 2
        assert all(isinstance(k, int) and k >= 1 for k in op.irrep)
        # Block and column indices are 1-based positive integers.
        assert isinstance(op.block, int) and op.block >= 1
        assert isinstance(op.index_block, int) and op.index_block >= 1
        # n is at least 2 (one Dirac index plus at least one derivative index).
        assert op.n >= 2


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
        # n is the total number of Lorentz indices (≥ 2 for physical operators).
        assert isinstance(n, int) and n >= 2, f"Unexpected n={n} in key {key}"
        assert X in ('V', 'A', 'T'), f"Unexpected X={X!r} in key {key}"


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
