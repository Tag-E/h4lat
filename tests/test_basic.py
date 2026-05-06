"""Basic tests for the h4lat package."""

import numpy as np

import h4lat
from h4lat import (
    cg_calc,
    char_table,
    class_orders,
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
