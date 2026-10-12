"""Correctness tests for the H(4) matrices and the Clebsch-Gordan database.

The matrices generated from alpha_list / beta_list / gamma_list are checked against the
group structure (generator relations, homomorphism, character table).  The bundled CG
blocks are checked to span H(4)-invariant subspaces that transform as their irrep and,
all together, the whole tensor-product space.
"""

from pathlib import Path

import numpy as np
import pytest
import sympy as sym

import h4lat.cg_calculator as cc
from h4lat import cg_calc, char_table, class_orders, irrep_index, rep_dim_list, rep_label_list

BUNDLED_PRODUCTS = [
    ((4, 1), (4, 1)),
    ((4, 4), (4, 1)),
    ((6, 1), (4, 1)),
    ((4, 1), (4, 1), (4, 1)),
    ((4, 4), (4, 1), (4, 1)),
    ((6, 1), (4, 1), (4, 1)),
]

TINY = ((2, 1), (2, 1))  # 4-dimensional product, computed in a few seconds


def _params(values, *expected_failures):
    """pytest params for *values*; each (failing_values, reason) marks those as strict xfail."""
    return [
        pytest.param(
            v,
            id=str(v).replace(' ', ''),
            marks=[
                pytest.mark.xfail(strict=True, reason=reason) for failing, reason in expected_failures if v in failing
            ],
        )
        for v in values
    ]


def _as_arrays(h4_mat):
    """cg_calc.h4_mat as a list of (384, d, d) arrays, one per irrep."""
    return [np.array([np.reshape(m, (d, d)) for m in mats]) for mats, d in zip(h4_mat, rep_dim_list)]


def _key(mat):
    """Hashable form of a (4,1) matrix (a signed permutation matrix)."""
    return tuple(np.rint(mat).astype(int).ravel())


def _element_index(h4):
    """Map each group element, given by its (4,1) matrix, to its position in the element lists."""
    return {_key(m): i for i, m in enumerate(h4[cc.fund_index])}


def _product_matrices(h4, irreps):
    """T(g) = D_1(g) ⊗ … ⊗ D_n(g) for every element g, as a (384, dim, dim) array."""
    T = h4[irrep_index[irreps[0]]]
    for label in irreps[1:]:
        D = h4[irrep_index[label]]
        T = np.einsum('gij,gkl->gikjl', T, D).reshape(len(T), T.shape[1] * D.shape[1], -1)
    return T


def _assert_blocks_transform_as_their_irrep(h4, irreps, cg_dict):
    """Each CG block C spans an invariant subspace, T(g) C = C M(g), with tr M(g) = χ(g) of its irrep.

    Moreover, with its columns scaled to unit norm (CGmat_from_block scales each column
    separately), C transforms exactly with the matrices of its irrep: T(g) C = C D(g).
    """
    T = _product_matrices(h4, irreps)
    for irep, blocks in cg_dict.items():
        chi = np.trace(h4[irep], axis1=1, axis2=2)
        for m, C in enumerate(blocks):
            TC = T @ C
            M = np.linalg.pinv(C) @ TC
            assert np.allclose(C @ M, TC), f"block {irep}_{m} {rep_label_list[irep]} is not invariant"
            assert np.allclose(
                np.trace(M, axis1=1, axis2=2), chi
            ), f"block {irep}_{m} does not transform as {rep_label_list[irep]}"
            unit = C / np.linalg.norm(C, axis=0)
            assert np.allclose(T @ unit, unit @ h4[irep]), f"block {irep}_{m} is not in the basis of D(g)"


def _assert_blocks_are_independent(irreps, cg_dict):
    """Together the CG blocks span the whole tensor-product space: no multiplet is missing."""
    dim = int(np.prod([label[0] for label in irreps]))
    blocks = np.hstack([C for copies in cg_dict.values() for C in copies])
    assert blocks.shape[1] == dim
    assert np.linalg.matrix_rank(blocks) == dim


def _snapshot(folder):
    return {p.relative_to(folder): p.read_bytes() for p in Path(folder).rglob('*') if p.is_file()}


@pytest.fixture(scope="module")
def h4(tmp_path_factory):
    """Matrices of all 384 elements in every irrep, generated from the generators."""
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp_path_factory.mktemp("h4gen"))  # force_h4gen writes ./h4_ele
        return _as_arrays(cg_calc((4, 1), (4, 1), force_h4gen=True, verbose=False).h4_mat)


@pytest.fixture(scope="module")
def bundled_h4():
    """Matrices of all 384 elements in every irrep, loaded from the bundled h4_ele data."""
    return _as_arrays(cg_calc((4, 1), (4, 1), verbose=False).h4_mat)


@pytest.fixture(scope="module")
def fake_bundled(tmp_path_factory):
    """A stand-in for the bundled CG database, holding the product TINY."""
    folder = tmp_path_factory.mktemp("fake_bundled")
    cg_calc(*TINY, cgdatabase=str(folder), verbose=False)
    return folder


# ---------------------------------------------------------------------------
# H(4) generators and group elements
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", _params(rep_label_list))
def test_generator_relations(label):
    """α, β satisfy the S(4) presentation α² = β⁴ = (αβ)³ = 1; α, β, γ₁ are orthogonal, γ₁² = 1."""
    ir = irrep_index[label]
    d = rep_dim_list[ir]
    a, b, g = (np.reshape(gen[ir], (d, d)) for gen in (cc.alpha_list, cc.beta_list, cc.gamma_list))
    one = np.eye(d)
    assert np.allclose(a @ a, one)
    assert np.allclose(np.linalg.matrix_power(b, 4), one)
    assert np.allclose(np.linalg.matrix_power(a @ b, 3), one)
    assert np.allclose(g @ g, one)
    for mat in (a, b, g):
        assert np.allclose(mat @ mat.T, one)


def test_generated_elements_are_distinct(h4):
    """The faithful (4,1) representation gives 384 distinct elements."""
    assert len(_element_index(h4)) == 384


@pytest.mark.parametrize("label", _params(rep_label_list))
def test_generated_matrices_form_a_representation(h4, label):
    """D(e) = 1 and D(s g) = D(s) D(g) for s = α, β, γ₁ (which generate H(4)) and every g."""
    index = _element_index(h4)
    fund = h4[cc.fund_index]
    D = h4[irrep_index[label]]
    assert np.allclose(D[0], np.eye(D.shape[1]))
    for gen in (cc.alpha_list, cc.beta_list, cc.gamma_list):
        s = index[_key(gen[cc.fund_index])]
        for g in range(len(fund)):
            assert np.allclose(D[index[_key(fund[s] @ fund[g])]], D[s] @ D[g])


def test_generated_characters_match_char_table(h4):
    """The traces of the generated matrices reproduce char_table, with class sizes class_orders."""
    index = _element_index(h4)
    fund = h4[cc.fund_index]
    inverse = [index[_key(m.T)] for m in fund]

    # conjugacy classes, computed in the faithful (4,1) representation
    cls = np.full(len(fund), -1)
    n_cls = 0
    for g in range(len(fund)):
        if cls[g] < 0:
            for h in range(len(fund)):
                cls[index[_key(fund[h] @ fund[g] @ fund[inverse[h]])]] = n_cls
            n_cls += 1
    assert n_cls == len(class_orders)

    chars = np.array([np.trace(D, axis1=1, axis2=2) for D in h4])
    matched = []
    for c in range(n_cls):
        col = chars[:, cls == c]
        assert np.allclose(col, col[:, :1]), "the characters are not class functions"
        candidates = [j for j in range(n_cls) if np.allclose(col[:, 0], char_table[:, j])]
        assert len(candidates) == 1, f"class {c} matches char_table columns {candidates}"
        assert np.sum(cls == c) == class_orders[candidates[0]]
        matched.append(candidates[0])
    assert sorted(matched) == list(range(n_cls))


@pytest.mark.parametrize("label", _params(rep_label_list))
def test_bundled_h4_ele_matches_generators(h4, bundled_h4, label):
    """The bundled matrices are the ones generated from the current generators."""
    ir = irrep_index[label]
    assert np.allclose(bundled_h4[ir], h4[ir])


# ---------------------------------------------------------------------------
# Bundled CG database
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("irreps", _params(BUNDLED_PRODUCTS))
def test_bundled_cg_blocks_transform_as_their_irrep(h4, irreps):
    """Each bundled CG block spans an invariant subspace that transforms as its irrep."""
    _assert_blocks_transform_as_their_irrep(h4, irreps, cg_calc(*irreps, verbose=False).cg_dict)


@pytest.mark.parametrize("irreps", _params(BUNDLED_PRODUCTS))
def test_bundled_cg_blocks_are_independent(irreps):
    """Together the bundled CG blocks span the whole tensor-product space."""
    _assert_blocks_are_independent(irreps, cg_calc(*irreps, verbose=False).cg_dict)


@pytest.mark.parametrize("irreps", _params(BUNDLED_PRODUCTS))
def test_bundled_cg_blocks_match_their_raw_blocks(irreps):
    """The bundled final blocks are CGmat_from_block of the bundled raw blocks (not stale)."""
    cg = cg_calc(*irreps, verbose=False)
    for (irep, m), raw in cg.raw_cg.items():
        expected = cc.CGmat_from_block(raw, cg.mul_list[irep], m, irreps=irreps)
        assert np.allclose(cg.cg_dict[irep][m], expected), f"block {irep}_{m} {rep_label_list[irep]}"


@pytest.mark.parametrize("irreps", _params(BUNDLED_PRODUCTS))
def test_bundled_cg_blocks_have_definite_c_parity(h4, irreps):
    """Each copy is even or odd under reversal of the derivative indices (definite C-parity).

    The even copies come first, and the numbers m± of even and odd copies of each irrep are
    the ones the characters give: m₊ - m₋ = (1/|G|) Σ_g χ(g) tr(R T(g)).
    """
    perm = cc._c_parity_permutation(irreps)
    if perm is None:  # one derivative: the reversal is the identity
        perm = np.arange(int(np.prod([label[0] for label in irreps])))
    T = _product_matrices(h4, irreps)
    trace_RT = T[:, perm, np.arange(len(perm))].sum(axis=1)  # (R T(g))[r, c] = T(g)[perm[r], c]
    for irep, blocks in cg_calc(*irreps, verbose=False).cg_dict.items():
        signs = []
        for m, C in enumerate(blocks):
            even = np.allclose(C[perm], C, rtol=0, atol=1e-12)
            odd = np.allclose(C[perm], -C, rtol=0, atol=1e-12)
            assert even or odd, f"block {irep}_{m} {rep_label_list[irep]} has no definite C-parity"
            signs.append(1 if even else -1)
        assert signs == sorted(signs, reverse=True), f"{rep_label_list[irep]}: the even copies do not come first"
        chi = np.trace(h4[irep], axis1=1, axis2=2)
        assert np.isclose(sum(signs), np.sum(chi * trace_RT) / len(chi)), f"{rep_label_list[irep]}: wrong m±"


# ---------------------------------------------------------------------------
# CGmat_from_block
# ---------------------------------------------------------------------------


def _symbolic_block(coefficients):
    """A symbolic block whose entry (r, c) is Σ_k coefficients[r][c][k] A[0, k]."""
    coefficients = np.asarray(coefficients, dtype=float)
    A = sym.MatrixSymbol('A', 1, coefficients.shape[2])
    block = np.empty(coefficients.shape[:2], dtype=object)
    for r, c in np.ndindex(block.shape):
        block[r, c] = sum((x * A[0, k] for k, x in enumerate(coefficients[r, c]) if x), sym.S.Zero)
    return block


def test_cgmat_from_block_with_opposite_entries():
    """Multiplicity 1 with entries ℓ and -ℓ (different sympy expressions, one degree of freedom)."""
    ell = [2.0, -1.0, 0.5]
    block = _symbolic_block([[ell, ell], [[0, 0, 0], ell], [[-x for x in ell], [0, 0, 0]]])
    C = cc.CGmat_from_block(block, multiplicity=1, multiplicity_index=0, enforce_c_parity=False)
    assert np.allclose(C, [[1, 1], [0, 1], [-1, 0]])


def test_cgmat_from_block_gives_independent_copies():
    """With multiplicity 2 the copies are independent, even if the entries share symbols."""
    a, b = [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]  # independent linear forms with the symbol A[0, 1] in common
    block = _symbolic_block([[a], [b], [[x + y for x, y in zip(a, b)]]])
    copies = [
        cc.CGmat_from_block(block, multiplicity=2, multiplicity_index=m, enforce_c_parity=False) for m in range(2)
    ]
    assert np.allclose(copies[0], [[1], [0], [1]])
    assert np.allclose(copies[1], [[0], [1], [1]])


def test_cgmat_from_block_rejects_a_wrong_multiplicity():
    """A block with one degree of freedom is not two copies."""
    block = _symbolic_block([[[1.0, 2.0]], [[2.0, 4.0]]])
    with pytest.raises(ValueError, match="multiplicity 2"):
        cc.CGmat_from_block(block, multiplicity=2, multiplicity_index=0, enforce_c_parity=False)


def test_cgmat_from_block_needs_irreps_for_c_parity():
    """enforce_c_parity (the default) needs the factors of the product, to know what to reverse."""
    block = _symbolic_block([[[1.0, 2.0]], [[2.0, 4.0]]])
    with pytest.raises(ValueError, match="pass irreps"):
        cc.CGmat_from_block(block, multiplicity=1, multiplicity_index=0)


def test_cgmat_from_block_gives_exact_c_parity():
    """With enforce_c_parity every copy is exactly even or odd, to the last bit (C_parity compares with ==)."""
    irreps = ((4, 4), (4, 1), (4, 1))
    perm = cc._c_parity_permutation(irreps)
    cg = cg_calc(*irreps, verbose=False)
    for (irep, m), raw in cg.raw_cg.items():
        C = cc.CGmat_from_block(raw, cg.mul_list[irep], m, irreps=irreps)
        assert np.array_equal(C[perm], C) or np.array_equal(C[perm], -C), f"block {irep}_{m}"


@pytest.mark.parametrize("irreps", _params(BUNDLED_PRODUCTS))
def test_cg_dict_follows_multiplicity_order(irreps):
    """cg_dict[irep][m] is the file '{irep}_{m}.npy', whatever the filesystem order."""
    cg = cg_calc(*irreps, verbose=False)
    assert list(cg.cg_dict) == sorted(cg.cg_dict)
    for irep, blocks in cg.cg_dict.items():
        assert len(blocks) == cg.mul_list[irep]
        for m, C in enumerate(blocks):
            assert np.array_equal(C, np.load(f"{cg.cg_folder}/{irep}_{m}.npy", allow_pickle=True))


# ---------------------------------------------------------------------------
# Computing new products
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("irreps", _params([((1, 1), (4, 1)), ((1, 4), (4, 1))]))
def test_products_with_a_one_dimensional_factor(h4, irreps, tmp_path, monkeypatch):
    """Products with a 1-dim factor (scalar and pseudoscalar operators) are computed correctly."""
    monkeypatch.chdir(tmp_path)  # the new CGs are written to ./cg_database
    cg_dict = cg_calc(*irreps, verbose=False).cg_dict
    _assert_blocks_transform_as_their_irrep(h4, irreps, cg_dict)
    _assert_blocks_are_independent(irreps, cg_dict)


# ---------------------------------------------------------------------------
# Generated data never go into the package
# ---------------------------------------------------------------------------


def test_missing_products_are_computed_into_the_working_directory(tmp_path, monkeypatch):
    """A product that is not bundled is computed into ./cg_database and later read from there."""
    empty = tmp_path / "empty_bundled"
    empty.mkdir()
    monkeypatch.setattr(cc, "_BUNDLED_CG_DATABASE", str(empty))
    monkeypatch.chdir(tmp_path)

    cg_calc(*TINY, verbose=False)
    local = tmp_path / "cg_database" / "(2, 1)(2, 1)"
    assert local.is_dir() and (tmp_path / "cg_database" / "(2, 1)(2, 1)_raw").is_dir()
    assert list(empty.iterdir()) == []

    mtimes = {p: p.stat().st_mtime_ns for p in local.iterdir()}
    cg = cg_calc(*TINY, verbose=False)
    assert Path(cg.cg_folder).resolve() == local.resolve()
    assert {p: p.stat().st_mtime_ns for p in local.iterdir()} == mtimes


@pytest.mark.parametrize("flag", ["force_computation", "prescription_changed"])
def test_recomputation_never_writes_into_the_bundled_data(fake_bundled, tmp_path, monkeypatch, flag):
    """Recomputed CGs go to ./cg_database and the bundled data stay untouched."""
    monkeypatch.setattr(cc, "_BUNDLED_CG_DATABASE", str(fake_bundled))
    monkeypatch.chdir(tmp_path)
    before = _snapshot(fake_bundled)

    cg = cg_calc(*TINY, verbose=False, **{flag: True})

    assert _snapshot(fake_bundled) == before
    local = tmp_path / "cg_database" / "(2, 1)(2, 1)"
    assert Path(cg.cg_folder).resolve() == local.resolve()
    for suffix in ("", "_raw"):
        bundled_files = sorted(p.name for p in (fake_bundled / f"(2, 1)(2, 1){suffix}").iterdir())
        assert sorted(p.name for p in (tmp_path / "cg_database" / f"(2, 1)(2, 1){suffix}").iterdir()) == bundled_files


def test_missing_bundled_h4_ele_is_generated_into_the_working_directory(tmp_path, monkeypatch):
    """Without bundled h4_ele data the matrices are generated into ./h4_ele."""
    missing = tmp_path / "no_bundled_h4_ele"
    monkeypatch.setattr(cc, "_BUNDLED_H4_ELE", str(missing))
    monkeypatch.chdir(tmp_path)

    cg = cg_calc((4, 1), (4, 1), verbose=False)  # the CGs themselves are loaded from the bundled database

    assert cg.h4_ele_folder == "h4_ele"
    assert len(list((tmp_path / "h4_ele").iterdir())) == len(rep_label_list)
    assert not missing.exists()
