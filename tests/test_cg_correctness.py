"""Correctness tests for the H(4) matrices and the Clebsch-Gordan database.

The matrices generated from alpha_list / beta_list / gamma_list are checked against the
group structure (generator relations, homomorphism, character table).  The bundled CG
blocks are checked to span H(4)-invariant subspaces that transform as their irrep and,
all together, the whole tensor-product space.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
import sympy as sym

import h4lat.cg_calculator as cc
from h4lat import cg_calc, char_table, class_orders, irrep_index, rep_dim_list, rep_label_list
from h4lat.moments_operator import _X_ORDER, _operator_irreps

# The products of the bundled operator database: irrep(X) ⊗ (4,1)^k for X = S, P, V, A, T
# ((1,1), (1,4), (4,1), (4,4), (6,1)) and k = 0 (the irrep alone), 1, 2, 3 derivatives.
BUNDLED_PRODUCTS = [_operator_irreps(X, nder) for nder in range(4) for X in _X_ORDER]
# raw blocks are bundled only for products of up to three irreps, i.e. up to two derivatives
# (see scripts/regenerate_data.py)
BUNDLED_RAW_PRODUCTS = [p for p in BUNDLED_PRODUCTS if len(p) <= 3]
SINGLE_IRREPS = [p for p in BUNDLED_PRODUCTS if len(p) == 1]

TINY = ((2, 1), (2, 1))  # 4-dimensional product, computed in a few seconds

needs_numba = pytest.mark.skipif(not cc._NUMBA_AVAILABLE, reason="numba is not installed")


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


@pytest.mark.parametrize("irreps", _params(BUNDLED_RAW_PRODUCTS))
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


@pytest.mark.parametrize("irreps", _params(SINGLE_IRREPS))
def test_bundled_single_irreps_have_the_identity_as_cg_matrix(irreps):
    """The "product" of one irrep is the irrep itself: one copy, whose CG matrix is the identity (Schur's lemma)."""
    cg = cg_calc(*irreps, verbose=False)
    [(irep, blocks)] = cg.cg_dict.items()  # exactly one (irrep, blocks) pair
    assert rep_label_list[irep] == irreps[0] and len(blocks) == 1
    assert np.array_equal(blocks[0], np.eye(irreps[0][0]))


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


@pytest.mark.parametrize(
    "irreps", _params([((1, 1), (4, 1)), ((1, 4), (4, 1)), ((1, 4), (4, 1), (4, 1)), ((1, 4),), ((6, 1),)])
)
def test_products_with_a_one_dimensional_factor_or_a_single_irrep(h4, irreps, tmp_path):
    """Products with a 1-dim factor (S and P operators) and single irreps (no derivatives) are computed correctly."""
    cg_dict = cg_calc(*irreps, cgdatabase=str(tmp_path), force_computation=True, verbose=False).cg_dict
    _assert_blocks_transform_as_their_irrep(h4, irreps, cg_dict)
    _assert_blocks_are_independent(irreps, cg_dict)


def _symbolic_projection(h4_mat, irreps):
    """Raw CG blocks computed as cg_calc did before the sum over the group was done numerically.

    Σ_h T(h) A B(h)^T is summed with the symbolic matrix A, element by element (far too slow
    beyond small products): the reference that the numerical projection must reproduce.
    """
    mul_list = cc.get_multiplicities(*irreps)
    dim = int(np.prod([label[0] for label in irreps]))
    A = sym.MatrixSymbol('A', dim, dim)
    rep_to_use = [irep for irep, mul in enumerate(mul_list) for _ in range(mul)]

    res_mat = np.zeros((dim, dim))
    for ih in range(cc.n_ele_h4):
        Tp = np.eye(1)
        for label in irreps:
            Tp = np.kron(Tp, h4_mat[irrep_index[label]][ih])
        rows = [
            [
                h4_mat[ri][ih] if ib == jb else np.zeros((rep_dim_list[ri], rep_dim_list[rj]))
                for jb, rj in enumerate(rep_to_use)
            ]
            for ib, ri in enumerate(rep_to_use)
        ]
        res_mat = res_mat + Tp @ A @ np.block(rows).T

    rounded = np.empty((dim, dim), dtype=object)
    for i in range(dim):
        for j in range(dim):
            new_entry = 0
            for element in sym.Add.make_args(res_mat[i, j]):
                coeff, monom = element.as_coeff_Mul()
                if np.abs(coeff) < 1e-10 or np.abs(coeff) > 1e20:
                    coeff = 0
                new_entry = new_entry + coeff * monom
            rounded[i, j] = new_entry

    raw, d = {}, 0
    for irep, mul in enumerate(mul_list):
        for m in range(mul):
            raw[(irep, m)] = rounded[:, d : d + rep_dim_list[irep]]
            d += rep_dim_list[irep]
    return raw


def _assert_same_raw_blocks(raw, expected):
    assert raw.keys() == expected.keys()
    for key, block in raw.items():
        assert block.shape == expected[key].shape
        for entry, expected_entry in zip(block.ravel(), expected[key].ravel()):
            assert sym.sympify(entry) == sym.sympify(expected_entry), f"raw block {key}: {entry} != {expected_entry}"


def _assert_same_cg_dict(cg_dict, expected):
    assert cg_dict.keys() == expected.keys()
    for irep, blocks in cg_dict.items():
        assert len(blocks) == len(expected[irep])
        for m, (C, C_expected) in enumerate(zip(blocks, expected[irep])):
            assert np.array_equal(C, C_expected), f"block {irep}_{m} {rep_label_list[irep]}"


@pytest.mark.parametrize("irreps", _params([TINY, ((1, 4), (4, 1)), ((2, 1), (2, 1), (2, 1))]))
def test_numerical_projection_reproduces_the_symbolic_sum(irreps, tmp_path):
    """The raw blocks are those of the former symbolic sum over the group, to the last bit.

    Both serially and in parallel (if numba is installed).  The products include factors
    that are not signed permutation matrices, (2,1), and an irrep with multiplicity 2 in
    (2,1)⊗(2,1)⊗(2,1).
    """
    expected = None
    for parallel in [False, True] if cc._NUMBA_AVAILABLE else [False]:
        folder = str(tmp_path / f"parallel_{parallel}")
        cg = cg_calc(*irreps, cgdatabase=folder, force_computation=True, parallel=parallel, verbose=False)
        if expected is None:
            expected = _symbolic_projection(cg.h4_mat, irreps)
            expected_cg = {}
            for (irep, m), raw in expected.items():
                expected_cg.setdefault(irep, []).append(cc.CGmat_from_block(raw, cg.mul_list[irep], m, irreps=irreps))
        _assert_same_raw_blocks(cg.raw_cg, expected)
        _assert_same_cg_dict(cg.cg_dict, expected_cg)


@needs_numba
def test_parallel_and_serial_computations_are_identical(tmp_path):
    """parallel=True, with any number of threads, gives the same raw blocks and CG matrices as parallel=False."""
    import numba

    irreps = ((4, 1), (4, 1), (4, 1))
    n_threads_before = numba.get_num_threads()
    serial = cg_calc(
        *irreps, cgdatabase=str(tmp_path / "serial"), force_computation=True, parallel=False, verbose=False
    )
    for n_threads in (None, 1, 2):
        parallel = cg_calc(
            *irreps,
            cgdatabase=str(tmp_path / f"parallel_{n_threads}"),
            force_computation=True,
            parallel=True,
            n_threads=n_threads,
            verbose=False,
        )
        _assert_same_raw_blocks(parallel.raw_cg, serial.raw_cg)
        _assert_same_cg_dict(parallel.cg_dict, serial.cg_dict)
    assert numba.get_num_threads() == n_threads_before, "the numba setting of the caller was not restored"


@needs_numba
def test_n_threads_is_used(tmp_path, monkeypatch):
    """The sum runs on n_threads threads; giving n_threads turns on the parallel computation, also for 2 irreps."""
    import numba

    kernel = cc._numba_projection_kernel()
    used = []

    def recording_kernel(*args):
        used.append(numba.get_num_threads())
        kernel(*args)

    monkeypatch.setattr(cc, "_numba_projection_kernel", lambda: recording_kernel)
    cg_calc(*TINY, cgdatabase=str(tmp_path), force_computation=True, n_threads=2, verbose=False)
    assert used and set(used) == {2}


@needs_numba
@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(n_threads=0), "between 1 and"),
        (dict(n_threads=10**6), "between 1 and"),
        (dict(n_threads=2, parallel=False), "parallel=True"),
    ],
    ids=["zero", "too_many", "with_parallel_False"],
)
def test_n_threads_validation(kwargs, match, tmp_path):
    """A number of threads that numba cannot use, or one given with parallel=False, is an error."""
    with pytest.raises(ValueError, match=match):
        cg_calc(*TINY, cgdatabase=str(tmp_path), force_computation=True, verbose=False, **kwargs)


@pytest.mark.parametrize("irreps", _params(BUNDLED_RAW_PRODUCTS))  # the others take a minute each
def test_recomputed_products_match_the_bundled_data(irreps, tmp_path):
    """Recomputing a bundled product reproduces the bundled CG matrices (up to floating-point noise).

    Not to the last bit: scripts/regenerate_data.py only rewrites the files that change by more
    than 1e-10, so some bundled files come from older versions of the code, and the last bit of
    CGmat_from_block (the SVDs in np.linalg.pinv) depends on the OpenBLAS kernel of the CPU.
    """
    bundled = cg_calc(*irreps, verbose=False).cg_dict
    computed = cg_calc(*irreps, cgdatabase=str(tmp_path), force_computation=True, verbose=False).cg_dict
    assert computed.keys() == bundled.keys()
    for irep, blocks in computed.items():
        for m, (C, C_bundled) in enumerate(zip(blocks, bundled[irep])):
            assert np.allclose(C, C_bundled, rtol=0, atol=1e-12), f"block {irep}_{m} {rep_label_list[irep]}"


def test_parallel_without_numba(tmp_path, monkeypatch):
    """Without numba, parallel=True or n_threads is an error, and by default products of 3 irreps are computed serially."""
    monkeypatch.setattr(cc, "_NUMBA_AVAILABLE", False)
    monkeypatch.setattr(cc, "_numba_projection_kernel", lambda: pytest.fail("numba was used"))
    for kwargs in (dict(parallel=True), dict(n_threads=2)):
        with pytest.raises(ImportError, match="numba"):
            cg_calc(*TINY, cgdatabase=str(tmp_path), force_computation=True, verbose=False, **kwargs)
    irreps = ((2, 1), (2, 1), (2, 1))
    cg_dict = cg_calc(*irreps, cgdatabase=str(tmp_path), force_computation=True, verbose=False).cg_dict
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


def test_raw_blocks_are_loaded_on_access():
    """Loading a product reads only its final blocks; the raw blocks are read when raw_cg is accessed."""
    cg = cg_calc((4, 1), (4, 1), verbose=False)
    assert cg._raw_cg is None
    assert sorted(cg.raw_cg) == sorted((irep, m) for irep, blocks in cg.cg_dict.items() for m in range(len(blocks)))


def test_h4_matrices_are_read_on_access_and_shared():
    """Loading a product reads no H(4) matrix; they are read on first access, only once, and shared read-only."""
    cc._load_h4_matrices.cache_clear()
    a = cg_calc((4, 1), (4, 1), verbose=False)
    b = cg_calc((6, 1), (4, 1), verbose=False)
    assert a._h4_mat is None and b._h4_mat is None
    assert cc._load_h4_matrices.cache_info().currsize == 0
    # b reads the folder, a finds it in the cache: both hold the same array objects.
    assert b.h4_mat[10][5] is a.h4_mat[10][5]
    info = cc._load_h4_matrices.cache_info()
    assert (info.misses, info.hits) == (1, 1)
    # The shared arrays are read-only, so an in-place change cannot alter them for every instance.
    with pytest.raises(ValueError, match="read-only"):
        a.h4_mat[10][5][0, 0] = 2.0


def test_h4_matrix_cache_is_keyed_by_the_absolute_folder(tmp_path, monkeypatch):
    """The ./h4_ele of two working directories are two folders: each cg_calc reads the one of its own."""
    bundled = cc._BUNDLED_H4_ELE
    monkeypatch.setattr(cc, "_BUNDLED_H4_ELE", str(tmp_path / "no_bundled_h4_ele"))  # so that ./h4_ele is used
    for name in ("a", "b"):
        shutil.copytree(bundled, tmp_path / name / "h4_ele")
    # In b, the identity (element 0) of the irrep (4,1) becomes 2·𝟙.
    np.save(tmp_path / "b" / "h4_ele" / "(4, 1)" / "0.npy", 2 * np.eye(4))

    monkeypatch.chdir(tmp_path / "a")
    cg_a = cg_calc((4, 1), (4, 1), verbose=False)  # the CGs themselves are loaded from the bundled database
    monkeypatch.chdir(tmp_path / "b")
    cg_b = cg_calc((4, 1), (4, 1), verbose=False)
    assert cg_a.h4_ele_folder == cg_b.h4_ele_folder == "h4_ele"

    # Both are read in b, after the last chdir: each still reads the folder it was created in.
    ir = irrep_index[(4, 1)]
    assert np.array_equal(cg_b.h4_mat[ir][0], 2 * np.eye(4))
    assert np.array_equal(cg_a.h4_mat[ir][0], np.eye(4))


def test_generated_h4_matrices_clear_the_cache(tmp_path, monkeypatch):
    """Writing new matrices (force_h4gen) empties the cache, whose copy of that folder would be stale."""
    monkeypatch.chdir(tmp_path)  # force_h4gen writes ./h4_ele
    cc._load_h4_matrices.cache_clear()
    _ = cg_calc((4, 1), (4, 1), verbose=False).h4_mat  # caches the bundled matrices
    assert cc._load_h4_matrices.cache_info().currsize == 1
    generated = cg_calc((4, 1), (4, 1), force_h4gen=True, verbose=False)
    assert cc._load_h4_matrices.cache_info().currsize == 0
    assert generated._h4_mat is not None  # the generated matrices are kept, not read back from the files


def test_products_of_four_irreps_are_bundled_without_raw_blocks(tmp_path, monkeypatch):
    """A bundled 4-irrep product loads from its final blocks; asking for its raw blocks is a clear error."""
    monkeypatch.chdir(tmp_path)  # prescription_changed would write to ./cg_database
    irreps = ((4, 1), (4, 1), (4, 1), (4, 1))
    cg = cg_calc(*irreps, verbose=False)
    assert Path(cg.cg_folder).resolve() == (Path(cc._BUNDLED_CG_DATABASE) / "(4, 1)(4, 1)(4, 1)(4, 1)").resolve()
    _assert_blocks_are_independent(irreps, cg.cg_dict)
    with pytest.raises(FileNotFoundError, match="force_computation=True"):
        _ = cg.raw_cg
    with pytest.raises(FileNotFoundError, match="force_computation=True"):
        cg_calc(*irreps, prescription_changed=True, verbose=False)
    assert list(tmp_path.iterdir()) == []  # nothing was written
