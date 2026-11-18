######################################################
## honeycomb.py  (h4lat package module)             ##
## created by Emilio Taggi - 2026/11/18             ##
######################################################

#########################################################################
# This program is free software: you can redistribute it and/or modify  #
# it under the terms of the GNU General Public License as published by  #
# the Free Software Foundation, either version 3 of the License, or     #
# (at your option) any later version.                                   #
#                                                                       #
# This program is distributed in the hope that it will be useful,       #
# but WITHOUT ANY WARRANTY; without even the implied warranty of        #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         #
# GNU General Public License for more details.                          #
#                                                                       #
# You should have received a copy of the GNU General Public License     #
# along with this program.  If not, see <http://www.gnu.org/licenses/>. #
#########################################################################

"""
Construction of the 16-cell honeycomb group W(F₄) and of the matrices of its 25 irreps.

The 16-cell honeycomb is the lattice Z⁴ together with its copy shifted by (½, ½, ½, ½) (S. Katz and
D. Nogradi, arXiv:2611.21543, sec. 3).  Its point group W(F₄) has 1152 = 3 · 384 elements and
contains the hypercubic group H(4) of the ordinary lattice.  The paper's matrix A of eq. (3.2),

    A = ½ [[1, 1, 1, −1], [1, 1, −1, 1], [1, −1, 1, 1], [1, −1, −1, −1]],

is a rotation by 120° (A³ = 1) that permutes the three orthonormal frames of short lattice vectors
(the "triality"), and W(F₄) is the union of the three cosets H(4), H(4)·A and H(4)·A².

Everything here is built from h4lat's own H(4) matrices and from A, without external programs:

1. ``elements``: the 1152 matrices of the group in the defining 4-dim representation, in the order
   used everywhere in h4lat: element 384·j + i is H(4)[i] · Aʲ, where H(4)[i] is the i-th element of
   H(4) in h4lat's order.  So elements 0 … 383 are exactly h4lat's H(4), element 0 the identity.
2. ``conjugacy_classes``: the class of every element, the classes being numbered in the order in which
   they first appear in that list (the column order of the character table in ``groups.HONEYCOMB``).
3. ``irrep_matrices``: the matrices of all the elements in each of the 25 irreps, real and orthogonal.
   The 1-dim irreps are their characters, (4,1) = R the defining matrices, (4,4) = det·R, and
   (6,1) = Λ²R (antisymmetric tensors) in the basis of the axis pairs 12, 13, 23, 14, 24, 34, the order
   that moments_operator assumes for σ_μν.  Every other irrep is cut out of a tensor product of
   irreps already built, in which it appears exactly once (see _RECIPE): the character projector onto
   it is applied to the standard basis vectors, which are then orthonormalised (Gram-Schmidt).
   Finally every entry is replaced by the exact number it stands for (a rational or ±√rational), so
   that the matrices do not depend on rounding errors.

The functions are pure: they compute and return arrays and never read or write files.  The matrices
are stored in the package by ``cg_calculator`` (folder ``honeycomb_ele``).
"""

import math
from fractions import Fraction

import numpy as np

from .groups import HONEYCOMB

######################## Matrices of the paper ###########################

TRIALITY = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [1, -1, -1, -1]], dtype=float) / 2
"""The triality rotation A = e^(2) of arXiv:2611.21543, eq. (3.2): A³ = 1, W(F₄) = H(4) ∪ H(4)A ∪ H(4)A²."""

GENERATORS = (
    np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=float),
    np.array([[-1, -1, 1, 1], [1, -1, -1, 1], [1, 1, 1, 1], [1, -1, 1, -1]], dtype=float) / 2,
)
"""The two generators g₁ (eq. 3.1) and g₂ (eq. 3.3) of W(F₄) given in arXiv:2611.21543 (g₁² = g₂⁶ = 1)."""

# Pairs of axes (0-based) labelling the basis of the 6-dim irrep (6,1) = Λ²R: 12, 13, 23, 14, 24, 34.
# This must be the order of _SIGMA_PAIRS in moments_operator (and of the H(4) irrep (6,1)).
_PAIRS = ((0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3))

# How each remaining irrep is built: target ⊂ a ⊗ b, where the target appears exactly once in the
# tensor product of the two irreps a and b, which are built before it.  The order of the list is the
# construction order.  Several targets can come from the same product (e.g. the four 2-dim irreps).
_RECIPE = (
    ((4, 2), (1, 2), (4, 1)),  # χ_s ⊗ R
    ((4, 3), (1, 2), (4, 4)),  # χ_s ⊗ det ⊗ R = χ_l ⊗ R
    ((6, 2), (1, 2), (6, 1)),  # χ_s ⊗ A = χ_l ⊗ A
    ((9, 1), (4, 1), (4, 1)),  # H₂ ⊂ R ⊗ R
    ((9, 4), (4, 1), (4, 4)),  # det ⊗ H₂ ⊂ R ⊗ det R
    ((16, 1), (4, 1), (6, 1)),  # R ⊗ A = R ⊕ det R ⊕ (16,1)
    ((4, 5), (6, 1), (6, 1)),  # 2_l ⊗ 2_s ⊂ A ⊗ A
    ((9, 2), (1, 2), (9, 1)),  # χ_s ⊗ H₂
    ((9, 3), (1, 2), (9, 4)),  # χ_s ⊗ det ⊗ H₂ = χ_l ⊗ H₂
    ((8, 1), (4, 1), (9, 1)),  # R ⊗ H₂ = R ⊕ (8,1) ⊕ (8,3) ⊕ (16,1)
    ((8, 3), (4, 1), (9, 1)),
    ((8, 2), (4, 1), (9, 4)),  # R ⊗ det H₂ = det R ⊕ (8,2) ⊕ (8,4) ⊕ (16,1)
    ((8, 4), (4, 1), (9, 4)),
    ((12, 1), (4, 1), (16, 1)),
    ((2, 1), (4, 5), (4, 5)),  # (4,5) ⊗ (4,5) holds each 2-dim irrep once
    ((2, 2), (4, 5), (4, 5)),
    ((2, 3), (4, 5), (4, 5)),
    ((2, 4), (4, 5), (4, 5)),
)


######################## Elements and classes ############################


def _keys(matrices: np.ndarray) -> list[bytes]:
    """
    Exact, hashable keys of 4×4 matrices whose entries are multiples of ½ (all the elements of W(F₄)).

    Twice a matrix has integer entries: rounding them (np.rint) removes any floating-point noise, and
    the bytes of the integer array identify the matrix exactly.
    """
    integers = np.rint(2 * np.asarray(matrices)).astype(np.int64)
    return [m.tobytes() for m in integers.reshape(len(integers), -1)]


def elements(h4_vector: np.ndarray) -> np.ndarray:
    """
    The 1152 elements of W(F₄) as 4×4 matrices, in h4lat's order.

    Parameters
    ----------
    h4_vector : ndarray
        The 384 matrices of H(4) in its vector irrep (4,1), shape (384, 4, 4), in h4lat's order (as
        generated by cg_calculator or stored in the bundled data).

    Returns
    -------
    ndarray
        Shape (1152, 4, 4): element 384·j + i is h4_vector[i] · Aʲ (j = 0, 1, 2), with A = ``TRIALITY``.

    Raises
    ------
    ValueError
        If the matrices are not 1152 distinct elements: a check that A does not lie in H(4) and that
        h4_vector is the whole of H(4).
    """
    h4_vector = np.asarray(h4_vector, dtype=float).reshape(-1, 4, 4)
    powers = [np.linalg.matrix_power(TRIALITY, j) for j in range(3)]
    group = np.array([h @ power for power in powers for h in h4_vector])
    if len(group) != HONEYCOMB.order or len(set(_keys(group))) != HONEYCOMB.order:
        raise ValueError("H(4), H(4)·A and H(4)·A² are not 1152 distinct matrices: check the H(4) matrices")
    return group


def conjugacy_classes(group: np.ndarray) -> np.ndarray:
    """
    Conjugacy class of every element, the classes numbered in order of first appearance.

    The class of x is {g x g⁻¹ : g ∈ G}.  For orthogonal matrices g⁻¹ = gᵀ, so a whole class is
    obtained with one vectorised product over all the elements g.

    Parameters
    ----------
    group : ndarray
        All the elements, shape (n_ele, 4, 4), as returned by ``elements``.

    Returns
    -------
    ndarray of int
        Entry i is the class of element i: 0 for the identity (element 0), 1 for the class of the
        first element outside it, and so on.  This is the column order of ``HONEYCOMB.char_table``.
    """
    index = {key: i for i, key in enumerate(_keys(group))}
    classes = np.full(len(group), -1)
    n_classes = 0
    for i in range(len(group)):
        if classes[i] >= 0:
            continue
        # g @ x @ gᵀ for all g at once: einsum sums over the repeated matrix indices b and c.
        conjugates = np.einsum('gab,bc,gdc->gad', group, group[i], group)
        classes[[index[key] for key in _keys(conjugates)]] = n_classes
        n_classes += 1
    return classes


######################## Irreducible representations #####################


def _exact_float(x: float, max_denominator: int = 10**4, tol: float = 1e-9) -> float:
    """
    The float closest to the exact number that *x* approximates: a rational p/q or ±√(p/q).

    The matrices built by projection and Gram-Schmidt carry rounding errors of order 1e-16; this
    replaces every entry by the double nearest to its exact value (or, for ±√(p/q), the correctly
    rounded square root of the double nearest to p/q), so the result does not depend on the order of
    the floating-point operations.

    Raises
    ------
    ValueError
        If *x* is neither a rational nor ±√rational with denominator ≤ *max_denominator*.
    """
    if abs(x) < tol:
        return 0.0
    r = Fraction(x).limit_denominator(max_denominator)
    if abs(float(r) - x) < tol:
        return float(r)
    r2 = Fraction(x * x).limit_denominator(max_denominator)
    if abs(math.sqrt(r2) - abs(x)) < tol:
        return math.copysign(math.sqrt(float(r2)), x)
    raise ValueError(f"{x!r} is neither a rational number nor the square root of one")


def _exact(matrices: np.ndarray) -> np.ndarray:
    """Apply ``_exact_float`` to every entry of *matrices* (done once per distinct value)."""
    values, inverse = np.unique(matrices, return_inverse=True)
    exact = np.array([_exact_float(float(v)) for v in values])
    return exact[inverse].reshape(matrices.shape)


def _irrep_in_product(chi: np.ndarray, dim: int, product: np.ndarray) -> np.ndarray:
    """
    Matrices of the irrep with characters *chi* cut out of a representation that holds it once.

    The projector onto the irrep is P = (d/|G|) Σ_g χ(g) T(g) (characters are real here).  Its image
    is the d-dimensional subspace where the irrep lives: its columns, taken in order, are
    orthonormalised (Gram-Schmidt) until d vectors are found, which form the matrix B of a basis.
    The irrep is then D(g) = Bᵀ T(g) B, orthogonal because T(g) is.

    Parameters
    ----------
    chi : ndarray
        Character of the irrep on every element, shape (n_ele,).
    dim : int
        Dimension of the irrep.
    product : ndarray
        Matrices T(g) of the representation, shape (n_ele, n, n), orthogonal.

    Returns
    -------
    ndarray
        Shape (n_ele, dim, dim).
    """
    projector = dim / len(chi) * np.einsum('g,gab->ab', chi, product)
    basis = []
    for j in range(product.shape[1]):
        v = projector[:, j].copy()
        for b in basis:
            v -= (b @ v) * b
        norm = np.linalg.norm(v)
        if norm > 1e-9:  # a column already in the span of the previous ones is skipped
            basis.append(v / norm)
        if len(basis) == dim:
            break
    B = np.array(basis).T
    # Bᵀ T(g) B for every g
    return np.einsum('ai,gab,bj->gij', B, product, B)


def _tensor_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Kronecker product a(g) ⊗ b(g) for every element g: shape (n_ele, da·db, da·db)."""
    n = len(a)
    return np.einsum('gab,gcd->gacbd', a, b).reshape(n, a.shape[1] * b.shape[1], -1)


def irrep_matrices(group: np.ndarray, classes: np.ndarray | None = None) -> list[np.ndarray]:
    """
    Matrices of all the elements of W(F₄) in each of its 25 irreps.

    Parameters
    ----------
    group : ndarray
        All the elements, shape (1152, 4, 4), as returned by ``elements``.
    classes : ndarray or None
        Their conjugacy classes, as returned by ``conjugacy_classes`` (computed if None).

    Returns
    -------
    list of ndarray
        One array of shape (1152, d, d) per irrep, in the order of ``HONEYCOMB.rep_label_list``.
        Entry [g] is the real orthogonal matrix of element g.

    Raises
    ------
    ValueError
        If the classes or the matrices do not match ``HONEYCOMB`` (character table, class sizes):
        a consistency check of the construction.
    """
    group = np.asarray(group, dtype=float)
    if classes is None:
        classes = conjugacy_classes(group)
    if np.bincount(classes).tolist() != list(HONEYCOMB.class_orders):
        raise ValueError("the conjugacy classes do not match those of the honeycomb character table")
    # characters of every irrep on every element, shape (25, 1152)
    chi = HONEYCOMB.char_table[:, classes].astype(float)
    index = HONEYCOMB.irrep_index

    irreps: dict[tuple[int, int], np.ndarray] = {}
    # 1-dim irreps: the matrix of g is its character
    for label in HONEYCOMB.rep_label_list:
        if label[0] == 1:
            irreps[label] = chi[index[label]][:, None, None].copy()
    # the defining representation R, det ⊗ R and the antisymmetric tensors Λ²R
    det = np.rint(np.linalg.det(group))
    irreps[(4, 1)] = group.copy()
    irreps[(4, 4)] = group * det[:, None, None]
    # Λ²R on e_a ∧ e_b: g(e_c ∧ e_e) = Σ_{a<b} (g_ac g_be − g_ae g_bc) e_a ∧ e_b
    irreps[(6, 1)] = np.array(
        [[[g[a, c] * g[b, e] - g[a, e] * g[b, c] for (c, e) in _PAIRS] for (a, b) in _PAIRS] for g in group]
    )
    # the others, each cut out of a tensor product of two irreps built before it
    for target, a, b in _RECIPE:
        product = _tensor_product(irreps[a], irreps[b])
        multiplicity = chi[index[target]] @ np.trace(product, axis1=1, axis2=2) / len(group)
        if round(multiplicity) != 1:
            raise ValueError(f"{target} appears {multiplicity} times in {a} ⊗ {b}, not once")
        irreps[target] = _irrep_in_product(chi[index[target]], target[0], product)

    matrices = []
    for label in HONEYCOMB.rep_label_list:
        mats = _exact(irreps[label])
        if not np.allclose(np.trace(mats, axis1=1, axis2=2), chi[index[label]], atol=1e-9):
            raise ValueError(f"the matrices built for {label} do not have its characters")
        matrices.append(mats)
    return matrices
