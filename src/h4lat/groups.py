######################################################
## groups.py  (h4lat package module)                ##
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
The lattice symmetry groups known to h4lat, and their static data.

h4lat computes Clebsch-Gordan coefficients and lattice operators for two point groups:

* ``'H4'``: the hypercubic group H(4) = W(B₄) of the ordinary 4-dimensional hypercubic lattice,
  of order 384, with 20 irreducible representations (irreps) τ^(k)_l (Baake et al. 1982);
* ``'honeycomb'``: the group W(F₄) of the 16-cell honeycomb lattice, of order 1152, with 25 irreps
  ϱ^(k)_l (labels of Katz and Nogradi, arXiv:2611.21543).  It contains H(4) as a subgroup of index 3.

In both groups an irrep is labelled by a pair (k, l), where k is its dimension and l tells apart the
inequivalent irreps of the same dimension.  The same pair can denote different irreps in the two
groups (e.g. (4, 2)): an irrep label only has a meaning together with its group.

Each group is a ``LatticeGroup`` object holding the data that do not depend on a basis: the irrep
labels, the character table, the sizes of the conjugacy classes and the names of the data folders.
The explicit matrices of the group elements are built (and stored) by ``cg_calculator``; for the
honeycomb group the construction lives in the module ``honeycomb``.

Every function of h4lat that depends on the group takes an argument ``group``: ``None`` (the default)
or ``'H4'`` selects H(4), as in the versions of h4lat that knew only that group, and ``'honeycomb'``
selects W(F₄).  ``get_group`` turns any accepted value into the ``LatticeGroup`` object.
"""

import functools
from dataclasses import dataclass

import numpy as np

######################## The LatticeGroup class ##########################


@dataclass(frozen=True, eq=False)
class LatticeGroup:
    """
    Static data of a finite lattice symmetry group: irreps, character table and data folders.

    Instances are created once, in this module (``H4`` and ``HONEYCOMB``), and are compared by
    identity: use ``get_group`` to obtain them.  The dataclass is frozen, so its fields cannot be
    reassigned; the derived quantities (``irrep_index`` and so on) are computed on first access and
    cached.

    Attributes
    ----------
    name : str
        Identifier of the group, as accepted by ``get_group`` and stored in ``Operator.group``:
        ``'H4'`` or ``'honeycomb'``.
    title : str
        Name of the group for printing, e.g. ``'H(4)'`` or ``'W(F₄)'``.
    description : str
        One-line description of the lattice the group belongs to.
    order : int
        Number of elements of the group.
    rep_label_list : tuple of tuple[int, int]
        Labels (k, l) of the irreps, in the order used everywhere in h4lat: rows of the character
        table, entries of ``get_multiplicities``, keys of ``cg_calc.cg_dict`` and the order of the
        operators in the database.
    char_table : ndarray
        Character table, shape (n_rep, n_rep), read-only: entry [r, c] is the character of the
        irrep ``rep_label_list[r]`` on the conjugacy class c.
    class_orders : tuple of int
        Number of elements of each conjugacy class, in the column order of ``char_table``.
    latex_symbol : str
        LaTeX symbol of the irreps: ``\\tau`` for H(4), ``\\varrho`` for the honeycomb group.
    ele_folder : str
        Name of the folder holding the matrices of all the group elements in every irrep (the
        bundled one inside ``h4lat/data``, and the local one written by ``cg_calc``).
    ele_layout : str
        How that folder is organised: ``'element'`` (one sub-folder per irrep, one file per element,
        the historical layout of H(4)) or ``'irrep'`` (one file per irrep holding the matrices of all
        the elements).
    cg_database_folder : str
        Name of the folder of the CG-coefficient database (bundled and local).
    operator_database_folder : str
        Name of the folder of the bundled operator database.
    operator_file_suffix : str
        Text appended to the file names of the saved operators of this group (empty for H(4), so
        that the files saved before the group was recorded are read as H(4) operators).
    """

    name: str
    title: str
    description: str
    order: int
    rep_label_list: tuple
    char_table: np.ndarray
    class_orders: tuple
    latex_symbol: str
    ele_folder: str
    ele_layout: str
    cg_database_folder: str
    operator_database_folder: str
    operator_file_suffix: str

    def __post_init__(self) -> None:
        """Check the consistency of the data and make the character table read-only."""
        n = len(self.rep_label_list)
        if self.char_table.shape != (n, n) or len(self.class_orders) != n:
            raise ValueError(f"{self.name}: the character table must be {n}x{n}, with {n} class sizes")
        if sum(self.class_orders) != self.order:
            raise ValueError(f"{self.name}: the class sizes must add up to the order {self.order}")
        # The dimension of an irrep is its character on the identity (first column), and the
        # labelling convention (k, l) puts it in k.
        if any(int(self.char_table[r, 0]) != k for r, (k, _) in enumerate(self.rep_label_list)):
            raise ValueError(f"{self.name}: the first column of the character table must be the dimensions")
        # A frozen dataclass forbids reassigning a field, but the array itself could still be
        # changed in place: setting it read-only protects the data shared by the whole package.
        self.char_table.setflags(write=False)

    def __repr__(self) -> str:
        """Short representation, e.g. ``LatticeGroup('honeycomb': W(F₄), order 1152, 25 irreps)``."""
        return f"LatticeGroup({self.name!r}: {self.title}, order {self.order}, {self.n_rep} irreps)"

    def __reduce__(self):
        """Pickle a group by name, so that unpickling returns the very same object (see get_group)."""
        return (get_group, (self.name,))

    # --- derived quantities (cached on first access) ---------------------------------------------

    @functools.cached_property
    def n_rep(self) -> int:
        """Number of irreps (= number of conjugacy classes)."""
        return len(self.rep_label_list)

    @functools.cached_property
    def rep_dim_list(self) -> tuple[int, ...]:
        """Dimensions of the irreps, in the order of ``rep_label_list``."""
        return tuple(k for k, _ in self.rep_label_list)

    @functools.cached_property
    def irrep_index(self) -> dict[tuple[int, int], int]:
        """Position of each irrep label in ``rep_label_list``."""
        return {label: i for i, label in enumerate(self.rep_label_list)}

    @functools.cached_property
    def irrep_dim(self) -> dict[tuple[int, int], int]:
        """Dimension of each irrep, keyed by its label."""
        return dict(zip(self.rep_label_list, self.rep_dim_list))

    @functools.cached_property
    def rep_latex_names(self) -> tuple[str, ...]:
        """LaTeX names of the irreps, e.g. ``\\tau^{\\left(4\\right)}_1``, in the order of ``rep_label_list``."""
        return tuple(f"{self.latex_symbol}^{{\\left({k}\\right)}}_{l}" for k, l in self.rep_label_list)

    @functools.cached_property
    def irrep_texname(self) -> dict[tuple[int, int], str]:
        """LaTeX name of each irrep, keyed by its label."""
        return dict(zip(self.rep_label_list, self.rep_latex_names))

    # --- irreps and tensor products ------------------------------------------------------------

    def check_irreps(self, irreps) -> tuple[tuple[int, int], ...]:
        """
        Return *irreps* as a tuple of (k, l) tuples, after checking that they are irreps of this group.

        Raises
        ------
        ValueError
            If one of them is not the label of an irrep of the group.
        """
        irreps = tuple(tuple(int(x) for x in irrep) for irrep in irreps)
        for irrep in irreps:
            if irrep not in self.irrep_index:
                raise ValueError(
                    f"{irrep} is not an irrep of the {self.title} group ('{self.name}'): "
                    f"its irreps are {', '.join(str(label) for label in self.rep_label_list)}"
                )
        return irreps

    def multiplicities(self, *irreps: tuple[int, int]) -> list[int]:
        """
        Multiplicity of each irrep in the tensor product of *irreps*.

        The multiplicity of the irrep r in a representation with character χ is
        (1/|G|) Σ_c |c| χ_r(c) χ(c), summed over the conjugacy classes c of size |c| (all the
        characters of both groups are real).  The character of a tensor product is the product of
        the characters of its factors.  The sum is done with integers, so it is exact.

        Parameters
        ----------
        *irreps : tuple[int, int]
            Factors of the tensor product, e.g. (4, 1), (4, 1).

        Returns
        -------
        list[int]
            Entry r is the multiplicity of the irrep ``rep_label_list[r]``.
        """
        irreps = self.check_irreps(irreps)
        mult_list = []
        for ir in range(self.n_rep):
            total = 0
            for ic, class_order in enumerate(self.class_orders):
                term = int(class_order) * int(self.char_table[ir, ic])
                for irrep in irreps:
                    term *= int(self.char_table[self.irrep_index[irrep], ic])
                total += term
            mul, remainder = divmod(total, self.order)
            if remainder:  # impossible for a valid character table
                raise ArithmeticError(f"{self.name}: non-integer multiplicity {total}/{self.order}")
            mult_list.append(mul)
        return mult_list


######################## H(4): the hypercubic group ######################

# H(4) = Z₂⁴ ⋊ S(4): permutations of the 4 axes combined with reflections of any of them.  Labels,
# character table and class sizes follow Baake, Gemünden and Oedingen, J. Math. Phys. 23, 944 (1982).
# The explicit matrices are generated from the generators alpha_list / beta_list / gamma_list of
# cg_calculator.

_H4_LABELS = (
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 1),
    (2, 2),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
    (4, 1),
    (4, 2),
    (4, 3),
    (4, 4),
    (6, 1),
    (6, 2),
    (6, 3),
    (6, 4),
    (8, 1),
    (8, 2),
)

_H4_CHAR_TABLE = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1],
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1],
        [1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1],
        [2, 2, 0, 0, 2, -1, 0, 0, 2, -1, 0, 0, 2, 2, 0, -1, -1, 2, 0, 2],
        [2, -2, 0, 0, 2, 1, 0, 0, -2, -1, 0, 0, 2, -2, 0, -1, 1, 2, 0, 2],
        [3, 3, 1, 1, 3, 0, 1, 1, 3, 0, -1, -1, -1, -1, 1, 0, 0, -1, 1, 3],
        [3, -3, -1, 1, 3, 0, 1, -1, -3, 0, 1, -1, -1, 1, -1, 0, 0, -1, 1, 3],
        [3, 3, -1, -1, 3, 0, -1, -1, 3, 0, 1, 1, -1, -1, -1, 0, 0, -1, -1, 3],
        [3, -3, 1, -1, 3, 0, -1, 1, -3, 0, -1, 1, -1, 1, 1, 0, 0, -1, -1, 3],
        [4, 2, 2, 2, 0, 1, 0, 0, -2, 1, 0, 0, 0, 0, -2, -1, -1, 0, -2, -4],
        [4, -2, -2, 2, 0, -1, 0, 0, 2, 1, 0, 0, 0, 0, 2, -1, 1, 0, -2, -4],
        [4, 2, -2, -2, 0, 1, 0, 0, -2, 1, 0, 0, 0, 0, 2, -1, -1, 0, 2, -4],
        [4, -2, 2, -2, 0, -1, 0, 0, 2, 1, 0, 0, 0, 0, -2, -1, 1, 0, 2, -4],
        [6, 0, 2, 0, -2, 0, 0, -2, 0, 0, 0, 0, 2, 0, 2, 0, 0, -2, 0, 6],
        [6, 0, -2, 0, -2, 0, 0, 2, 0, 0, 0, 0, 2, 0, -2, 0, 0, -2, 0, 6],
        [6, 0, 0, 2, -2, 0, -2, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 2, 2, 6],
        [6, 0, 0, -2, -2, 0, 2, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 2, -2, 6],
        [8, 4, 0, 0, 0, -1, 0, 0, -4, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, -8],
        [8, -4, 0, 0, 0, 1, 0, 0, 4, -1, 0, 0, 0, 0, 0, 1, -1, 0, 0, -8],
    ],
    dtype=int,
)

_H4_CLASS_ORDERS = (1, 4, 12, 12, 6, 32, 24, 24, 4, 32, 48, 48, 12, 24, 12, 32, 32, 12, 12, 1)

H4 = LatticeGroup(
    name='H4',
    title='H(4)',
    description='hypercubic group H(4) = W(B₄) of the 4-dimensional hypercubic lattice',
    order=384,
    rep_label_list=_H4_LABELS,
    char_table=_H4_CHAR_TABLE,
    class_orders=_H4_CLASS_ORDERS,
    latex_symbol=r'\tau',
    ele_folder='h4_ele',
    ele_layout='element',
    cg_database_folder='cg_database',
    operator_database_folder='operator_database',
    operator_file_suffix='',
)
"""The hypercubic group H(4), of order 384 with 20 irreps (the default group of h4lat)."""


######################## W(F₄): the 16-cell honeycomb group ##############

# The point group of the 16-cell honeycomb lattice (Z⁴ together with Z⁴ + (½, ½, ½, ½)) is the Weyl
# group W(F₄), generated by the reflections along the 24 short roots ±e_μ, (±½, ±½, ±½, ±½) and the
# 24 long roots ±e_μ ± e_ν.  The irrep labels ϱ^(k)_l and their order are those of eq. (5.2) of
# S. Katz and D. Nogradi, "Hadron structure on the 16-cell honeycomb", arXiv:2611.21543:
#   (1,1) = 1, (1,2) = χ_s, (1,3) = χ_l, (1,4) = det,
#   (2,1) = 2_l, (2,2) = det⊗2_l, (2,3) = 2_s, (2,4) = det⊗2_s,
#   (4,1) = R, (4,2) = χ_s⊗R, (4,3) = χ_l⊗R, (4,4) = det⊗R, (4,5) = 2_l⊗2_s,
#   (6,1) = A = Λ²R, (6,2) = χ_l⊗A,
#   (8,1) = 2_l⊗R, (8,2) = det⊗2_l⊗R, (8,3) = 2_s⊗R, (8,4) = det⊗2_s⊗R,
#   (9,1) = H₂, (9,2) = χ_s⊗H₂, (9,3) = χ_l⊗H₂, (9,4) = det⊗H₂,
#   (12,1) = 2_l⊗A, (16,1) = 2_l⊗2_s⊗R,
# where R is the defining (vector) representation, H₂ the symmetric traceless rank-2 tensors, χ_s
# (χ_l) the sign (−1)^(number of short (long) reflections), and 2_s (2_l) the 2-dim irrep of S₃
# through which W(F₄) permutes the three orthonormal frames of short (long) roots.
#
# The character table was computed with GAP (The GAP Group, "GAP – Groups, Algorithms, and
# Programming", https://www.gap-system.org) and its rows were matched to the labels above by building
# each character from its definition.  tests/test_honeycomb.py checks it again from scratch: the
# orthogonality relations, the traces of the matrices built by h4lat, and the definitions above.
#
# The columns are the conjugacy classes in the order in which they first appear in the list of the
# 1152 elements built by h4lat (see the module honeycomb: H(4) in h4lat's order first, then the two
# cosets of the triality rotation); the first column is the identity.

_HONEYCOMB_LABELS = (
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (4, 1),
    (4, 2),
    (4, 3),
    (4, 4),
    (4, 5),
    (6, 1),
    (6, 2),
    (8, 1),
    (8, 2),
    (8, 3),
    (8, 4),
    (9, 1),
    (9, 2),
    (9, 3),
    (9, 4),
    (12, 1),
    (16, 1),
)

# fmt: off
_HONEYCOMB_CHAR_TABLE = np.array(
    [
        [ 1,  1,  1,  1,   1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        [ 1, -1,  1, -1,   1,  1, -1, -1,  1,  1, -1,  1, -1, -1,  1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1],
        [ 1,  1,  1,  1,   1, -1, -1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1],
        [ 1, -1,  1, -1,   1, -1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1, -1,  1,  1,  1, -1, -1,  1,  1,  1],
        [ 2,  2,  2,  2,   2,  0,  0,  2,  2,  0,  0,  0,  0,  0, -1, -1, -1, -1,  2,  2,  0,  0, -1, -1, -1],
        [ 2, -2,  2, -2,   2,  0,  0, -2,  2,  0,  0,  0,  0,  0, -1,  1,  1, -1,  2,  2,  0,  0, -1, -1, -1],
        [ 2,  0,  2,  0,   2,  2,  0,  0,  2,  2,  0,  2,  0,  0,  2,  0,  0,  2, -1, -1, -1, -1, -1, -1, -1],
        [ 2,  0,  2,  0,   2, -2,  0,  0,  2, -2,  0, -2,  0,  0,  2,  0,  0,  2, -1, -1,  1,  1, -1, -1, -1],
        [ 4,  2,  0, -2,  -4,  0,  0,  0,  0,  2,  0, -2,  2, -2,  1,  1, -1, -1,  1, -1,  1, -1,  0,  2, -2],
        [ 4, -2,  0,  2,  -4,  0,  0,  0,  0,  2,  0, -2, -2,  2,  1, -1,  1, -1,  1, -1,  1, -1,  0,  2, -2],
        [ 4,  2,  0, -2,  -4,  0,  0,  0,  0, -2,  0,  2, -2,  2,  1,  1, -1, -1,  1, -1, -1,  1,  0,  2, -2],
        [ 4, -2,  0,  2,  -4,  0,  0,  0,  0, -2,  0,  2,  2, -2,  1, -1,  1, -1,  1, -1, -1,  1,  0,  2, -2],
        [ 4,  0,  4,  0,   4,  0,  0,  0,  4,  0,  0,  0,  0,  0, -2,  0,  0, -2, -2, -2,  0,  0,  1,  1,  1],
        [ 6,  0, -2,  0,   6,  0,  0,  0,  2,  0, -2,  0,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0, -1,  3,  3],
        [ 6,  0, -2,  0,   6,  0,  0,  0,  2,  0,  2,  0, -2, -2,  0,  0,  0,  0,  0,  0,  0,  0, -1,  3,  3],
        [ 8,  4,  0, -4,  -8,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1,  1,  1,  2, -2,  0,  0,  0, -2,  2],
        [ 8, -4,  0,  4,  -8,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  1, -1,  1,  2, -2,  0,  0,  0, -2,  2],
        [ 8,  0,  0,  0,  -8,  0,  0,  0,  0,  4,  0, -4,  0,  0,  2,  0,  0, -2, -1,  1, -1,  1,  0, -2,  2],
        [ 8,  0,  0,  0,  -8,  0,  0,  0,  0, -4,  0,  4,  0,  0,  2,  0,  0, -2, -1,  1,  1, -1,  0, -2,  2],
        [ 9,  3,  1,  3,   9, -1, -1, -1, -3,  3,  1,  3,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 9, -3,  1, -3,   9, -1,  1,  1, -3,  3, -1,  3, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 9,  3,  1,  3,   9,  1,  1, -1, -3, -3, -1, -3, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 9, -3,  1, -3,   9,  1, -1,  1, -3, -3,  1, -3,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [12,  0, -4,  0,  12,  0,  0,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -3, -3],
        [16,  0,  0,  0, -16,  0,  0,  0,  0,  0,  0,  0,  0,  0, -2,  0,  0,  2, -2,  2,  0,  0,  0,  2, -2],
    ],
    dtype=int,
)

# sizes of the conjugacy classes, in the column order of the character table
_HONEYCOMB_CLASS_ORDERS = (1, 12, 18, 12, 1, 72, 144, 72, 12, 12, 72, 12, 36,
                           36, 32, 96, 96, 32, 32, 32, 96, 96, 96, 16, 16)
# fmt: on


HONEYCOMB = LatticeGroup(
    name='honeycomb',
    title='W(F₄)',
    description='group W(F₄) of the 16-cell honeycomb lattice (arXiv:2611.21543)',
    order=1152,
    rep_label_list=_HONEYCOMB_LABELS,
    char_table=_HONEYCOMB_CHAR_TABLE,
    class_orders=_HONEYCOMB_CLASS_ORDERS,
    latex_symbol=r'\varrho',
    ele_folder='honeycomb_ele',
    ele_layout='irrep',
    cg_database_folder='cg_database_honeycomb',
    operator_database_folder='operator_database_honeycomb',
    operator_file_suffix='_honeycomb',
)
"""The 16-cell honeycomb group W(F₄), of order 1152 with 25 irreps; it contains H(4) with index 3."""


######################## Looking up a group ##############################

AVAILABLE_GROUPS = ('H4', 'honeycomb')
"""Names of the groups known to h4lat, as accepted by ``get_group`` (and stored in ``Operator.group``)."""

_GROUPS = {'H4': H4, 'honeycomb': HONEYCOMB}

# Other names accepted by get_group, compared after _normalise_name: lower case, without spaces,
# brackets, hyphens and underscores, and with the subscript ₄ written as 4.
_ALIASES = {
    'h4': H4,
    'hypercubic': H4,
    'wb4': H4,
    'b4': H4,
    'honeycomb': HONEYCOMB,
    '16cellhoneycomb': HONEYCOMB,
    'wf4': HONEYCOMB,
    'f4': HONEYCOMB,
}


def _normalise_name(name: str) -> str:
    """Lower-case *name* and drop the characters that do not tell groups apart: 'W(F₄)' → 'wf4'."""
    name = name.strip().lower().replace('₄', '4')
    for char in ' ()-_':
        name = name.replace(char, '')
    return name


def get_group(group: 'str | LatticeGroup | None' = None) -> LatticeGroup:
    """
    Return the ``LatticeGroup`` object of a lattice symmetry group.

    Parameters
    ----------
    group : str, LatticeGroup or None
        ``None`` (default) or ``'H4'`` (also ``'H(4)'``, ``'hypercubic'``, ``'W(B4)'``) for the
        hypercubic group; ``'honeycomb'`` (also ``'W(F4)'``, ``'WF4'``, ``'F4'``) for the 16-cell
        honeycomb group.  Case does not matter.  A ``LatticeGroup`` is returned as it is.

    Returns
    -------
    LatticeGroup
        ``groups.H4`` or ``groups.HONEYCOMB``.

    Raises
    ------
    ValueError
        If *group* is not the name of a known group.

    Examples
    --------
    >>> get_group().title
    'H(4)'
    >>> get_group('W(F4)').order
    1152
    """
    if group is None:
        return H4
    if isinstance(group, LatticeGroup):
        return group
    if isinstance(group, str):
        found = _ALIASES.get(_normalise_name(group))
        if found is not None:
            return found
    raise ValueError(f"unknown lattice group {group!r}: the available groups are {', '.join(AVAILABLE_GROUPS)}")
