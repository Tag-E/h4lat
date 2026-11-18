######################################################
## cg_calculator.py  (h4lat package module)         ##
## originally cgh4_calculator.py                    ##
## created by Emilio Taggi - 2024/12/09             ##
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
Clebsch-Gordan coefficient calculator for the lattice symmetry groups of h4lat.

Two groups are supported (see the module groups): the hypercubic group H(4), the default, and the
group W(F₄) of the 16-cell honeycomb lattice, selected with ``group='honeycomb'``.

H(4) = Z₂⁴ ⋊ S(4) is the symmetry group of the 4-dimensional hypercubic
lattice, of order 384.  It has 20 irreducible representations labelled (k,l)
where k is the dimension and l distinguishes inequivalent irreps of the same
dimension; this labelling follows Baake et al. (1982) [J. Math. Phys. 23, 944].
W(F₄), of order 1152, has 25 irreps, labelled in the same way following Katz and
Nogradi (2026) [arXiv:2611.21543]; it contains H(4) as a subgroup.

CG coefficients are computed via the projection formula of Sakata (1974)
[J. Math. Phys. 15, 1702]: summing the tensor-product matrices against the
target irrep matrices over all group elements extracts the CG vectors as
linear forms in a matrix of free symbols A.  The sum is done numerically, on
the coefficients of these forms (see _projection), optionally in parallel with
numba.  The resulting free parameters (freedom within degenerate
subspaces) are fixed by CGmat_from_block, which picks one independent copy
of the irrep for each unit of multiplicity, each with a definite C-parity
(symmetric or antisymmetric under reversal of the derivative indices).
cg_calc does this directly on the numerical coefficients
(_linear_forms_from_coefficients, _cgmat_from_linear_forms): the symbolic
"raw" blocks are built only to be stored, if asked for (save_raw).

The module-level constants (rep_label_list, char_table, …) are those of H(4),
as in the versions of h4lat that knew only that group; those of any group are
attributes of its LatticeGroup object (groups.get_group).

Main entry point: cg_calc.
Module-level helpers: get_multiplicities, latex_print_multiplicities.
"""

######################## Library Imports ################################

import functools
import sys
import time
import warnings
from importlib.resources import files as _resource_files
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import sympy as sym
from tqdm import tqdm

from . import honeycomb as _honeycomb
from .groups import H4 as _H4
from .groups import HONEYCOMB as _HONEYCOMB
from .groups import LatticeGroup, get_group

try:
    from pylatex import Command, Document, Math, Matrix, Section
    from pylatex.utils import NoEscape

    _PYLATEX_AVAILABLE = True
except ImportError:
    _PYLATEX_AVAILABLE = False

try:
    from IPython.display import Math as MathDisplay
    from IPython.display import display

    _IPYTHON_AVAILABLE = True
except ImportError:
    _IPYTHON_AVAILABLE = False

# numba (optional, for cg_calc(parallel=True)) is only imported when it is used: see _numba_projection_kernel
_NUMBA_AVAILABLE = find_spec("numba") is not None


######################## Package Data Paths #############################


def _pkg_data(*parts: str) -> str:
    """Return the filesystem path to a file/dir inside h4lat/data/."""
    p = _resource_files('h4lat') / 'data'
    for part in parts:
        p = p / part
    return str(p)


# The bundled data of each group, keyed by group name and then by kind of data:
#   'ele'                matrices of all the group elements in every irrep,
#   'cg_database'        CG coefficients of the products of the operator database,
#   'operator_database'  the operator database.
# The folder names come from the LatticeGroup objects (h4_ele, cg_database, operator_database for
# H(4); honeycomb_ele, cg_database_honeycomb, operator_database_honeycomb for W(F₄)).  The paths are
# looked up in this dict when they are used (_bundled), so a test can replace one of them.
_BUNDLED_DATA = {
    group.name: {
        'ele': _pkg_data(group.ele_folder),
        'cg_database': _pkg_data(group.cg_database_folder),
        'operator_database': _pkg_data(group.operator_database_folder),
    }
    for group in (_H4, _HONEYCOMB)
}

# The H(4) entries under their historical names (OPERATOR_DATABASE of the package is the last one).
_BUNDLED_H4_ELE = _BUNDLED_DATA['H4']['ele']
_BUNDLED_CG_DATABASE = _BUNDLED_DATA['H4']['cg_database']
_BUNDLED_OPERATOR_DATABASE = _BUNDLED_DATA['H4']['operator_database']


def _bundled(group: LatticeGroup, kind: str) -> str:
    """Path of the bundled data of *kind* ('ele', 'cg_database' or 'operator_database') of *group*."""
    return _BUNDLED_DATA[group.name][kind]


# Data generated at runtime goes to folders in the working directory with the same names as the
# bundled ones (group.ele_folder, group.cg_database_folder), never into the (possibly read-only)
# package data.  To update the bundled data, use scripts/regenerate_data.py.  The H(4) names:
_LOCAL_H4_ELE = _H4.ele_folder
_LOCAL_CG_DATABASE = _H4.cg_database_folder


def _sorted_cg_files(folder: str) -> list[Path]:
    """Return the CG files '{irep}_{m}.npy' in *folder* sorted by (irep, m), not in filesystem order."""
    return sorted(Path(folder).glob('*.npy'), key=lambda f: tuple(int(x) for x in f.stem.split('_')))


######################## Global Variables ###############################

## specifics of the H(4) irreps ##

# The tables of H(4) are defined, with those of the honeycomb group, in the module groups.  They are
# also module-level names here (and in the package namespace), where the code and the users of h4lat
# have always found them: lists and dicts as before, with the same content as the H4 group object.

n_ele_s4 = 4 * 3 * 2  # order of S(4), the permutations of the 4 axes
n_ele_h4 = _H4.order  # 2**4 * 4 * 3 * 2 = 384
n_rep = _H4.n_rep  # 20
rep_dim_list = list(_H4.rep_dim_list)
fund_index = _H4.irrep_index[(4, 1)]  # position of the vector irrep (4,1): 10
rep_label_list = list(_H4.rep_label_list)
rep_latex_names = list(_H4.rep_latex_names)
irrep_index = dict(_H4.irrep_index)
irrep_dim = dict(_H4.irrep_dim)
irrep_texname = dict(_H4.irrep_texname)
char_table = _H4.char_table  # read-only array
class_orders = list(_H4.class_orders)


## matrix representations for the generators of H(4) ##

# H(4) = Z₂⁴ ⋊ S(4): the reflections Z₂⁴ form the normal subgroup (a permutation conjugates a
# reflection into another reflection).  S(4) is generated by α (a transposition in the
# fundamental representation) and β (a cyclic 4-rotation).  Z₂⁴ is generated
# by four independent reflections γ₁,…,γ₄, one per lattice axis.
# The three lists below give the matrix of each generator in all 20 irreps,
# ordered to match rep_label_list.  In every irrep they must satisfy the S(4)
# relations α² = β⁴ = (αβ)³ = 1 (checked in tests/test_cg_correctness.py).
# The 6-dim irreps act on pairs of axes in the order 12, 13, 23, 14, 24, 34
# (the order assumed by cg_remapping_T): (6,1) on e_i ∧ e_j as ∧²(4,1), (6,3) on
# the symmetric e_i e_j (i ≠ j), (6,2) = (6,1) ⊗ (1,2) and (6,4) = (6,3) ⊗ (1,4).

alpha_list = []

a = np.array([1], dtype=float)
alpha_list.append(a)
a = np.array([1], dtype=float)
alpha_list.append(a)
a = np.array([-1], dtype=float)
alpha_list.append(a)
a = np.array([-1], dtype=float)
alpha_list.append(a)

a = np.array([[1, 0], [0, -1]], dtype=float)
alpha_list.append(a)
# (2,2) = (2,1) ⊗ (1,2) has the same α as (2,1).  Baake et al. (1982) give α = 1 here,
# which violates (αβ)³ = 1 and so does not define a representation (see the wiki page "The hypercubic group",
# https://github.com/Tag-E/h4lat/wiki/The-hypercubic-group#generators).
a = np.array([[1, 0], [0, -1]], dtype=float)
alpha_list.append(a)

a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
alpha_list.append(a)
a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
alpha_list.append(a)
a = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
alpha_list.append(a)
a = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
alpha_list.append(a)

a = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
alpha_list.append(a)
a = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
alpha_list.append(a)
a = np.array([[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=float)
alpha_list.append(a)
a = np.array([[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=float)
alpha_list.append(a)

a = np.array(
    [
        [-1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1],
    ],
    dtype=float,
)
alpha_list.append(a)
a = np.array(
    [
        [-1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1],
    ],
    dtype=float,
)
alpha_list.append(a)
a = np.array(
    [
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1],
    ],
    dtype=float,
)
alpha_list.append(a)
a = np.array(
    [
        [-1, 0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0],
        [0, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 0],
        [0, 0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0, -1],
    ],
    dtype=float,
)
alpha_list.append(a)

a = np.array(
    [
        [0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, 0, 0],
        [0, 0, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, -1],
    ],
    dtype=float,
)
alpha_list.append(a)
a = np.array(
    [
        [0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, 0, 0],
        [0, 0, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, -1],
    ],
    dtype=float,
)
alpha_list.append(a)


beta_list = []

s2 = np.sqrt(2)
s3 = np.sqrt(3)
s6 = np.sqrt(6)
s8 = np.sqrt(8)

b = np.array([1], dtype=float)
beta_list.append(b)
b = np.array([1], dtype=float)
beta_list.append(b)
b = np.array([-1], dtype=float)
beta_list.append(b)
b = np.array([-1], dtype=float)
beta_list.append(b)

b = np.array([[-1 / 2, -s3 / 2], [-s3 / 2, 1 / 2]], dtype=float)
beta_list.append(b)
b = np.array([[-1 / 2, -s3 / 2], [-s3 / 2, 1 / 2]], dtype=float)
beta_list.append(b)

b = np.array([[-1 / 3, s8 / 3, 0], [-s2 / 3, -1 / 6, s3 / 2], [-s6 / 3, -s3 / 6, -1 / 2]], dtype=float)
beta_list.append(b)
b = np.array([[-1 / 3, s8 / 3, 0], [-s2 / 3, -1 / 6, s3 / 2], [-s6 / 3, -s3 / 6, -1 / 2]], dtype=float)
beta_list.append(b)
b = np.array([[1 / 3, -s8 / 3, 0], [s2 / 3, 1 / 6, -s3 / 2], [s6 / 3, s3 / 6, 1 / 2]], dtype=float)
beta_list.append(b)
b = np.array([[1 / 3, -s8 / 3, 0], [s2 / 3, 1 / 6, -s3 / 2], [s6 / 3, s3 / 6, 1 / 2]], dtype=float)
beta_list.append(b)

b = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
beta_list.append(b)
b = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
beta_list.append(b)
b = np.array([[0, 0, 0, -1], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0]], dtype=float)
beta_list.append(b)
b = np.array([[0, 0, 0, -1], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0]], dtype=float)
beta_list.append(b)

b = np.array(
    [
        [0, 0, 0, -1, 0, 0],
        [0, 0, 0, 0, -1, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -1],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ],
    dtype=float,
)
beta_list.append(b)
b = np.array(
    [
        [0, 0, 0, -1, 0, 0],
        [0, 0, 0, 0, -1, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -1],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ],
    dtype=float,
)
beta_list.append(b)
b = np.array(
    [
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ],
    dtype=float,
)
beta_list.append(b)
b = np.array(
    [
        [0, 0, 0, -1, 0, 0],
        [0, 0, 0, 0, -1, 0],
        [-1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -1],
        [0, -1, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0],
    ],
    dtype=float,
)
beta_list.append(b)

b = np.array(
    [
        [0, 0, 0, -1 / 2, 0, 0, 0, -s3 / 2],
        [-1 / 2, 0, 0, 0, -s3 / 2, 0, 0, 0],
        [0, -1 / 2, 0, 0, 0, -s3 / 2, 0, 0],
        [0, 0, -1 / 2, 0, 0, 0, -s3 / 2, 0],
        [0, 0, 0, -s3 / 2, 0, 0, 0, 1 / 2],
        [-s3 / 2, 0, 0, 0, 1 / 2, 0, 0, 0],
        [0, -s3 / 2, 0, 0, 0, 1 / 2, 0, 0],
        [0, 0, -s3 / 2, 0, 0, 0, 1 / 2, 0],
    ],
    dtype=float,
)
beta_list.append(b)
b = np.array(
    [
        [0, 0, 0, -1 / 2, 0, 0, 0, -s3 / 2],
        [-1 / 2, 0, 0, 0, -s3 / 2, 0, 0, 0],
        [0, -1 / 2, 0, 0, 0, -s3 / 2, 0, 0],
        [0, 0, -1 / 2, 0, 0, 0, -s3 / 2, 0],
        [0, 0, 0, -s3 / 2, 0, 0, 0, 1 / 2],
        [-s3 / 2, 0, 0, 0, 1 / 2, 0, 0, 0],
        [0, -s3 / 2, 0, 0, 0, 1 / 2, 0, 0],
        [0, 0, -s3 / 2, 0, 0, 0, 1 / 2, 0],
    ],
    dtype=float,
)
beta_list.append(b)


gamma_list = []

g = np.array([1], dtype=float)
gamma_list.append(g)
g = np.array([-1], dtype=float)
gamma_list.append(g)
g = np.array([1], dtype=float)
gamma_list.append(g)
g = np.array([-1], dtype=float)
gamma_list.append(g)

g = np.diag([1, 1])
gamma_list.append(g)
g = np.diag([-1, -1])
gamma_list.append(g)

g = np.diag([1, 1, 1])
gamma_list.append(g)
g = np.diag([-1, -1, -1])
gamma_list.append(g)
g = np.diag([1, 1, 1])
gamma_list.append(g)
g = np.diag([-1, -1, -1])
gamma_list.append(g)

g = np.diag([-1, 1, 1, 1])
gamma_list.append(g)
g = np.diag([1, -1, -1, -1])
gamma_list.append(g)
g = np.diag([-1, 1, 1, 1])
gamma_list.append(g)
g = np.diag([1, -1, -1, -1])
gamma_list.append(g)

g = np.diag([-1, -1, 1, -1, 1, 1])
gamma_list.append(g)
g = np.diag([1, 1, -1, 1, -1, -1])
gamma_list.append(g)
g = np.diag([-1, -1, 1, -1, 1, 1])
gamma_list.append(g)
g = np.diag([1, 1, -1, 1, -1, -1])
gamma_list.append(g)

g = np.diag([-1, 1, 1, 1, -1, 1, 1, 1])
gamma_list.append(g)
g = np.diag([1, -1, -1, -1, 1, -1, -1, -1])
gamma_list.append(g)


beta_inv_list = [np.linalg.inv(beta) if np.shape(beta)[0] > 1 else np.array(beta) for beta in beta_list]

# γ₂, γ₃, γ₄ are derived from γ₁ by conjugation with β:
#   γⱼ = β^{j-1} γ₁ β^{-(j-1)},  j = 2, 3, 4,
# which reflects axis j: β^{-(j-1)} maps axis j onto axis 1 (β maps axis 1 → 2 → 3 → 4 → 1),
# γ₁ reflects it, and β^{j-1} maps it back.  Since β⁴ = 1, β^{j-1} = β^{-(5-j)}, which is how the
# loop below writes it: γ₂ = β⁻³ γ₁ β³, γ₃ = β⁻² γ₁ β², γ₄ = β⁻¹ γ₁ β.
# This ties the four Z₂ reflections to the S(4) action, consistent with the
# semidirect-product structure H(4) = Z₂⁴ ⋊ S(4).
gamma1_list = gamma_list
gamma2_list = []
gamma3_list = []
gamma4_list = []

for ir in range(n_rep):
    b = np.reshape(beta_list[ir], (rep_dim_list[ir], rep_dim_list[ir]))
    binv = np.reshape(beta_inv_list[ir], (rep_dim_list[ir], rep_dim_list[ir]))
    g = np.reshape(gamma_list[ir], (rep_dim_list[ir], rep_dim_list[ir]))
    gamma2_list.append(binv @ binv @ binv @ g @ b @ b @ b)
    gamma3_list.append(binv @ binv @ g @ b @ b)
    gamma4_list.append(binv @ g @ b)


######################## Matrices of the group elements ###############################


def _generate_h4_matrices() -> list[list[np.ndarray]]:
    """
    Matrices of all the 384 elements of H(4) in every irrep, built from the generators.

    Coset enumeration for S(4): all the words α^i β^k (i ∈ {0,1}, k ∈ {0,1,2,3}) are applied to every
    element found so far, until all 24 = 4! elements are generated.  Duplicates are detected in the
    fundamental irrep (4,1) only, and the same word is applied in every irrep at once.  Then S(4) is
    extended to H(4) = Z₂⁴ ⋊ S(4): every element of S(4) is multiplied by the 2⁴ = 16 words
    γ₁^{a₁} γ₂^{a₂} γ₃^{a₃} γ₄^{a₄} of the reflections, giving 24 × 16 = 384 elements.

    The loops and their order are those used since the first version of h4lat, so the matrices, and
    the files written from them, are the same to the last bit.

    Returns
    -------
    list of list of ndarray
        Entry [ir][i] is the matrix of the i-th element in the irrep rep_label_list[ir].  The
        matrices of the 1-dim irreps have shape (1,) or (1, 1) (as in the bundled files).
    """
    s4_mat = [[np.eye(d)] for d in rep_dim_list]
    start = 0
    while len(s4_mat[0]) != n_ele_s4:
        for i in range(2):
            for k in range(4):
                res = s4_mat[fund_index][start]
                for _ in range(i):
                    res = res @ alpha_list[fund_index]
                for _ in range(k):
                    res = res @ beta_list[fund_index]

                duplicate = 0
                for known in s4_mat[fund_index]:
                    if (known == res).all():
                        duplicate = 1
                        break

                if duplicate == 0:
                    for ir, rep in enumerate(s4_mat):
                        res = rep[start]
                        res = np.reshape(res, (rep_dim_list[ir], rep_dim_list[ir]))
                        for _ in range(i):
                            res = res @ alpha_list[ir]
                            res = np.reshape(res, (rep_dim_list[ir], rep_dim_list[ir]))
                        for _ in range(k):
                            res = res @ beta_list[ir]
                            res = np.reshape(res, (rep_dim_list[ir], rep_dim_list[ir]))
                        s4_mat[ir].append(np.reshape(res, (rep_dim_list[ir], rep_dim_list[ir])))
        start += 1

    h4_mat = [[] for _ in range(n_rep)]
    for ir in range(n_rep):
        for s4_ele in s4_mat[ir]:
            for a1 in range(2):
                for a2 in range(2):
                    for a3 in range(2):
                        for a4 in range(2):
                            new_ele = s4_ele
                            for _ in range(a1):
                                new_ele = new_ele @ gamma1_list[ir]
                            for _ in range(a2):
                                new_ele = new_ele @ gamma2_list[ir]
                            for _ in range(a3):
                                new_ele = new_ele @ gamma3_list[ir]
                            for _ in range(a4):
                                new_ele = new_ele @ gamma4_list[ir]
                            h4_mat[ir].append(new_ele)
    return h4_mat


def _stack(matrices, dim: int) -> np.ndarray:
    """The matrices of all the elements in one irrep of dimension *dim*, as one (n_ele, dim, dim) float array."""
    return np.array([np.reshape(ele, (dim, dim)) for ele in matrices], dtype=float)


def _generate_honeycomb_matrices() -> list[np.ndarray]:
    """
    Matrices of all the 1152 elements of the honeycomb group W(F₄) in every irrep (see the module honeycomb).

    The group is built from the H(4) matrices generated by _generate_h4_matrices (not from stored
    data), so the result depends only on the code.

    Returns
    -------
    list of ndarray
        One (1152, d, d) array per irrep, in the order of HONEYCOMB.rep_label_list.
    """
    vector = _stack(_generate_h4_matrices()[fund_index], 4)  # H(4) in its vector irrep (4,1)
    return _honeycomb.irrep_matrices(_honeycomb.elements(vector))


def _generate_group_matrices(group: LatticeGroup) -> list:
    """
    Matrices of all the elements of *group* in every irrep, generated by the code.

    Returns, in the order of group.rep_label_list, a list of lists of matrices for H(4) (as
    _generate_h4_matrices) and a list of (n_ele, d, d) arrays for the honeycomb group.
    """
    if group is _H4:
        return _generate_h4_matrices()
    if group is _HONEYCOMB:
        return _generate_honeycomb_matrices()
    raise ValueError(f"no generator of the matrices of {group!r}")


def _save_group_matrices(group: LatticeGroup, folder: str, matrices: list) -> None:
    """
    Write the matrices of all the elements of *group* (as _generate_group_matrices) to *folder*.

    The layout of the folder is group.ele_layout: for 'element' (H(4)) one sub-folder per irrep,
    named after its label (e.g. '(4, 1)'), with one file per element '0.npy', '1.npy', …; for 'irrep'
    (the honeycomb group) one file per irrep, e.g. '(4, 1).npy', holding the (n_ele, d, d) array.
    """
    for label, mats in zip(group.rep_label_list, matrices):
        if group.ele_layout == 'element':
            rep_folder = f"{folder}/{label}"
            Path(rep_folder).mkdir(parents=True, exist_ok=True)
            for i, ele in enumerate(mats):
                with open(f'{rep_folder}/{i}.npy', 'wb') as f:
                    np.save(f, ele)
        else:
            Path(folder).mkdir(parents=True, exist_ok=True)
            with open(f'{folder}/{label}.npy', 'wb') as f:
                np.save(f, np.asarray(mats, dtype=float))
    # The files of this folder have changed, so any copy of them that _load_group_matrices has
    # cached is stale (functools.cache can only forget all its entries at once).
    _load_group_matrices.cache_clear()


@functools.cache
def _load_group_matrices(folder: str, group_name: str) -> tuple[np.ndarray, ...]:
    """
    Read the matrices of all the elements of a group, in every irrep, from *folder*.

    The folder has the layout written by _save_group_matrices: for H(4), 20 × 384 = 7680 small files
    '(k, l)/{i}.npy'; for the honeycomb group, 25 files '(k, l).npy'.  functools.cache stores the
    result of each call, keyed by (*folder*, *group_name*): a later call with the same arguments
    returns the same objects without reading the files again.  cg_calc therefore passes an absolute
    path (the same relative folder seen from two working directories is two different folders), and
    _save_group_matrices calls _load_group_matrices.cache_clear() after writing new matrices.

    The arrays are shared by every cg_calc that uses the folder: they are made read-only, so that
    changing one in place raises an error instead of silently changing it for all the others.  The
    container is a tuple for the same reason.

    Parameters
    ----------
    folder : str
        Absolute path of a folder of group matrices (h4_ele or honeycomb_ele).
    group_name : str
        Name of the group, 'H4' or 'honeycomb'.

    Returns
    -------
    tuple of ndarray
        Entry [ir] is the (n_ele, d, d) array of the matrices of all the elements in the irrep
        rep_label_list[ir] of the group.

    Raises
    ------
    ValueError
        If the stored matrices do not have the shape (n_ele, d, d) of the irrep.
    """
    group = get_group(group_name)
    matrices = []
    for label, dim in zip(group.rep_label_list, group.rep_dim_list):
        if group.ele_layout == 'element':
            elements = []
            for i in range(group.order):
                with open(f'{folder}/{label}/{i}.npy', 'rb') as f:
                    elements.append(np.load(f, allow_pickle=True))
            mats = _stack(elements, dim)
        else:
            mats = np.asarray(np.load(f'{folder}/{label}.npy'), dtype=float)
        if mats.shape != (group.order, dim, dim):
            raise ValueError(f"the matrices of {label} in {folder} have shape {mats.shape}")
        mats.setflags(write=False)  # read-only: see above
        matrices.append(mats)
    return tuple(matrices)


######################## Main Class ###############################


class cg_calc:
    '''
    Create one class instance to obtain the CG coefficients related to the tensor product of the
    irreps of a lattice symmetry group specified as input: H(4) (default) or the 16-cell honeycomb
    group W(F₄) (group='honeycomb').

    The class attributes below are those of H(4), kept for compatibility; every instance also has
    them as instance attributes, with the values of its own group (self.group).
    '''

    n_ele_s4 = n_ele_s4
    n_ele_h4 = n_ele_h4
    n_rep = n_rep
    rep_dim_list = rep_dim_list
    fund_index = fund_index
    rep_label_list = rep_label_list
    rep_latex_names = rep_latex_names
    irrep_index = irrep_index
    irrep_dim = irrep_dim
    irrep_texname = irrep_texname

    cg_pdf_folder = 'cg_pdf'

    def __init__(
        self,
        *irreps: tuple[int, int],
        group: str | LatticeGroup | None = None,
        cgdatabase: str | None = None,
        force_computation: bool = False,
        force_group_gen: bool = False,
        prescription_changed: bool = False,
        parallel: bool | None = None,
        n_threads: int | None = None,
        verbose: bool = True,
        save_raw: bool | None = None,
        force_h4gen: bool = False,
    ) -> None:
        """
        Initialise the CG-coefficient calculator for a tensor product of irreps.

        Parameters
        ----------
        *irreps : tuple[int,int]
            Irreps of the group, e.g. (4,1), (4,4), (6,1), …  A single irrep is allowed: its
            "product" is the irrep itself, whose CG matrix is the identity (the operators
            without derivatives).
        group : str, LatticeGroup or None
            The symmetry group: None (default) or 'H4' for the hypercubic group H(4),
            'honeycomb' for the 16-cell honeycomb group W(F₄) (see groups.get_group).
        cgdatabase : str or None
            Path to a CG-coefficient database, used for both reading and writing.
            If None (default), a product is read from the bundled database of the group if it
            is there, else from the local database (./cg_database for H(4),
            ./cg_database_honeycomb for W(F₄)), and new results are written to the local
            database; the bundled data are never modified.
        force_computation : bool
            Recompute CG coefficients even if the database entry already exists.
        force_group_gen : bool
            Recompute the matrices of all the group elements in every irrep from the code
            (generators of H(4), and for W(F₄) the construction of the module honeycomb) and
            save them to the local folder of the group (./h4_ele or ./honeycomb_ele) instead
            of loading the bundled ones.
        prescription_changed : bool
            Re-apply CGmat_from_block to the stored raw symbolic matrices (without
            recomputing them) and save the new numerical matrices to the output
            database (see cgdatabase).  Raises FileNotFoundError if the raw matrices
            are not stored, as for the bundled H(4) products of more than three irreps
            and for all the bundled products of the honeycomb group.
        parallel : bool or None
            When the CG coefficients are computed, do the sum over the group elements
            in parallel threads with numba (see n_threads).  The result is the same, to
            the last bit, as without.  None (default) means True for a product of three
            or more irreps, or if n_threads is given, provided that numba is installed;
            else False.  True requires numba (pip install "h4lat[parallel]").
        n_threads : int or None
            Number of threads of the parallel computation, between 1 and the number
            of CPU cores (or the environment variable NUMBA_NUM_THREADS, if set).
            None (default) uses all of them.  Giving it implies parallel=True; it
            cannot be combined with parallel=False.
        verbose : bool
            Print progress messages.
        save_raw : bool or None
            When the CG coefficients are computed, also build the raw symbolic blocks (the
            input of CGmat_from_block, needed by prescription_changed) and save them to the
            folder '{product}_raw' next to the coefficients.  The coefficients themselves are
            computed numerically and do not depend on this choice.  None (default) means True
            for H(4), as in the versions of h4lat before the honeycomb group, and False for the
            honeycomb group, whose symbolic blocks are too large to build for products of four
            irreps (minutes per copy of an irrep).
        force_h4gen : bool
            Deprecated name of force_group_gen.

        Raises
        ------
        ValueError
            If the group is unknown or an irrep is not an irrep of the group.
        """

        # --- the group and its irreps ----------------------------------------------
        self.group = get_group(group)
        self.chosen_irreps = self.group.check_irreps(irreps)
        # The class attributes are those of H(4); these instance attributes give the values of
        # this instance's group (the same values for H(4)).
        self.n_ele = self.group.order
        self.n_rep = self.group.n_rep
        self.rep_dim_list = list(self.group.rep_dim_list)
        self.rep_label_list = list(self.group.rep_label_list)
        self.rep_latex_names = list(self.group.rep_latex_names)
        self.irrep_index = dict(self.group.irrep_index)
        self.irrep_dim = dict(self.group.irrep_dim)
        self.irrep_texname = dict(self.group.irrep_texname)
        self.fund_index = self.group.irrep_index[(4, 1)]

        if force_h4gen:
            warnings.warn(
                "force_h4gen is deprecated: use force_group_gen, which regenerates the matrices of the elements "
                "of the group of the calculator (H(4) or the honeycomb group)",
                DeprecationWarning,
                stacklevel=2,
            )
            force_group_gen = True
        if save_raw is None:
            save_raw = self.group is _H4

        self._raw_cg = None  # see the raw_cg property
        self._group_mat = None  # see the group_mat property

        # --- data paths -----------------------------------------------------------
        # The bundled package data are only read, never written: regenerated group
        # matrices and new CG coefficients go to the local folders, in the working directory.
        bundled_ele = _bundled(self.group, 'ele')
        if force_group_gen or not Path(bundled_ele).exists():
            self.ele_folder = self.group.ele_folder
        else:
            self.ele_folder = bundled_ele
        # The same folder as an absolute path, fixed now: the local folder is relative to the
        # working directory, which may change before the group_mat property reads the matrices.
        self._ele_path = str(Path(self.ele_folder).resolve())

        # CG database: a user-supplied path is used for both reading and writing.
        # Otherwise new results are written to the local database, and a product is
        # read from the bundled data if it is there, else from the local database.
        product_name = ''.join([str(ir) for ir in self.chosen_irreps])
        self.cg_database_folder = self.group.cg_database_folder if cgdatabase is None else cgdatabase
        self.cg_folder = self.cg_database_folder + '/' + product_name
        bundled_product = _bundled(self.group, 'cg_database') + '/' + product_name
        if cgdatabase is None and Path(bundled_product).exists():
            self.cg_folder = bundled_product
        # where the raw blocks of the product are (or would be) stored: see the raw_cg property
        self._raw_folder = self.cg_folder + "_raw"
        # --------------------------------------------------------------------------

        if not Path(self.ele_folder).exists() or force_group_gen:

            if verbose:
                print(f"\nConstructing matrix representations for all the elements of {self.group.title} ...\n")
            generated = _generate_group_matrices(self.group)

            if verbose:
                print("\nSaving to file the matrix representations ...\n")
            _save_group_matrices(self.group, self.ele_folder, generated)

            # The generated matrices are kept (not read back from the files), stacked and
            # read-only as the loaded ones.
            self.group_mat = [_stack(mats, dim) for mats, dim in zip(generated, self.rep_dim_list)]
            for mats in self.group_mat:
                mats.setflags(write=False)

        # Otherwise the folder exists and force_group_gen is False: the stored matrices (the bundled
        # ones, or those saved earlier in the local folder) are not read here.  They are needed
        # only to compute new CG coefficients, and the group_mat property reads them on first
        # access, once per session and folder (see _load_group_matrices).

        self.mul_list = self.group.multiplicities(*self.chosen_irreps)

        if not Path(self.cg_folder).exists() or force_computation:

            if verbose:
                print(f"\nComputing the cg coefficients for the tensor product {self.chosen_irreps} ...\n")

            if n_threads is not None:
                if parallel is False:
                    raise ValueError(
                        "n_threads is the number of threads of the parallel computation: use parallel=True"
                    )
                parallel = True
            if parallel is None:
                parallel = len(self.chosen_irreps) >= 3 and _NUMBA_AVAILABLE
                if verbose and len(self.chosen_irreps) >= 3 and not _NUMBA_AVAILABLE:
                    print('\nnumba is not installed: the computation runs serially (pip install "h4lat[parallel]")\n')
            elif parallel and not _NUMBA_AVAILABLE:
                raise ImportError(
                    'the parallel computation (parallel=True or n_threads) needs numba.  '
                    'Install it with: pip install "h4lat[parallel]"'
                )
            if parallel:
                max_threads = _numba_max_threads()
                if n_threads is None:
                    n_threads = max_threads
                elif not 1 <= n_threads <= max_threads:
                    raise ValueError(
                        f"n_threads must be between 1 and {max_threads}, the number of threads numba may use "
                        f"(the CPU cores, or the environment variable NUMBA_NUM_THREADS), not {n_threads}"
                    )

            if verbose:
                how = f"in parallel on {n_threads} threads, with numba" if parallel else "serially"
                print(
                    f"\nSumming over the group elements ({how}) ... "
                    "(loop over irreps: one step per copy of each irrep in the decomposition)\n"
                )

            # Sakata projection formula (J. Math. Phys. 15, 1702, 1974):
            #   res_mat = Σ_h T(h) @ A @ B(h)^T,  h ∈ G,
            # where T(h) is the tensor-product representation of h, B(h) is
            # the block-diagonal target representation (one block per irrep
            # copy in the decomposition), and A is a symbolic dim×dim matrix.
            # Each column of res_mat is a linear form in the A_{ij}; the
            # coefficients are the CG matrix entries, with the A_{ij} labelling
            # the remaining freedom.  The sum is done numerically, on these
            # coefficients: the coefficient of A[k, l] in res_mat[i, j] is
            # Σ_h T(h)[i, k] B(h)[j, l], and as B(h) is block-diagonal, the
            # coefficients of a block are those of its irrep (_projection),
            # computed once for all the copies of the irrep.  The copies are then
            # extracted from these numerical coefficients (_cgmat_from_linear_forms);
            # the symbolic blocks (_raw_block) are built only to be saved (save_raw).
            time_start = time.time()
            time_projection = time_raw = 0.0

            T = _product_matrices(self.group_mat, self.chosen_irreps, self.group)
            dim = T.shape[1]
            A = sym.MatrixSymbol('A', dim, dim)

            self.cg_dict = {}
            raw_cg = {}
            offset = 0  # column of res_mat where the block of the current copy starts
            with tqdm(total=int(np.sum(self.mul_list)), disable=not verbose) as progress_bar:
                for irep, mul in enumerate(self.mul_list):
                    if mul == 0:
                        continue
                    time_before_projection = time.time()
                    coeffs = _projection(
                        _element_matrices(self.group_mat, irep, self.group),
                        T,
                        parallel=parallel,
                        n_threads=n_threads,
                    )
                    time_projection += time.time() - time_before_projection
                    # the linear forms of the entries of a block: the same for every copy of the irrep
                    forms = _linear_forms_from_coefficients(coeffs)
                    for m in range(mul):
                        self.cg_dict.setdefault(irep, []).append(
                            _cgmat_from_linear_forms(forms, mul, m, irreps=self.chosen_irreps)
                        )
                        if save_raw:
                            time_before_raw = time.time()
                            raw_cg[(irep, m)] = _raw_block(coeffs, offset, A)
                            time_raw += time.time() - time_before_raw
                        offset += self.rep_dim_list[irep]
                        progress_bar.update()
                    del coeffs, forms
            del T

            if verbose:
                total = time.time() - time_start
                time_cg = total - time_projection - time_raw
                print(
                    f"\nTotal Time: {round(total, 2)} s\nProjection: {round(time_projection / total * 100, 2)}%\n"
                    f"CG blocks: {round(time_cg / total * 100, 2)}%\nRaw blocks: {round(time_raw / total * 100, 2)}%\n"
                )

            if verbose:
                print("\nSaving to file the cg coefficient to the database ...\n")

            # New results always go to the output database, never to the bundled data.
            self.cg_folder = self.cg_database_folder + '/' + product_name
            self._raw_folder = self.cg_folder + "_raw"
            Path(self.cg_folder).mkdir(parents=True, exist_ok=True)

            for k, v in self.cg_dict.items():
                for i, arr in enumerate(v):
                    with open(f'{self.cg_folder}/{k}_{i}.npy', 'wb') as f:
                        np.save(f, arr)

            if save_raw:
                self.raw_cg = raw_cg
                Path(self._raw_folder).mkdir(parents=True, exist_ok=True)
                for (irep, m), v in raw_cg.items():
                    with open(f'{self._raw_folder}/{irep}_{m}.npy', 'wb') as f:
                        np.save(f, v)

        elif Path(self.cg_folder).exists():

            if verbose:
                print("\nLoading the cg coefficients for the given tensor product from the database ...\n")

            # the raw blocks are only read when needed (see the raw_cg property)

            if prescription_changed:
                # A changed prescription is saved like a new result: to the output database,
                # together with the raw matrices it is derived from.
                out_folder = self.cg_database_folder + '/' + product_name
                raw_blocks = self.raw_cg  # raises before anything is written if they are not stored
                Path(out_folder).mkdir(parents=True, exist_ok=True)
                Path(out_folder + "_raw").mkdir(parents=True, exist_ok=True)
                for (irep, m), raw in raw_blocks.items():
                    if out_folder != self.cg_folder:
                        with open(f'{out_folder}_raw/{irep}_{m}.npy', 'wb') as f:
                            np.save(f, raw)
                    with open(f'{out_folder}/{irep}_{m}.npy', 'wb') as f:
                        np.save(f, CGmat_from_block(raw, self.mul_list[irep], m, irreps=self.chosen_irreps))
                self.cg_folder = out_folder
                self._raw_folder = out_folder + "_raw"

            # Files come in (irep, m) order, so cg_dict[irep][m] is always '{irep}_{m}.npy'
            # and the keys are sorted.
            self.cg_dict = {}
            for file in _sorted_cg_files(self.cg_folder):
                irep = int(file.stem.split("_")[0])
                self.cg_dict.setdefault(irep, []).append(np.load(file, allow_pickle=True))

    @property
    def raw_cg(self) -> dict[tuple[int, int], np.ndarray]:
        """
        Raw symbolic CG blocks {(irep, m): block}, from which CGmat_from_block derives cg_dict.

        For a product loaded from a database they are read from '{product}_raw/' on first
        access (they can be large: unpickling the symbolic expressions takes seconds).  They
        are not bundled for the H(4) products of more than three irreps nor for the honeycomb
        group, and they are saved only if the product was computed with save_raw=True:
        recompute the product with force_computation=True and save_raw=True to get them.
        """
        if self._raw_cg is None:
            folder = Path(self._raw_folder)
            if not folder.is_dir():
                raise FileNotFoundError(
                    f"the raw CG blocks of {''.join(str(ir) for ir in self.chosen_irreps)} are not stored in "
                    f"{folder} (they are not bundled for products of more than three irreps of H(4), nor for the "
                    "honeycomb group): recompute them with force_computation=True and save_raw=True"
                )
            self._raw_cg = {
                tuple(int(x) for x in file.stem.split("_")): np.load(file, allow_pickle=True)
                for file in _sorted_cg_files(folder)
            }
        return self._raw_cg

    @raw_cg.setter
    def raw_cg(self, value: dict[tuple[int, int], np.ndarray]) -> None:
        self._raw_cg = value

    @property
    def group_mat(self) -> list[np.ndarray]:
        """
        Matrices of all the elements of the group in every irrep: group_mat[ir] is the read-only
        (n_ele, d, d) array of the matrices of all the elements in the irrep rep_label_list[ir], and
        group_mat[ir][i] the matrix of the i-th element.

        They are needed only to compute new CG coefficients.  Generated matrices (force_group_gen,
        or no folder of matrices) are set in __init__; stored ones are read on first access, through
        _load_group_matrices, which reads the files of a folder only once per session.  The arrays
        are then shared with every other cg_calc of the same folder, and read-only.
        """
        if self._group_mat is None:
            # A list of this instance, holding the shared, cached arrays.
            self._group_mat = list(_load_group_matrices(self._ele_path, self.group.name))
        return self._group_mat

    @group_mat.setter
    def group_mat(self, value: list[np.ndarray]) -> None:
        self._group_mat = value

    # Older names, from when h4lat knew only H(4): h4_mat[ir][e] still is the matrix of element e in
    # the irrep rep_label_list[ir], and h4_ele_folder the folder of the matrices.
    h4_mat = property(group_mat.fget, group_mat.fset, doc="Older name of group_mat.")

    @property
    def h4_ele_folder(self) -> str:
        """Older name of ele_folder: the folder of the matrices of the group elements."""
        return self.ele_folder

    def get_multiplicities(self) -> list[int]:
        """Return the multiplicity list for the tensor product under study."""
        return self.mul_list

    def latex_print(
        self,
        digits: int = 5,
        verbose: bool = True,
        title: str | None = None,
        author: str = "E.T.",
        clean_tex: bool = True,
    ) -> None:
        """Generate a LaTeX PDF with all CG coefficients.  Requires pylatex."""
        if not _PYLATEX_AVAILABLE:
            raise ImportError("pylatex is required for latex_print().  Install it with: pip install pylatex")

        if title is None:
            title = ' x '.join([str(ir) for ir in self.chosen_irreps])
            if self.group is not _H4:
                title += f" ({self.group.title})"

        Path(self.cg_pdf_folder).mkdir(parents=True, exist_ok=True)
        doc_name = ''.join([str(ir) for ir in self.chosen_irreps]) + self.group.operator_file_suffix

        doc = Document(default_filepath=f'{doc_name}.tex', documentclass='article')
        doc.preamble.append(Command("title", title))
        doc.preamble.append(Command("author", author))
        doc.preamble.append(Command("date", NoEscape(r"\today")))
        doc.append(NoEscape(r"\maketitle"))
        doc.append(NoEscape(r"\newpage"))

        for k, v in self.cg_dict.items():
            section = Section(str(self.rep_label_list[k]), numbering=False)
            for i, cgmat in enumerate(v):
                matrix = Matrix(np.matrix(np.round(np.asarray(cgmat).astype(np.float64), digits)), mtype="b")
                math = Math(data=[f"M_{i+1}=", matrix])
                section.append(math)
            doc.append(section)
            doc.append(NoEscape(r"\newpage"))

        doc.generate_pdf(self.cg_pdf_folder + '/' + doc_name, clean_tex=clean_tex)

        if verbose:
            print(f"\nPdf files containing the cg coefficients located in {self.cg_pdf_folder}/{doc_name}\n")


######################## Auxiliary Functions ###############################


def get_multiplicities(*irreps: tuple[int, int], group: str | LatticeGroup | None = None) -> list[int]:
    """
    Compute the multiplicity of each irrep in the given tensor product.

    Parameters
    ----------
    *irreps : tuple[int,int]
        Irreps of the group, e.g. (4,1), (4,4), …
    group : str, LatticeGroup or None
        None (default) or 'H4' for H(4), 'honeycomb' for the 16-cell honeycomb group W(F₄).

    Returns
    -------
    list[int]
        Entry i is the multiplicity of the i-th irrep of the group (20 entries for H(4), 25 for
        the honeycomb group), in the order of its rep_label_list.

    Raises
    ------
    ValueError
        If an irrep is not an irrep of the group.
    """
    return get_group(group).multiplicities(*irreps)


def latex_print_multiplicities(
    *irreps: tuple[int, int],
    display_eq: bool = False,
    return_str: bool = True,
    group: str | LatticeGroup | None = None,
) -> str | None:
    """
    Return (and optionally display) the LaTeX string for the irrep decomposition.

    Parameters
    ----------
    *irreps : tuple[int,int]
        Irreps forming the tensor product.
    display_eq : bool
        If True and inside a Jupyter notebook, render the equation inline.
    return_str : bool
        If True, return the LaTeX string.
    group : str, LatticeGroup or None
        None (default) or 'H4' for H(4) (irreps τ), 'honeycomb' for W(F₄) (irreps ϱ).
    """
    group = get_group(group)
    irreps = group.check_irreps(irreps)
    mult_list = group.multiplicities(*irreps)
    out_str = r"$"

    for i, (a, b) in enumerate(irreps):
        out_str += f"{group.latex_symbol}^{{\\left({a}\\right)}}_{{{b}}}"
        if i != len(irreps) - 1:
            out_str += " \\otimes "

    out_str += " = "

    non_zero = [(i, mult) for i, mult in enumerate(mult_list) if mult != 0]
    for j, (i, mult) in enumerate(non_zero):
        out_str += f"{mult if mult > 1 else ''} \\, {group.rep_latex_names[i]}"
        if j != len(non_zero) - 1:
            out_str += " \\oplus "

    out_str += "$"

    if display_eq and _IPYTHON_AVAILABLE:
        display(MathDisplay(out_str))

    return out_str if return_str else None


def _element_matrices(group_mat, irep: int, group: str | LatticeGroup | None = None) -> np.ndarray:
    """The matrices of all the elements of the group in the irrep of index *irep*, as an (n_ele, d, d) array."""
    return _stack(group_mat[irep], get_group(group).rep_dim_list[irep])


def _product_matrices(
    group_mat, irreps: tuple[tuple[int, int], ...], group: str | LatticeGroup | None = None
) -> np.ndarray:
    """T(h) = D_1(h) ⊗ … ⊗ D_n(h) for every element h of the group, as an (n_ele, dim, dim) array."""
    group = get_group(group)
    factors = [_element_matrices(group_mat, group.irrep_index[tuple(rep)], group) for rep in irreps]
    dim = int(np.prod([D.shape[1] for D in factors]))
    T = np.empty((group.order, dim, dim))
    for ih in range(group.order):
        Tp = np.eye(1)
        for D in factors:
            Tp = np.kron(Tp, D[ih])
        T[ih] = Tp
    return T


def _projection(D: np.ndarray, T: np.ndarray, parallel: bool = False, n_threads: int | None = None) -> np.ndarray:
    """
    Coefficients of the Sakata projection onto one irrep.

    Parameters
    ----------
    D : ndarray
        Matrices of all the group elements in the irrep, shape (n_ele, d, d).
    T : ndarray
        Matrices of all the group elements in the tensor product, shape (n_ele, dim, dim).
    parallel : bool
        Run the loop in parallel threads with numba (numba must be installed).
    n_threads : int or None
        Number of threads when *parallel*; None means all those numba may use
        (_numba_max_threads).  The caller's numba setting is restored afterwards.

    Returns
    -------
    ndarray
        P of shape (d, d, dim, dim), with P[c, l, i, k] = Σ_h D[h, c, l] T[h, i, k]: the
        coefficient of A[k, l] in entry (i, c) of Σ_h T(h) A D(h)^T.

    Notes
    -----
    The sum runs over h in order, as the symbolic loop that this function replaces did
    (res_mat = res_mat + T(h) @ A @ B(h)^T), so the coefficients are the same to the last
    bit.  In parallel every thread computes all the terms of its own entries, again with
    h in order, so the result does not depend on *parallel* or on the number of threads.
    """
    D = np.ascontiguousarray(D, dtype=float)
    T = np.ascontiguousarray(T, dtype=float)
    d, dim = D.shape[1], T.shape[1]
    P = np.zeros((d, d, dim, dim))
    if parallel:
        import numba

        kernel = _numba_projection_kernel()
        previous_n_threads = numba.get_num_threads()
        numba.set_num_threads(_numba_max_threads() if n_threads is None else n_threads)
        try:
            kernel(D, T, P)
        finally:
            numba.set_num_threads(previous_n_threads)
    else:
        for h in range(len(D)):
            P += np.multiply.outer(D[h], T[h])
    return P


def _numba_max_threads() -> int:
    """The number of threads numba may use: the CPU cores, or the environment variable NUMBA_NUM_THREADS if set."""
    import numba

    return numba.config.NUMBA_NUM_THREADS


@functools.cache
def _numba_projection_kernel():
    """The loop of _projection compiled with numba, which is imported and compiles it on first use."""
    import numba

    @numba.njit(parallel=True)
    def kernel(D, T, P):
        n_ele, d, _ = D.shape
        dim = T.shape[1]
        # rows i of the result in parallel; for each entry the terms are added with h in order
        for i in numba.prange(dim):
            for h in range(n_ele):
                for c in range(d):
                    for l in range(d):
                        b = D[h, c, l]
                        if b != 0.0:  # a zero term would leave the entry unchanged
                            for k in range(dim):
                                P[c, l, i, k] += b * T[h, i, k]

    return kernel


def _raw_block(coeffs: np.ndarray, offset: int, A: sym.MatrixSymbol, eps: float = 1e-10, infth: float = 1e20):
    """
    Symbolic CG block of one copy of an irrep, from the coefficients given by _projection.

    Entry (i, c) is Σ_{k,l} coeffs[c, l, i, k] A[k, offset + l], where *offset* is the
    column at which the block of this copy starts.  Coefficients with |c| < eps are
    floating-point noise and |c| > infth flags a divergent entry: both are dropped.
    """
    d, dim = coeffs.shape[0], coeffs.shape[2]
    block = np.empty((dim, d), dtype=object)
    for i in range(dim):
        for c in range(d):
            entry = coeffs[c, :, i, :]  # entry[l, k] is the coefficient of A[k, offset + l]
            ls, ks = np.nonzero((np.abs(entry) >= eps) & (np.abs(entry) <= infth))
            block[i, c] = sym.Add(*[sym.Float(entry[l, k]) * A[k, offset + l] for l, k in zip(ls, ks)])
    return block


def _linear_forms_from_coefficients(coeffs: np.ndarray, eps: float = 1e-10, infth: float = 1e20) -> np.ndarray:
    """
    The linear forms of the entries of a CG block, computed from the coefficients given by _projection.

    Entry (i, c) of the block of any copy of the irrep is Σ_{k,l} coeffs[c, l, i, k] A[k, offset + l]
    (see _raw_block).  This returns the array L of _linear_forms(_raw_block(coeffs, offset, A)),
    built without sympy: L[i, c, s] is the coefficient of the s-th symbol A[k, offset + l] in entry
    (i, c), the symbols being those with at least one coefficient kept, sorted by (k, l).  The
    coefficients dropped by _raw_block (|c| < eps, noise, and |c| > infth, divergent) are set to 0.
    Every float is the same as in the symbolic route (sympy stores a Python float exactly and gives
    it back unchanged), so the CG blocks derived from L are the same to the last bit; the offset only
    renames the symbols, so L is the same for every copy of the irrep.

    Parameters
    ----------
    coeffs : ndarray
        Output of _projection, shape (d, d, dim, dim): coeffs[c, l, i, k].
    eps, infth : float
        Thresholds of _raw_block.

    Returns
    -------
    ndarray
        L, shape (dim, d, n_symbols).
    """
    d, dim = coeffs.shape[0], coeffs.shape[2]
    kept = (np.abs(coeffs) >= eps) & (np.abs(coeffs) <= infth)
    # transpose(2, 0, 3, 1): [c, l, i, k] → [i, c, k, l]; then (k, l) flattened with l fastest, which
    # is the order of the symbols A[k, offset + l] sorted by (row, column)
    forms = np.where(kept, coeffs, 0.0).transpose(2, 0, 3, 1).reshape(dim, d, dim * d)
    used = kept.transpose(2, 0, 3, 1).reshape(dim, d, dim * d).any(axis=(0, 1))
    # The selection along the last axis does not give a C-ordered array, while _linear_forms does.
    # The memory layout matters: matrix products (BLAS) may add the terms in a different order
    # for a different layout, which changes the last bits of the CG blocks.
    return np.ascontiguousarray(forms[:, :, used])


def _c_parity_permutation(irreps: tuple[tuple[int, int], ...]) -> np.ndarray | None:
    """Row permutation of a CG block that reverses the order of the factors after the first.

    For an operator ψ̄ Γ D_{μ₂} … D_{μₙ} ψ, with Γ transforming as irreps[0] and each
    derivative as one of irreps[1:], charge conjugation reverses the derivative indices
    (see C_parity in moments_operator): on the rows of a CG block this is the returned
    permutation, up to an overall sign.  Returns None when there is nothing to reverse
    (fewer than two factors after the first) or when the factors after the first are
    not all the same irrep (then the reversal is not a symmetry of the product).

    The irreps may be those of any group of h4lat: in all of them the irrep (k, l) has dimension k.
    """
    irreps = tuple(tuple(irrep) for irrep in irreps)
    if len(irreps) < 3 or len(set(irreps[1:])) != 1:
        return None
    dims = [int(irrep[0]) for irrep in irreps]
    rows = np.arange(int(np.prod(dims))).reshape(dims)
    return np.transpose(rows, [0] + list(range(len(dims) - 1, 0, -1))).ravel()


def CGmat_from_block(
    block: np.ndarray,
    multiplicity: int,
    multiplicity_index: int,
    normalize: bool = True,
    threshold: float = 1e-10,
    enforce_c_parity: bool = True,
    irreps: tuple[tuple[int, int], ...] | None = None,
) -> np.ndarray:
    """
    Extract one numerical copy of an irrep from a symbolic CG block.

    Parameters
    ----------
    block : ndarray
        Symbolic block from the projection formula, of shape (dim of the tensor
        product, dim of the irrep) (a 1-D block is read as a single column).  Each
        entry is a linear combination of the free symbols A[i, j].
    multiplicity : int
        Multiplicity of the irrep in the tensor product.
    multiplicity_index : int
        Which copy (0 … multiplicity-1) to extract.
    normalize : bool
        Divide each column by the absolute value of its first non-zero entry.
    threshold : float
        Entries smaller than this are set to zero (floating-point noise).
    enforce_c_parity : bool
        Choose copies with a definite C-parity, i.e. symmetric or antisymmetric under
        reversal of the factors after the first (the derivative indices of an
        operator); the symmetric copies come first.  Requires *irreps*.  Has no
        effect when there are fewer than two factors after the first, or when they
        are not all the same irrep.
    irreps : tuple of tuple[int, int], optional
        Factors of the tensor product, e.g. ((6, 1), (4, 1), (4, 1)).  Required
        when *enforce_c_parity* is True.

    Returns
    -------
    ndarray
        Float matrix of the same shape as *block*, whose columns span one copy of
        the irrep.

    Raises
    ------
    ValueError
        If the free parameters of the block do not have *multiplicity* degrees of
        freedom, i.e. the block is not a projection onto *multiplicity* copies; or
        if *enforce_c_parity* is True and *irreps* is missing or does not match
        the block.

    Notes
    -----
    The projection formula gives block = P(A), linear in the matrix of symbols A,
    and P(A) is an exact Clebsch-Gordan block for *any* numerical value of A: the
    symbols only parametrise the multiplicity-dimensional space of copies.  By
    Schur's lemma a copy is fixed by its first column, so the entries of column 0,
    as linear forms in A, span exactly *multiplicity* dimensions.

    The copies are chosen as follows: take the first *multiplicity* rows of column 0
    whose linear forms are linearly independent (the reference rows), and for copy m
    choose the A (of minimal norm) that makes reference row i equal to δ_{i,m}.
    Copy m is then non-zero at reference row m and zero at the others, so the copies
    are independent, and each is an exact CG block because it is P(A) for one A.
    Everything is done on the numerical coefficients of the symbols, so the result
    does not depend on how sympy orders the terms of an expression.

    The first non-zero entry of column 0 of every copy is then +1 (its reference
    row); with multiplicity 1, where the copy is the same for any A up to a factor,
    this is what fixes the factor.  With *normalize*, every column is further divided
    by the absolute value of its first non-zero entry, so that entry is ±1: +1 for
    column 0, while the other columns keep their sign.

    With *enforce_c_parity*, every copy also has a definite C-parity.  Charge
    conjugation reverses the derivative indices, which on the rows of the block is a
    permutation R (up to a sign that does not depend on the copy).  R commutes with
    the action of the group, since it only permutes identical factors, so R·P(A) is
    again a CG block: R maps the space of copies into itself.  As R² = 1, that space
    splits into the copies even and odd under R, which are those of the blocks
    (1 + R)/2 · P(A) and (1 − R)/2 · P(A).  The copies are chosen as above within each
    of the two blocks, the even ones first.  A copy of multiplicity 1 is already even
    or odd, so it is the same as without the projection.

    cg_calc does the same without sympy: it builds the linear forms directly from the
    numerical coefficients of the projection (_linear_forms_from_coefficients) and calls
    _cgmat_from_linear_forms, the numerical part of this function, with the same result.
    """
    return _cgmat_from_linear_forms(
        _linear_forms(block),
        multiplicity,
        multiplicity_index,
        normalize=normalize,
        threshold=threshold,
        enforce_c_parity=enforce_c_parity,
        irreps=irreps,
    )


def _linear_forms(block: np.ndarray) -> np.ndarray:
    """
    The coefficients of the free symbols A[i, j] in every entry of a symbolic CG block.

    Parameters
    ----------
    block : ndarray
        Symbolic block (see CGmat_from_block); a 1-D block is read as a single column.

    Returns
    -------
    ndarray
        L of shape (rows, columns, n_symbols): L[r, c, s] is the coefficient of the s-th symbol
        in entry (r, c), the symbols being all those that appear in the block, sorted by (i, j).
        Then block = L @ A once A is flattened in that order.
    """
    # a 1-D block is a single column
    mat = np.asarray(block, dtype=object)
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    nrows, ncols = mat.shape

    # coefficients of every entry, {symbol A[i, j]: coefficient}; constant terms (key 1)
    # cannot appear in a projection, which is linear in A
    coeff_dicts = [[sym.sympify(entry).as_coefficients_dict() for entry in row] for row in mat]
    symbols = sorted(
        {s for row in coeff_dicts for coeffs in row for s in coeffs if s != 1},
        key=lambda s: (int(s.i), int(s.j)),
    )
    symbol_index = {s: k for k, s in enumerate(symbols)}

    # L[r, c, k] = coefficient of the k-th symbol in entry (r, c), so that block = L @ A
    # once A is flattened in the order of symbols
    L = np.zeros((nrows, ncols, len(symbols)))
    for r in range(nrows):
        for c in range(ncols):
            for s, coeff in coeff_dicts[r][c].items():
                if s != 1:
                    L[r, c, symbol_index[s]] = float(coeff)
    return L


def _cgmat_from_linear_forms(
    L: np.ndarray,
    multiplicity: int,
    multiplicity_index: int,
    normalize: bool = True,
    threshold: float = 1e-10,
    enforce_c_parity: bool = True,
    irreps: tuple[tuple[int, int], ...] | None = None,
) -> np.ndarray:
    """
    Extract one numerical copy of an irrep from the linear forms of a CG block.

    The numerical part of CGmat_from_block, which explains the algorithm and the parameters.

    Parameters
    ----------
    L : ndarray
        Shape (rows, columns, n_symbols): L[r, c, s] is the coefficient of the s-th free symbol in
        entry (r, c) of the block (from _linear_forms or _linear_forms_from_coefficients).
    multiplicity, multiplicity_index, normalize, threshold, enforce_c_parity, irreps
        As in CGmat_from_block.

    Returns
    -------
    ndarray
        Float matrix of shape (rows, columns), whose columns span one copy of the irrep.
    """
    nrows, ncols, n_symbols = L.shape

    # column 0 fixes the copy: its linear forms must span exactly `multiplicity` dimensions
    L0 = L[:, 0, :]
    tol = threshold * max(1.0, np.abs(L0).max()) if n_symbols else threshold
    rank = np.linalg.matrix_rank(L0, tol=tol) if n_symbols else 0
    if rank != multiplicity:
        raise ValueError(
            f"column 0 of the block has {rank} independent entries, expected the multiplicity {multiplicity}"
        )

    # C-parity: split L into the parts even and odd under the reversal R of the derivative
    # indices, (1 ± R)/2 · L; the copies of each part are exactly the even / odd ones
    perm = None
    if enforce_c_parity:
        if irreps is None:
            raise ValueError("enforce_c_parity needs the factors of the tensor product: pass irreps")
        perm = _c_parity_permutation(irreps)
        if perm is not None and len(perm) != nrows:
            raise ValueError(f"the block has {nrows} rows, but the product {irreps} has dimension {len(perm)}")
    parts = [(L, 1)] if perm is None else [((L + L[perm]) / 2, 1), ((L - L[perm]) / 2, -1)]
    part_ranks = [np.linalg.matrix_rank(part[:, 0, :], tol=tol) if n_symbols else 0 for part, _ in parts]
    if sum(part_ranks) != multiplicity:
        raise ValueError(
            f"the even and odd parts of the block have {part_ranks} copies, expected the multiplicity {multiplicity}"
        )

    # the part that holds copy `multiplicity_index` (even copies first), and the copy's index in it
    ipart, index = 0, multiplicity_index
    while index >= part_ranks[ipart]:
        index -= part_ranks[ipart]
        ipart += 1
    (L_part, sign), n_copies = parts[ipart], part_ranks[ipart]
    L0_part = L_part[:, 0, :]

    # reference rows: the first rows of column 0 (in row order) with independent linear forms
    ref_rows = []
    for r in range(nrows):
        if np.linalg.matrix_rank(L0_part[ref_rows + [r]], tol=tol) > len(ref_rows):
            ref_rows.append(r)
            if len(ref_rows) == n_copies:
                break

    # the A of minimal norm with reference row i of column 0 equal to δ_{i, index}
    target = np.zeros(n_copies)
    target[index] = 1.0
    a = np.linalg.pinv(L0_part[ref_rows]) @ target

    # the copy: the block evaluated at that A
    newmat = L_part @ a
    if perm is not None:
        # exactly even / odd, also in the last bit (rounding in L_part @ a can break it)
        newmat = (newmat + sign * newmat[perm]) / 2
    newmat[np.abs(newmat) < threshold] = 0.0

    # each column scaled so that its first non-zero entry is ±1
    if normalize:
        for j in range(ncols):
            first = np.flatnonzero(newmat[:, j])[0]
            newmat[:, j] /= np.abs(newmat[first, j])
        newmat[np.abs(newmat) < threshold] = 0.0

    return newmat


def print_CGmat(cgmat: np.ndarray, digits: int = 5) -> None:
    """Pretty-print the CG matrix inside a Jupyter notebook."""
    if not _IPYTHON_AVAILABLE:
        raise ImportError("IPython is required for print_CGmat().")
    display(sym.Matrix(np.round(np.asarray(cgmat).astype(np.float64), digits)))


######################## Execution of the program as Main ###################

if __name__ == "__main__":
    # python cg_calculator.py [n] [scalar|pseudoscalar|vector|axial|tensor|all]: compute (and print to
    # PDF) the CG coefficients of the leading irrep of the chosen Dirac structure times n-1 (4,1) factors.
    n = 2
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print(f"\nSpecified n was {sys.argv[1]}, proceeding with n={n}\n")

    # irrep of the Dirac structure Γ of the operator ψ̄ Γ D … D ψ
    leading_irreps = {"scalar": (1, 1), "pseudoscalar": (1, 4), "vector": (4, 1), "axial": (4, 4), "tensor": (6, 1)}
    which = "all"
    if len(sys.argv) > 2:
        if str(sys.argv[2]) in list(leading_irreps) + ["all"]:
            which = str(sys.argv[2])

    for name, leading in leading_irreps.items():
        if which in ("all", name):
            cg = cg_calc(leading, *[(4, 1)] * (n - 1), force_computation=True)
            cg.latex_print()
