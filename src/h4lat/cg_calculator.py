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
Clebsch-Gordan coefficient calculator for the hypercubic group H(4).

H(4) = S(4) ⋊ Z₂⁴ is the symmetry group of the 4-dimensional hypercubic
lattice, of order 384.  It has 20 irreducible representations labelled (k,l)
where k is the dimension and l distinguishes inequivalent irreps of the same
dimension; this labelling follows Baake et al. (1982) [J. Math. Phys. 23, 944].

CG coefficients are computed via the projection formula of Sakata (1974)
[J. Math. Phys. 15, 1702]: summing the tensor-product matrices against the
target irrep matrices over all group elements extracts the CG vectors as
linear forms in a matrix of free symbols A.  The sum is done numerically, on
the coefficients of these forms (see _projection), optionally in parallel with
numba.  The resulting free parameters (freedom within degenerate
subspaces) are fixed by CGmat_from_block, which picks one independent copy
of the irrep for each unit of multiplicity, each with a definite C-parity
(symmetric or antisymmetric under reversal of the derivative indices).

Main entry point: cg_calc.
Module-level helpers: get_multiplicities, latex_print_multiplicities.
"""

######################## Library Imports ################################

import functools
import sys
import time
from importlib.resources import files as _resource_files
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import sympy as sym
from tqdm import tqdm

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


_BUNDLED_H4_ELE = _pkg_data('h4_ele')
_BUNDLED_CG_DATABASE = _pkg_data('cg_database')
_BUNDLED_OPERATOR_DATABASE = _pkg_data('operator_database')

# Data generated at runtime goes to these folders in the working directory, never
# into the (possibly read-only) package data.  To update the bundled data, use
# scripts/regenerate_data.py, or copy them into src/h4lat/data/ by hand.
_LOCAL_H4_ELE = 'h4_ele'
_LOCAL_CG_DATABASE = 'cg_database'


def _sorted_cg_files(folder: str) -> list[Path]:
    """Return the CG files '{irep}_{m}.npy' in *folder* sorted by (irep, m), not in filesystem order."""
    return sorted(Path(folder).glob('*.npy'), key=lambda f: tuple(int(x) for x in f.stem.split('_')))


######################## Global Variables ###############################

## specifics of the H(4) irreps ##

n_ele_s4 = 4 * 3 * 2
n_ele_h4 = 2**4 * 4 * 3 * 2
n_rep = 20
rep_dim_list = [1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 6, 6, 6, 6, 8, 8]
fund_index = 10
rep_label_list = [
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
]
rep_latex_names = [f"\\tau^{{\\left({i[0]}\\right)}}_{i[1]}" for i in rep_label_list]
irrep_index = dict(zip(rep_label_list, range(n_rep)))
irrep_dim = dict(zip(rep_label_list, rep_dim_list))
irrep_texname = dict(zip(rep_label_list, rep_latex_names))

char_table = np.array(
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

class_orders = [1, 4, 12, 12, 6, 32, 24, 24, 4, 32, 48, 48, 12, 24, 12, 32, 32, 12, 12, 1]


## matrix representations for the generators of H(4) ##

# H(4) = S(4) ⋊ Z₂⁴.  S(4) is generated by α (a transposition in the
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
# which violates (αβ)³ = 1 and so does not define a representation (see README).
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
#   γⱼ = β^{-(j-1)} γ₁ β^{j-1},  j = 2, 3, 4.
# This ties the four Z₂ reflections to the S(4) action, consistent with the
# semidirect-product structure H(4) = S(4) ⋊ Z₂⁴.
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


######################## Main Class ###############################


class cg_calc:
    '''
    Create one class instance to obtain the CG coefficients related to the
    tensor product of H(4) irreps specified as input.
    '''

    n_ele_s4 = 4 * 3 * 2
    n_ele_h4 = 2**4 * 4 * 3 * 2
    n_rep = 20
    rep_dim_list = [1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 6, 6, 6, 6, 8, 8]
    fund_index = 10
    rep_label_list = rep_label_list
    rep_latex_names = rep_latex_names
    irrep_index = irrep_index
    irrep_dim = irrep_dim
    irrep_texname = irrep_texname

    cg_pdf_folder = 'cg_pdf'

    def __init__(
        self,
        *irreps: tuple[int, int],
        cgdatabase: str | None = None,
        force_computation: bool = False,
        force_h4gen: bool = False,
        prescription_changed: bool = False,
        parallel: bool | None = None,
        n_threads: int | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Initialise the CG-coefficient calculator for a tensor product of H(4) irreps.

        Parameters
        ----------
        *irreps : tuple[int,int]
            Irreps of H(4), e.g. (4,1), (4,4), (6,1), …  A single irrep is allowed: its
            "product" is the irrep itself, whose CG matrix is the identity (the operators
            without derivatives).
        cgdatabase : str or None
            Path to a CG-coefficient database, used for both reading and writing.
            If None (default), a product is read from the bundled database if it is
            there, else from ./cg_database, and new results are written to
            ./cg_database; the bundled data are never modified.
        force_computation : bool
            Recompute CG coefficients even if the database entry already exists.
        force_h4gen : bool
            Recompute all H(4) matrix representations from the generators and save
            them to ./h4_ele instead of loading the bundled ones.
        prescription_changed : bool
            Re-apply CGmat_from_block to the stored raw symbolic matrices (without
            recomputing them) and save the new numerical matrices to the output
            database (see cgdatabase).  Raises FileNotFoundError if the raw matrices
            are not stored, as for the bundled products of more than three irreps.
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
        """

        self.chosen_irreps = irreps
        self._raw_cg = None  # see the raw_cg property
        self._h4_mat = None  # see the h4_mat property

        # --- data paths -----------------------------------------------------------
        # The bundled package data are only read, never written: regenerated H(4)
        # matrices and new CG coefficients go to the local folders (_LOCAL_*).
        if force_h4gen or not Path(_BUNDLED_H4_ELE).exists():
            self.h4_ele_folder = _LOCAL_H4_ELE
        else:
            self.h4_ele_folder = _BUNDLED_H4_ELE
        # The same folder as an absolute path, fixed now: the local folder is relative to the
        # working directory, which may change before the h4_mat property reads the matrices.
        self._h4_ele_path = str(Path(self.h4_ele_folder).resolve())

        # CG database: a user-supplied path is used for both reading and writing.
        # Otherwise new results are written to the local database, and a product is
        # read from the bundled data if it is there, else from the local database.
        product_name = ''.join([str(ir) for ir in self.chosen_irreps])
        self.cg_database_folder = _LOCAL_CG_DATABASE if cgdatabase is None else cgdatabase
        self.cg_folder = self.cg_database_folder + '/' + product_name
        if cgdatabase is None and Path(_BUNDLED_CG_DATABASE + '/' + product_name).exists():
            self.cg_folder = _BUNDLED_CG_DATABASE + '/' + product_name
        # --------------------------------------------------------------------------

        if not Path(self.h4_ele_folder).exists() or force_h4gen:

            if verbose:
                print("\nConstructing matrix representations for all the elements of H(4) ...\n")

            self.s4_mat = [[np.eye(d)] for d in self.rep_dim_list]
            self.start = 0

            # Coset enumeration for S(4): apply all words α^i β^k (i∈{0,1},
            # k∈{0,1,2,3}) to every known element until all 24 = 4! elements
            # are generated.  Duplicate detection is done in the fundamental
            # (4,1) irrep only; the same word is applied in every irrep
            # simultaneously, avoiding redundant computation.
            while len(self.s4_mat[0]) != self.n_ele_s4:
                for i in range(2):
                    for k in range(4):
                        res = self.s4_mat[self.fund_index][self.start]
                        for _ in range(i):
                            res = res @ alpha_list[self.fund_index]
                        for _ in range(k):
                            res = res @ beta_list[self.fund_index]

                        duplicate = 0
                        for l in self.s4_mat[self.fund_index]:
                            if (l == res).all():
                                duplicate = 1
                                break

                        if duplicate == 0:
                            for ir, rep in enumerate(self.s4_mat):
                                res = rep[self.start]
                                res = np.reshape(res, (self.rep_dim_list[ir], self.rep_dim_list[ir]))
                                for _ in range(i):
                                    res = res @ alpha_list[ir]
                                    res = np.reshape(res, (self.rep_dim_list[ir], self.rep_dim_list[ir]))
                                for _ in range(k):
                                    res = res @ beta_list[ir]
                                    res = np.reshape(res, (self.rep_dim_list[ir], self.rep_dim_list[ir]))
                                self.s4_mat[ir].append(np.reshape(res, (self.rep_dim_list[ir], self.rep_dim_list[ir])))
                self.start += 1

            # Extend S(4) to H(4) = S(4) ⋊ Z₂⁴: for every S(4) element,
            # apply all 2⁴ = 16 sign-flip words from {γ₁,γ₂,γ₃,γ₄},
            # giving 24 × 16 = 384 distinct group elements.
            self.h4_mat = [[] for _ in range(self.n_rep)]
            for ir in range(self.n_rep):
                for s4_ele in self.s4_mat[ir]:
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
                                    self.h4_mat[ir].append(new_ele)

            if verbose:
                print("\nSaving to file the matrix representations ...\n")

            for ir in range(self.n_rep):
                rep_folder = self.h4_ele_folder + f"/{str(self.rep_label_list[ir])}"
                Path(rep_folder).mkdir(parents=True, exist_ok=True)
                for i, ele in enumerate(self.h4_mat[ir]):
                    with open(f'{rep_folder}/{i}.npy', 'wb') as f:
                        np.save(f, ele)

            # The files of this folder have changed, so any copy of them that _load_h4_matrices
            # has cached is stale (functools.cache can only forget all its entries at once).
            _load_h4_matrices.cache_clear()

        # Otherwise the folder exists and force_h4gen is False: the stored matrices (the bundled
        # ones, or those saved earlier in the local folder) are not read here.  They are needed
        # only to compute new CG coefficients, and the h4_mat property reads them on first
        # access, once per session and folder (see _load_h4_matrices).

        self.mul_list = get_multiplicities(*irreps)

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
            #   res_mat = Σ_h T(h) @ A @ B(h)^T,  h ∈ H(4),
            # where T(h) is the tensor-product representation of h, B(h) is
            # the block-diagonal target representation (one block per irrep
            # copy in the decomposition), and A is a symbolic dim×dim matrix.
            # Each column of res_mat is a linear form in the A_{ij}; the
            # coefficients are the CG matrix entries, with the A_{ij} labelling
            # the remaining freedom.  The sum is done numerically, on these
            # coefficients: the coefficient of A[k, l] in res_mat[i, j] is
            # Σ_h T(h)[i, k] B(h)[j, l], and as B(h) is block-diagonal, the
            # coefficients of a block are those of its irrep (_projection),
            # computed once for all the copies of the irrep.
            time_start_loop = time.time()
            totalT_projection = 0

            T = _product_matrices(self.h4_mat, self.chosen_irreps)
            dim = T.shape[1]
            A = sym.MatrixSymbol('A', dim, dim)

            self.raw_cg = {}
            offset = 0  # column of res_mat where the block of the current copy starts
            with tqdm(total=int(np.sum(self.mul_list)), disable=not verbose) as progress_bar:
                for irep, mul in enumerate(self.mul_list):
                    if mul == 0:
                        continue
                    time_before_projection = time.time()
                    coeffs = _projection(
                        _element_matrices(self.h4_mat, irep), T, parallel=parallel, n_threads=n_threads
                    )
                    totalT_projection += time.time() - time_before_projection
                    for m in range(mul):
                        self.raw_cg[(irep, m)] = _raw_block(coeffs, offset, A)
                        offset += rep_dim_list[irep]
                        progress_bar.update()
            del T, coeffs

            time_end_loop = time.time()

            self.cg_dict = {}
            for (irep, m), raw in self.raw_cg.items():
                self.cg_dict.setdefault(irep, []).append(
                    CGmat_from_block(raw, self.mul_list[irep], m, irreps=self.chosen_irreps)
                )

            time_end_evaluation = time.time()

            totalT_loop = time_end_loop - time_start_loop
            totalT_evaluating = time_end_evaluation - time_end_loop
            totalT = totalT_loop + totalT_evaluating

            if verbose:
                print(
                    f"\nTotal Time (Projection+Raw blocks+Evaluation): {round(totalT,2)} s\nProjection: {round(totalT_projection/totalT*100,2)}%\nRaw blocks: {round((totalT_loop-totalT_projection)/totalT*100,2)}%\nEvaluation: {round(totalT_evaluating/totalT*100,2)}%\n"
                )

            if verbose:
                print("\nSaving to file the cg coefficient to the database ...\n")

            # New results always go to the output database, never to the bundled data.
            self.cg_folder = self.cg_database_folder + '/' + product_name
            Path(self.cg_folder).mkdir(parents=True, exist_ok=True)
            Path(self.cg_folder + "_raw").mkdir(parents=True, exist_ok=True)

            for k, v in self.cg_dict.items():
                for i, arr in enumerate(v):
                    with open(f'{self.cg_folder}/{k}_{i}.npy', 'wb') as f:
                        np.save(f, arr)

            for k, v in self.raw_cg.items():
                irep, m = k
                with open(f'{self.cg_folder}_raw/{irep}_{m}.npy', 'wb') as f:
                    np.save(f, v)

        elif Path(self.cg_folder).exists():

            if verbose:
                print("\nLoading the cg coefficients for the given tensor product from the database ...\n")

            # the raw blocks are only read when needed (see the raw_cg property)
            self._raw_folder = self.cg_folder + "_raw"

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
        are not bundled for products of more than three irreps: recompute those with
        force_computation=True to get them.
        """
        if self._raw_cg is None:
            folder = Path(self._raw_folder)
            if not folder.is_dir():
                raise FileNotFoundError(
                    f"the raw CG blocks of {''.join(str(ir) for ir in self.chosen_irreps)} are not stored in "
                    f"{folder} (they are not bundled for products of more than three irreps): "
                    "recompute them with force_computation=True"
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
    def h4_mat(self) -> list[list[np.ndarray]]:
        """
        Matrices of all the elements of H(4) in every irrep: h4_mat[ir][i] is the matrix of the
        i-th element in the irrep rep_label_list[ir].

        They are needed only to compute new CG coefficients.  Generated matrices (force_h4gen,
        or no h4_ele folder) are set in __init__; stored ones are read on first access, through
        _load_h4_matrices, which reads the files of a folder only once per session.  The
        arrays are then shared with every other cg_calc of the same folder, and read-only.
        """
        if self._h4_mat is None:
            # A list of lists of this instance (the type that the generation in __init__ builds),
            # holding the shared, cached arrays.
            self._h4_mat = [list(rep) for rep in _load_h4_matrices(self._h4_ele_path)]
        return self._h4_mat

    @h4_mat.setter
    def h4_mat(self, value: list[list[np.ndarray]]) -> None:
        self._h4_mat = value

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

        Path(self.cg_pdf_folder).mkdir(parents=True, exist_ok=True)
        doc_name = ''.join([str(ir) for ir in self.chosen_irreps])

        doc = Document(default_filepath=f'{doc_name}.tex', documentclass='article')
        doc.preamble.append(Command("title", title))
        doc.preamble.append(Command("author", author))
        doc.preamble.append(Command("date", NoEscape(r"\today")))
        doc.append(NoEscape(r"\maketitle"))
        doc.append(NoEscape(r"\newpage"))

        for k, v in self.cg_dict.items():
            section = Section(str(rep_label_list[k]), numbering=False)
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


def get_multiplicities(*irreps: tuple[int, int]) -> list[int]:
    """
    Compute the multiplicity of each H(4) irrep in the given tensor product.

    Parameters
    ----------
    *irreps : tuple[int,int]
        Irreps of H(4), e.g. (4,1), (4,4), …

    Returns
    -------
    list[int]
        Length-20 list; entry i is the multiplicity of the i-th irrep.
    """
    mult_list = []
    for ir in range(n_rep):
        mul = 0
        for ic, class_ord in enumerate(class_orders):
            tmp = class_ord * char_table[ir, ic]
            for k in irreps:
                tmp *= char_table[irrep_index[k], ic]
            mul += tmp
        mul /= n_ele_h4
        mult_list.append(int(mul))
    return mult_list


def latex_print_multiplicities(
    *irreps: tuple[int, int], display_eq: bool = False, return_str: bool = True
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
    """
    mult_list = get_multiplicities(*irreps)
    out_str = r"$"

    for i, (a, b) in enumerate(irreps):
        out_str += f"\\tau^{{\\left({a}\\right)}}_{{{b}}}"
        if i != len(irreps) - 1:
            out_str += " \\otimes "

    out_str += " = "

    non_zero = [(i, mult) for i, mult in enumerate(mult_list) if mult != 0]
    for j, (i, mult) in enumerate(non_zero):
        out_str += f"{mult if mult > 1 else ''} \\, {rep_latex_names[i]}"
        if j != len(non_zero) - 1:
            out_str += " \\oplus "

    out_str += "$"

    if display_eq and _IPYTHON_AVAILABLE:
        display(MathDisplay(out_str))

    return out_str if return_str else None


@functools.cache
def _load_h4_matrices(folder: str) -> tuple[tuple[np.ndarray, ...], ...]:
    """
    Read the matrices of all the elements of H(4), in every irrep, from *folder*.

    The folder holds one sub-folder per irrep, named after its label (e.g. '(4, 1)'), with
    the files '0.npy' … '383.npy' written by cg_calc: 20 × 384 = 7680 small files.
    functools.cache stores the result of each call, keyed by *folder*: a later call with the
    same folder returns the same objects without reading the files again.  cg_calc therefore
    passes an absolute path (the same relative folder seen from two working directories is
    two different folders), and calls _load_h4_matrices.cache_clear() after writing new
    matrices into a folder.

    The arrays are shared by every cg_calc that uses the folder: they are made read-only, so
    that changing one in place raises an error instead of silently changing it for all the
    others.  The containers are tuples for the same reason.

    Parameters
    ----------
    folder : str
        Absolute path of an h4_ele folder.

    Returns
    -------
    tuple of tuple of ndarray
        Entry [ir][i] is the matrix of the i-th element in the irrep rep_label_list[ir].
    """
    matrices = []
    for label in rep_label_list:
        rep_matrices = []
        for i in range(n_ele_h4):
            with open(f'{folder}/{label}/{i}.npy', 'rb') as f:
                ele = np.load(f, allow_pickle=True)
            ele.setflags(write=False)  # read-only: see above
            rep_matrices.append(ele)
        matrices.append(tuple(rep_matrices))
    return tuple(matrices)


def _element_matrices(h4_mat: list[list[np.ndarray]], irep: int) -> np.ndarray:
    """The matrices of all the elements of H(4) in the irrep of index *irep*, as an (n_ele_h4, d, d) array."""
    d = rep_dim_list[irep]
    return np.array([np.reshape(ele, (d, d)) for ele in h4_mat[irep]], dtype=float)


def _product_matrices(h4_mat: list[list[np.ndarray]], irreps: tuple[tuple[int, int], ...]) -> np.ndarray:
    """T(h) = D_1(h) ⊗ … ⊗ D_n(h) for every element h of H(4), as an (n_ele_h4, dim, dim) array."""
    factors = [_element_matrices(h4_mat, irrep_index[rep]) for rep in irreps]
    dim = int(np.prod([D.shape[1] for D in factors]))
    T = np.empty((n_ele_h4, dim, dim))
    for ih in range(n_ele_h4):
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


def _c_parity_permutation(irreps: tuple[tuple[int, int], ...]) -> np.ndarray | None:
    """Row permutation of a CG block that reverses the order of the factors after the first.

    For an operator ψ̄ Γ D_{μ₂} … D_{μₙ} ψ, with Γ transforming as irreps[0] and each
    derivative as one of irreps[1:], charge conjugation reverses the derivative indices
    (see C_parity in moments_operator): on the rows of a CG block this is the returned
    permutation, up to an overall sign.  Returns None when there is nothing to reverse
    (fewer than two factors after the first) or when the factors after the first are
    not all the same irrep (then the reversal is not a symmetry of the product).
    """
    irreps = tuple(tuple(irrep) for irrep in irreps)
    if len(irreps) < 3 or len(set(irreps[1:])) != 1:
        return None
    dims = [irrep_dim[irrep] for irrep in irreps]
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

    # column 0 fixes the copy: its linear forms must span exactly `multiplicity` dimensions
    L0 = L[:, 0, :]
    tol = threshold * max(1.0, np.abs(L0).max())
    rank = np.linalg.matrix_rank(L0, tol=tol) if len(symbols) else 0
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
    part_ranks = [np.linalg.matrix_rank(part[:, 0, :], tol=tol) if len(symbols) else 0 for part, _ in parts]
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
