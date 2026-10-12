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
target irrep matrices over all group elements extracts the CG vectors
symbolically.  The resulting free parameters (freedom within degenerate
subspaces) are fixed by CGmat_from_block, which picks one independent copy
of the irrep for each unit of multiplicity.

Main entry point: cg_calc.
Module-level helpers: get_multiplicities, latex_print_multiplicities.
"""


######################## Library Imports ################################

import itertools as it
import sys
import time
from importlib.resources import files as _resource_files
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
        *kwarg: tuple[int, int],
        cgdatabase: str | None = None,
        force_computation: bool = False,
        force_h4gen: bool = False,
        prescription_changed: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Initialise the CG-coefficient calculator for a tensor product of H(4) irreps.

        Parameters
        ----------
        *kwarg : tuple[int,int]
            Irreps of H(4), e.g. (4,1), (4,4), (6,1), …
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
            database (see cgdatabase).
        verbose : bool
            Print progress messages.
        """

        self.chosen_irreps = kwarg

        # --- data paths -----------------------------------------------------------
        # The bundled package data are only read, never written: regenerated H(4)
        # matrices and new CG coefficients go to the local folders (_LOCAL_*).
        if force_h4gen or not Path(_BUNDLED_H4_ELE).exists():
            self.h4_ele_folder = _LOCAL_H4_ELE
        else:
            self.h4_ele_folder = _BUNDLED_H4_ELE

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

        elif Path(self.h4_ele_folder).exists():

            if verbose:
                print("\nLoading matrix representations for all the elements of H(4) ...\n")

            self.h4_mat = [[] for _ in range(self.n_rep)]
            for ir in range(self.n_rep):
                rep_folder = self.h4_ele_folder + f"/{str(self.rep_label_list[ir])}"
                for i in range(self.n_ele_h4):
                    with open(f'{rep_folder}/{i}.npy', 'rb') as f:
                        self.h4_mat[ir].append(np.load(f, allow_pickle=True))

        self.mul_list = get_multiplicities(*kwarg)

        if not Path(self.cg_folder).exists() or force_computation:

            if verbose:
                print(f"\nComputing the cg coefficients for the tensor product {self.chosen_irreps} ...\n")

            dim = 1
            for rep in self.chosen_irreps:
                dim *= irrep_dim[rep]

            A = sym.MatrixSymbol('A', dim, dim)
            res_mat = np.zeros((dim, dim))

            if verbose:
                print("\nLooping over group elements...\n")

            time_start_loop = time.time()
            totalT_mat_construction = 0
            totalT_mat_mul = 0

            # Sakata projection formula (J. Math. Phys. 15, 1702, 1974):
            #   res_mat += T(h) @ A @ B(h)^T  for each h ∈ H(4),
            # where T(h) is the tensor-product representation of h, B(h) is
            # the block-diagonal target representation (one block per irrep
            # copy in the decomposition), and A is a symbolic dim×dim matrix.
            # After summation, each column of res_mat is a linear form in the
            # A_{ij}; the coefficients are the CG matrix entries, with the
            # A_{ij} labelling the remaining freedom.
            for ih in tqdm(range(self.n_ele_h4)):
                time_before_construction = time.time()

                Tp = np.eye(1)
                for rep in kwarg:
                    Tp = np.kron(Tp, self.h4_mat[irrep_index[rep]][ih])

                nblocks = np.sum(self.mul_list)
                rows = [[] for _ in range(nblocks)]

                rep_to_use = []
                for irep, mul in enumerate(self.mul_list):
                    for _ in range(mul):
                        rep_to_use.append(irep)

                for ib in range(nblocks):
                    rep_i = rep_to_use[ib]
                    for jb in range(nblocks):
                        rep_j = rep_to_use[jb]
                        if ib == jb:
                            rows[ib].append(self.h4_mat[rep_i][ih])
                        else:
                            rows[ib].append(np.zeros((rep_dim_list[rep_i], rep_dim_list[rep_j])))

                Bd = np.block(rows)

                time_before_mul = time.time()
                res_mat = res_mat + Tp @ A @ Bd.T
                time_after_mul = time.time()

                totalT_mat_construction += time_before_mul - time_before_construction
                totalT_mat_mul += time_after_mul - time_before_mul

            time_end_loop = time.time()

            # Zero out floating-point noise: coefficients with |c| < eps are
            # rounding artefacts; |c| > infth flags a divergent symbolic entry.
            rounded_res = np.empty((dim, dim), dtype=object)
            eps = 10 ** (-10)
            infth = 10**20

            for i in range(dim):
                for j in range(dim):
                    entry = sym.Add.make_args(res_mat[i, j])
                    new_entry = 0
                    for element in entry:
                        coeff = element.as_coeff_Mul()[0]
                        monom = element.as_coeff_Mul()[1]
                        if np.abs(coeff) < eps or np.abs(coeff) > infth:
                            coeff = 0
                        new_entry = new_entry + coeff * monom
                    rounded_res[i, j] = new_entry

            time_end_rounding = time.time()

            self.cg_dict = {}
            self.raw_cg = {}
            d = 0
            for irep, mul in enumerate(self.mul_list):
                for m in range(mul):
                    if irep not in self.cg_dict.keys():
                        self.cg_dict[irep] = []
                    self.raw_cg[(irep, m)] = rounded_res[:, d : d + rep_dim_list[irep]]
                    self.cg_dict[irep].append(CGmat_from_block(self.raw_cg[(irep, m)], mul, m))
                    d += rep_dim_list[irep]

            time_end_evaluation = time.time()

            totalT_loop = time_end_loop - time_start_loop
            totalT_rounding = time_end_rounding - time_end_loop
            totalT_evaluating = time_end_evaluation - time_end_rounding
            totalT = totalT_loop + totalT_rounding + totalT_evaluating

            if verbose:
                print(
                    f"\nH(4) Loop time: {round(totalT_loop,2)} s\nMatrix Construction Time: {round(totalT_mat_construction/totalT_loop*100,2)}%\nMatrix Multiplication Time: {round(totalT_mat_mul/totalT_loop*100,2)}%\n"
                )
                print(
                    f"\nTotal Time (Loop+Rounding+Evaluation): {round(totalT,2)} s\nLoop: {round(totalT_loop/totalT*100,2)}%\nRounding: {round(totalT_rounding/totalT*100,2)}%\nEvaluation: {round(totalT_evaluating/totalT*100,2)}%\n"
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

            self.raw_cg = {}
            # A changed prescription is saved like a new result: to the output database,
            # together with the raw matrices it is derived from.
            out_folder = self.cg_database_folder + '/' + product_name
            if prescription_changed:
                Path(out_folder).mkdir(parents=True, exist_ok=True)
                Path(out_folder + "_raw").mkdir(parents=True, exist_ok=True)

            for file in _sorted_cg_files(self.cg_folder + "_raw"):
                irep, m = (int(x) for x in file.stem.split("_"))
                self.raw_cg[(irep, m)] = np.load(file, allow_pickle=True)

                if prescription_changed:
                    if out_folder != self.cg_folder:
                        with open(f'{out_folder}_raw/{irep}_{m}.npy', 'wb') as f:
                            np.save(f, self.raw_cg[(irep, m)])
                    with open(f'{out_folder}/{irep}_{m}.npy', 'wb') as f:
                        np.save(f, CGmat_from_block(self.raw_cg[(irep, m)], self.mul_list[irep], m))

            if prescription_changed:
                self.cg_folder = out_folder

            # Files come in (irep, m) order, so cg_dict[irep][m] is always '{irep}_{m}.npy'
            # and the keys are sorted.
            self.cg_dict = {}
            for file in _sorted_cg_files(self.cg_folder):
                irep = int(file.stem.split("_")[0])
                self.cg_dict.setdefault(irep, []).append(np.load(file, allow_pickle=True))

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


def get_multiplicities(*kwarg: tuple[int, int]) -> list[int]:
    """
    Compute the multiplicity of each H(4) irrep in the given tensor product.

    Parameters
    ----------
    *kwarg : tuple[int,int]
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
            for k in kwarg:
                tmp *= char_table[irrep_index[k], ic]
            mul += tmp
        mul /= n_ele_h4
        mult_list.append(int(mul))
    return mult_list


def latex_print_multiplicities(
    *kwarg: tuple[int, int], display_eq: bool = False, return_str: bool = True
) -> str | None:
    """
    Return (and optionally display) the LaTeX string for the irrep decomposition.

    Parameters
    ----------
    *kwarg : tuple[int,int]
        Irreps forming the tensor product.
    display_eq : bool
        If True and inside a Jupyter notebook, render the equation inline.
    return_str : bool
        If True, return the LaTeX string.
    """
    mult_list = get_multiplicities(*kwarg)
    out_str = r"$"

    for i, (a, b) in enumerate(kwarg):
        out_str += f"\\tau^{{\\left({a}\\right)}}_{{{b}}}"
        if i != len(kwarg) - 1:
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


def CGmat_from_block(
    block: np.ndarray,
    multiplicity: int,
    multiplicity_index: int,
    normalize: bool = True,
    threshold: float = 1e-10,
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

    Returns
    -------
    ndarray
        Float matrix of the same shape as *block*, whose columns span one copy of
        the irrep.

    Raises
    ------
    ValueError
        If the free parameters of the block do not have *multiplicity* degrees of
        freedom, i.e. the block is not a projection onto *multiplicity* copies.

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

    With multiplicity 1 there is a single copy, the same for any A up to a factor;
    the factor is fixed by the first non-zero entry of column 0 being +1.
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

    # reference rows: the first rows of column 0 (in row order) with independent linear forms
    ref_rows = []
    for r in range(nrows):
        if np.linalg.matrix_rank(L0[ref_rows + [r]], tol=tol) > len(ref_rows):
            ref_rows.append(r)
            if len(ref_rows) == multiplicity:
                break

    # the A of minimal norm with reference row i of column 0 equal to δ_{i, multiplicity_index}
    target = np.zeros(multiplicity)
    target[multiplicity_index] = 1.0
    a = np.linalg.pinv(L0[ref_rows]) @ target

    # the copy: the block evaluated at that A
    newmat = L @ a
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
    n = 2
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print(f"\nSpecified n was {sys.argv[1]}, proceeding with n={n}\n")

    which = "all"
    if len(sys.argv) > 2:
        if str(sys.argv[2]) in ["vector", "axial", "tensor", "all"]:
            which = str(sys.argv[2])

    chosen_irreps_vector = [(4, 1)]
    chosen_irreps_axial = [(4, 4)]
    chosen_irreps_tensor = [(6, 1)]
    while len(chosen_irreps_vector) < n:
        chosen_irreps_vector.append((4, 1))
        chosen_irreps_axial.append((4, 1))
        chosen_irreps_tensor.append((4, 1))

    if which in ("all", "vector"):
        cg_v = cg_calc(*chosen_irreps_vector, force_computation=True)
        cg_v.latex_print()
    if which in ("all", "axial"):
        cg_a = cg_calc(*chosen_irreps_axial, force_computation=True)
        cg_a.latex_print()
    if which in ("all", "tensor"):
        cg_t = cg_calc(*chosen_irreps_tensor, force_computation=True)
        cg_t.latex_print()
