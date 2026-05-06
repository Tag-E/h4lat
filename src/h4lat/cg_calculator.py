######################################################
## cg_calculator.py  (h4lat package module)        ##
## originally cgh4_calculator.py                   ##
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
symbolically.  The resulting free parameters (gauge freedom within degenerate
subspaces) are fixed by the prescription in CGmat_from_block.

Main entry point: cg_calc.
Module-level helpers: get_multiplicities, latex_print_multiplicities.
"""


######################## Library Imports ################################

import itertools as it
import sys
import time
from importlib.resources import files as _resource_files
from pathlib import Path
from statistics import mode

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
# ordered to match rep_label_list.

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
a = np.array([[1, 0], [0, 1]], dtype=float)
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
        [0, 0, 1, 0, 0, 1],
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
        [0, 0, 1, 0, 0, 1],
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
        [0, 0, 1, 0, 0, 1],
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
        [0, 0, -1, 0, 0, 1],
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
        enforce_symmetry: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Initialise the CG-coefficient calculator for a tensor product of H(4) irreps.

        Parameters
        ----------
        *kwarg : tuple[int,int]
            Irreps of H(4), e.g. (4,1), (4,4), (6,1), …
        cgdatabase : str or None
            Path to the CG-coefficient database.  Defaults to the bundled database.
        force_computation : bool
            Recompute CG coefficients even if the database entry already exists.
        force_h4gen : bool
            Recompute all H(4) matrix representations even if they are cached.
        prescription_changed : bool
            Re-apply CGmat_from_block to the raw symbolic matrices and overwrite
            the stored numerical matrices.
        enforce_symmetry : bool
            Zero out entries incompatible with alpha/gamma symmetry (default True).
        verbose : bool
            Print progress messages.
        """

        self.chosen_irreps = kwarg

        # --- data paths -----------------------------------------------------------
        # When force_h4gen=True the user wants to regenerate; write to cwd.
        # Otherwise read from the bundled package data (read-only).
        if force_h4gen:
            self.h4_ele_folder = 'h4_ele'
        else:
            self.h4_ele_folder = _BUNDLED_H4_ELE

        # CG database: user-supplied path takes priority, else bundled data.
        if cgdatabase is None:
            self.cg_database_folder = _BUNDLED_CG_DATABASE
        else:
            self.cg_database_folder = cgdatabase
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
        self.cg_folder = self.cg_database_folder + '/' + ''.join([str(ir) for ir in self.chosen_irreps])

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
            # A_{ij} labelling the remaining gauge freedom.
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
                    if enforce_symmetry is True:
                        symm_mat = force_symmetry_gamma(
                            rounded_res[:, d : d + rep_dim_list[irep]],
                            rep_label_list[irep],
                            list(self.chosen_irreps),
                            remove_bad_monom=False,
                        )
                        symm_mat = force_symmetry_alpha(
                            symm_mat,
                            rep_label_list[irep],
                            list(self.chosen_irreps),
                            remove_bad_monom=False if self.mul_list[irep] == 1 else True,
                        )
                        symm_mat = force_symmetry_beta(
                            symm_mat,
                            rep_label_list[irep],
                            list(self.chosen_irreps),
                            remove_bad_monom=True,
                            multiplicity=self.mul_list[irep],
                        )
                        self.cg_dict[irep].append(CGmat_from_block(symm_mat, m, mul))
                    else:
                        self.cg_dict[irep].append(CGmat_from_block(rounded_res[:, d : d + rep_dim_list[irep]], m, mul))
                    self.raw_cg[(irep, m)] = rounded_res[:, d : d + rep_dim_list[irep]]
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
            p = Path(self.cg_folder + "_raw").glob('**/*')
            files = [x for x in p if x.is_file()]

            for _, file in enumerate(files):
                irep = int(file.name.split(".")[0].split("_")[0])
                m = int(file.name.split(".")[0].split("_")[1])
                with open(f'{self.cg_folder}_raw/{file.name}', 'rb') as f:
                    self.raw_cg[(irep, m)] = np.load(f, allow_pickle=True)

                if prescription_changed:
                    with open(f'{self.cg_folder}/{irep}_{m}.npy', 'wb') as f:
                        if enforce_symmetry is True:
                            symm_mat = force_symmetry_gamma(
                                self.raw_cg[(irep, m)],
                                rep_label_list[irep],
                                list(self.chosen_irreps),
                                remove_bad_monom=False,
                            )
                            symm_mat = force_symmetry_alpha(
                                symm_mat,
                                rep_label_list[irep],
                                list(self.chosen_irreps),
                                remove_bad_monom=False if self.mul_list[irep] == 1 else True,
                            )
                            symm_mat = force_symmetry_beta(
                                symm_mat,
                                rep_label_list[irep],
                                list(self.chosen_irreps),
                                remove_bad_monom=True,
                                multiplicity=self.mul_list[irep],
                            )
                            np.save(f, CGmat_from_block(symm_mat, m, self.mul_list[irep]))
                        else:
                            np.save(f, CGmat_from_block(self.raw_cg[(irep, m)], m, self.mul_list[irep]))

            self.cg_dict = {}
            p = Path(self.cg_folder).glob('**/*')
            files = [x for x in p if x.is_file()]

            for _, file in enumerate(files):
                irep = int(file.name.split(".")[0].split("_")[0])
                with open(f'{self.cg_folder}/{file.name}', 'rb') as f:
                    if irep not in self.cg_dict.keys():
                        self.cg_dict[irep] = [np.load(f, allow_pickle=True)]
                    else:
                        self.cg_dict[irep].append(np.load(f, allow_pickle=True))

            self.cg_dict = {k: self.cg_dict[k] for k in sorted(list(self.cg_dict.keys()))}

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


def CGmat_from_block(block: np.ndarray, m: int = 0, mul: int = 1, gram_schmidt: bool = False) -> np.ndarray:
    """
    Convert a symbolic CG block (output of the projection formula) to a
    numerical matrix by choosing one free parameter = 1 and the rest = 0.

    Parameters
    ----------
    block : ndarray
        Symbolic matrix from the CG projection formula.
    m : int
        Index (0 … mul-1) selecting which multiplicity copy to extract.
    mul : int
        Total multiplicity of the irrep.
    gram_schmidt : bool
        If True, Gram–Schmidt orthogonalise the columns.

    Notes
    -----
    The symbolic block from the projection formula contains one or more free
    sympy monomials A_{ij} (the gauge freedom within degenerate subspaces).
    The gauge is fixed by setting one monomial to 1 and the rest to 0.  For
    multiplicity 1, the monomial appearing in the most rows is chosen
    (maximally constrained selection).  For multiplicity > 1, a mode-based
    rule distributes the degrees of freedom evenly across the mul copies.
    Each column is subsequently normalised so its leading non-zero entry is +1.
    """
    mat = block.copy()

    if len(np.shape(mat)) == 1:
        mat = np.reshape(mat, (np.shape(mat)[0], 1))

    monom_counts = {}
    monom_rows = {}
    for i in range(np.shape(mat)[0]):
        entry = sym.Add.make_args(mat[i, 0])
        for element in entry:
            monom = element.as_coeff_Mul()[1]
            if (monom not in monom_counts) and (not isinstance(monom, sym.core.numbers.One)):
                monom_counts[monom] = 1
                monom_rows[monom] = [i]
            elif (monom in monom_counts) and (not isinstance(monom, sym.core.numbers.One)):
                monom_counts[monom] += 1
                monom_rows[monom].append(i)

    entry_sizes = np.zeros(shape=np.shape(mat), dtype=int)
    monoms_in_colum = {}
    for j in range(np.shape(mat)[1]):
        monoms_in_colum[j] = []
        for i in range(np.shape(mat)[0]):
            entry = sym.Add.make_args(mat[i, j])
            entry_sizes[i, j] = len(entry)
            for element in entry:
                monom = element.as_coeff_Mul()[1]
                if monom not in monoms_in_colum[j]:
                    monoms_in_colum[j].append(monom)

    for k in monom_counts.keys():
        for j in range(np.shape(mat)[1]):
            if k not in monoms_in_colum[j]:
                monom_counts[k] = -1

    monom_dict = {}
    for j in range(np.shape(mat)[1]):
        for i in range(np.shape(mat)[0]):
            entry = sym.Add.make_args(mat[i, j])
            for element in entry:
                monom = element.as_coeff_Mul()[1]
                if (monom not in monom_dict) and (not isinstance(monom, sym.core.numbers.One)):
                    monom_dict[monom] = 0.0

    target = mul
    newmat = mat.copy()

    if mul == 1:
        index = max(monom_counts, key=monom_counts.get)
        index_list = [max(monom_counts, key=monom_counts.get)]
        for _ in range(target - 1):
            for k in monom_counts:
                if monom_rows[k] == monom_rows[index]:
                    monom_counts[k] = -1
            index = max(monom_counts, key=monom_counts.get)
            index_list.append(max(monom_counts, key=monom_counts.get))
        monom_dict[index_list[m]] = 1.0
    else:
        max_len = mode([e for e in entry_sizes.flatten() if e != 1])
        for j in range(np.shape(mat)[1]):
            for i in range(np.shape(mat)[0]):
                if entry_sizes[i, j] < max_len:
                    newmat[i, j] = sym.core.numbers.Zero()

        index_list = []
        occurences_mode = mode([v for v in monom_counts.values() if v > 0])
        for k in monom_counts:
            if monom_counts[k] == occurences_mode:
                index_list.append(k)
        monom_dict[index_list[int((m + 1 / 2) * len(index_list) / mul)]] = 1.0

    for j in range(np.shape(mat)[1]):
        for i in range(np.shape(mat)[0]):
            tmp = newmat[i, j]
            for k, v in monom_dict.items():
                tmp = tmp.subs(k, v)
            newmat[i, j] = tmp

    for j in range(np.shape(newmat)[1]):
        index = (newmat[:, j] != 0).argmax(axis=0)
        norm = newmat[index, j]
        newmat[:, j] /= np.abs(norm)

    if gram_schmidt is True:
        for j in range(np.shape(newmat)[1]):
            for jp in range(j):
                newmat[:, j] = (
                    newmat[:, j]
                    - np.dot(newmat[:, jp], newmat[:, j]) / np.dot(newmat[:, jp], newmat[:, jp]) * newmat[:, jp]
                )
            newmat[:, j] /= newmat[(newmat[:, j] != 0).argmax(axis=0), j]

    newmat = np.asarray(newmat).astype(np.float64)
    treshold = 10 ** (-10)

    for j in range(np.shape(newmat)[1]):
        index = np.abs(newmat[:, j]).argmax(axis=0)
        norm = np.abs(newmat[index, j])
        if norm > 1 / treshold:
            newmat[:, j] /= norm

    newmat[np.abs(newmat) < treshold] = 0.0

    return newmat


def force_symmetry_gamma(
    cg_symbolic_mat: np.ndarray,
    irrep_mat: tuple[int, int],
    irrep_indices: list[tuple[int, int]],
    remove_bad_monom: bool = False,
) -> np.ndarray:
    """Zero entries of the symbolic CG matrix incompatible with gamma symmetry."""
    generator_gamma_mat = gamma_list[irrep_index[irrep_mat]]
    if generator_gamma_mat.ndim == 1:
        generator_gamma_mat = np.array([generator_gamma_mat])

    dim_list = [irrep[0] for irrep in irrep_indices]
    nrows, ncols = np.shape(cg_symbolic_mat)
    bad_monoms = []
    out_mat = cg_symbolic_mat.copy()

    for i_col in range(ncols):
        desired_gamma_parity = generator_gamma_mat[i_col, i_col]
        for i_row, indices in enumerate(it.product(*[range(dim) for dim in dim_list])):
            actual_parity = 1
            for j, ind in enumerate(indices):
                actual_parity *= gamma_list[irrep_index[irrep_indices[j]]][ind, ind]
            if actual_parity != desired_gamma_parity:
                monom = sym.Add.make_args(out_mat[i_row, i_col])[0].as_coeff_Mul()[1]
                if monom not in bad_monoms:
                    bad_monoms.append(monom)
                out_mat[i_row, i_col] = sym.core.numbers.Zero()

    if remove_bad_monom is True:
        for i_row in range(nrows):
            for i_col in range(ncols):
                for monom in bad_monoms:
                    out_mat[i_row, i_col] = out_mat[i_row, i_col].subs({monom: 0})

    return out_mat


def force_symmetry_alpha(
    cg_symbolic_mat: np.ndarray,
    irrep_mat: tuple[int, int],
    irrep_indices: list[tuple[int, int]],
    remove_bad_monom: bool = False,
) -> np.ndarray:
    """Zero entries of the symbolic CG matrix incompatible with alpha symmetry."""
    generator_alpha_mat = alpha_list[irrep_index[irrep_mat]]
    if generator_alpha_mat.ndim == 1:
        generator_alpha_mat = np.array([generator_alpha_mat])

    dim_list = [irrep[0] for irrep in irrep_indices]
    nrows, ncols = np.shape(cg_symbolic_mat)
    bad_monoms = []
    out_mat = cg_symbolic_mat.copy()

    row_ind_dict = {indices: i_row for i_row, indices in enumerate(it.product(*[range(dim) for dim in dim_list]))}

    for i_col in range(ncols):
        transf_col = np.where(generator_alpha_mat[:, i_col] != 0)[0][0]
        for i_row, indices in enumerate(it.product(*[range(dim) for dim in dim_list])):
            transf_indices = tuple(
                [
                    np.where(alpha_list[irrep_index[irrep_indices[j]]][:, ind] != 0)[0][0]
                    for j, ind in enumerate(indices)
                ]
            )
            transf_row = row_ind_dict[transf_indices]
            if out_mat[transf_row, transf_col] == sym.core.numbers.Zero():
                monom = sym.Add.make_args(out_mat[i_row, i_col])[0].as_coeff_Mul()[1]
                if monom not in bad_monoms:
                    bad_monoms.append(monom)
                out_mat[i_row, i_col] = sym.core.numbers.Zero()

    if remove_bad_monom is True:
        for i_row in range(nrows):
            for i_col in range(ncols):
                for monom in bad_monoms:
                    out_mat[i_row, i_col] = out_mat[i_row, i_col].subs({monom: 0})

    return out_mat


def force_symmetry_beta(
    cg_symbolic_mat: np.ndarray,
    irrep_mat: tuple[int, int],
    irrep_indices: list[tuple[int, int]],
    remove_bad_monom: bool = False,
    multiplicity: int = 1,
) -> np.ndarray:
    """Zero entries of the symbolic CG matrix incompatible with beta symmetry."""
    generator_beta_mat = beta_list[irrep_index[irrep_mat]]
    if generator_beta_mat.ndim == 1:
        generator_beta_mat = np.array([generator_beta_mat])

    dim_list = [irrep[0] for irrep in irrep_indices]
    nrows, ncols = np.shape(cg_symbolic_mat)
    bad_monoms = []
    out_mat = cg_symbolic_mat.copy()

    row_ind_dict = {indices: i_row for i_row, indices in enumerate(it.product(*[range(dim) for dim in dim_list]))}

    for i_col in range(ncols):
        transf_col_list = np.where(generator_beta_mat[:, i_col] != 0)[0]
        for i_row, indices in enumerate(it.product(*[range(dim) for dim in dim_list])):
            remove_entry = True
            transf_indices_generating_list = [
                np.where(beta_list[irrep_index[irrep_indices[j]]][:, ind] != 0)[0] for j, ind in enumerate(indices)
            ]
            for non_zero_transf_ind in it.product(*[range(len(e)) for e in transf_indices_generating_list]):
                transf_indices = tuple(
                    [e[non_zero_transf_ind[i_ind]] for i_ind, e in enumerate(transf_indices_generating_list)]
                )
                transf_row = row_ind_dict[transf_indices]
                for transf_col in transf_col_list:
                    if out_mat[transf_row, transf_col] != sym.core.numbers.Zero():
                        remove_entry = False

            if remove_entry is True:
                if multiplicity == 1:
                    monom = sym.Add.make_args(out_mat[i_row, i_col])[0].as_coeff_Mul()[1]
                    if monom not in bad_monoms:
                        bad_monoms.append(monom)
                else:
                    monom_list = [e.as_coeff_Mul()[1] for e in sym.Add.make_args(out_mat[i_row, i_col])]
                    for monom in monom_list:
                        if monom not in bad_monoms:
                            bad_monoms.append(monom)
                out_mat[i_row, i_col] = sym.core.numbers.Zero()

    if remove_bad_monom is True:
        for i_row in range(nrows):
            for i_col in range(ncols):
                for monom in bad_monoms:
                    out_mat[i_row, i_col] = out_mat[i_row, i_col].subs({monom: 0})

    return out_mat


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

    which = "both"
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
