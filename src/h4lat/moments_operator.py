######################################################
## moments_operator.py  (h4lat package module)      ##
## created by Emilio Taggi - 2025/01/15             ##
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
moments_operator.py — lattice moments operators built from H(4) CG coefficients.

An ``Operator`` object represents a single irreducible lattice operator of the form

    O^X_{μ₁ μ₂ … μₙ}  =  Σ_{i₁,…,iₙ} c_{i₁…iₙ}  Γ^X_{i₁}  D_{i₂} … D_{iₙ}

where Γ^X is the Dirac structure (V: γ_μ, A: γ_μ γ₅, T: σ_{μν}), the c_{i₁…iₙ}
are Clebsch-Gordan coefficients stored in ``cgmat``, and the derivatives D are 
left minus right acting covariant derivatives.

The kinematic factor K(p, m) is the ratio of the tree-level three-point correlator
to its kinematic prefactor, allowing lattice matrix elements to be normalised to
physical parton distribution moments.  The class computes K symbolically via SymPy
and caches the result.

Reference:
    Göckeler et al., Phys. Rev. D 54, 5705 (1996), arXiv:hep-lat/9602029.
    https://doi.org/10.1103/PhysRevD.54.5705
"""

import itertools as it
import sys
from pathlib import Path
from typing import Self

import numpy as np
import sympy as sym
from sympy import lambdify
from sympy.tensor.array.expressions import ArraySymbol
from tqdm import tqdm

try:
    from IPython.display import Math, display

    _IPYTHON_AVAILABLE = True
except ImportError:
    _IPYTHON_AVAILABLE = False

try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

try:
    import gvar as gv  # noqa: F401 — guard import; alias used by callers via _GVAR_AVAILABLE

    _GVAR_AVAILABLE = True
except ImportError:
    _GVAR_AVAILABLE = False

try:
    import h5py  # noqa: F401 — guard import; availability flag used downstream

    _H5PY_AVAILABLE = True
except ImportError:
    _H5PY_AVAILABLE = False

from .cg_calculator import (
    _BUNDLED_OPERATOR_DATABASE,
    cg_calc,
    get_multiplicities,
    irrep_index,
    rep_label_list,
)
from .kinematic_data import (
    E,
    Gamma_pol,
    I,
    Id_4,
    den_K,
    gamma5,
    gamma_mu,
    mN,
    numerics_to_latex_conv,
    p1,
    p2,
    p3,
    p_mu,
    pslash,
)
from .utilities import all_equal, parity

######################## Main Class #####################################


class Operator:
    """
    A single irreducible lattice moments operator characterised by its
    Clebsch-Gordan coefficient tensor and Dirac structure.

    The ``cgmat`` is an n-index NumPy array of shape (4,)×n, where n is the
    total number of Lorentz indices (one for the Dirac structure plus one per
    derivative).  Each entry c_{i₁…iₙ} is the CG coefficient that selects the
    appropriate linear combination of elementary tensors.

    On construction the class computes and caches:
    * ``O``  — the symbolic operator expression (for display/LaTeX).
    * ``K``  — the tree-level kinematic factor as a SymPy expression.
    * ``C``  — C-parity (+1, -1, or 'mixed').
    * ``tr`` — whether the trace Tr[cgmat] vanishes.
    * ``symm`` — index-permutation symmetry ('Symmetric', 'Antisymmetric',
      'Mixed Symmetry', or '[Invalid Operator]' for tensorial cases).
    """

    def __init__(
        self,
        cgmat: np.ndarray,
        id: int | str,
        X: str,
        irrep: tuple[int, int] | None,
        block: int | None,
        index_block: int | None,
    ) -> None:
        """
        Parameters
        ----------
        cgmat : ndarray, shape (4,)*n
            CG coefficient tensor in the remapped form (see ``cg_remapping``).
        id : int or str
            Operator identifier (used in filenames and catalogues).
        X : str
            Dirac structure: ``'V'`` (vector γ_μ), ``'A'`` (axial γ_μγ₅),
            or ``'T'`` (tensor σ_{μν} = i/2 [γ_μ, γ_ν]).
        irrep : tuple[int,int] or None
            H(4) irrep label, e.g. ``(4,1)``.
        block : int or None
            Multiplicity index (1-based) within the irrep.
        index_block : int or None
            Column index (1-based) within the CG block.
        """
        self.cgmat = cgmat[:]
        self.id = id
        self.X = X
        self.irrep = irrep
        self.block = block
        self.index_block = index_block

        # Number of Lorentz indices; number of derivatives is one less for V/A,
        # two less for T (which carries two Dirac indices for σ_{μν}).
        self.n = cgmat.ndim
        self.nder = self.n - 1
        if X == 'T':
            self.nder -= 1

        # Build symbolic representations.  These are computed once and cached.
        self.O = symO_from_Cgmat(cgmat)
        self.K = Kfactor_from_diracO(diracO_from_cgmat(cgmat, X))

        # Discrete symmetry properties.
        self.C = C_parity(cgmat, X)
        self.tr = trace_symm(cgmat = cgmat, X = X)
        self.symm = index_symm(cgmat)
        if X == 'T':
            # For tensorial operators the first two indices carry σ_{μν}, which
            # is antisymmetric.  An operator that is symmetric under exchange of
            # those two indices is invalid (it would vanish by the antisymmetry
            # of σ_{μν}).
            if index_symm_index_fixed(cgmat, *(i for i in range(2, cgmat.ndim))) == "Symmetric":
                self.symm = "[Invalid Operator]"
            else:
                self.symm = index_symm_index_fixed(cgmat, 0)

        # The three-point correlator is real (imaginary) depending on the number
        # of derivatives and the Dirac structure.  We determine this from the
        # presence of the imaginary unit in the kinematic factor K.
        self.p3corr_is_real = (I in self.K.atoms()) if self.nder % 2 == 1 else (I not in self.K.atoms())

        # Pre-compute LaTeX strings for display.
        self.latex_O = latexO_from_diracO(self.O, self.X)
        self.latex_K = str(self.K).replace('**', '^').replace('*', '').replace('I', 'i')
        if '/' in self.latex_K:
            self.latex_K = "\\frac{" + self.latex_K.split('/')[0] + "}{ " + self.latex_K.split('/')[1] + "}"

    # ------------------------------------------------------------------
    # String representations
    # ------------------------------------------------------------------

    def __repr__(self):
        return str(self.O.simplify(rational=True))

    def __str__(self):
        return self.latex_O

    # ------------------------------------------------------------------
    # Arithmetic — operators form a vector space over the reals
    # ------------------------------------------------------------------

    def __add__(self, other: Self) -> Self:
        """Add two operators with matching X and index count."""
        if self.X != other.X or self.n != other.n:
            raise ValueError("Operator addition requires equal X and number of indices.")
        new_irrep = self.irrep if self.irrep == other.irrep else None
        new_block = self.block if (new_irrep and self.block == other.block) else None
        return Operator(
            cgmat=np.round(self.cgmat + other.cgmat, decimals=13),
            id=None,
            X=self.X,
            irrep=new_irrep,
            block=new_block,
            index_block=None,
        )

    def __sub__(self, other: Self) -> Self:
        """Subtract two operators with matching X and index count."""
        if self.X != other.X or self.n != other.n:
            raise ValueError("Operator subtraction requires equal X and number of indices.")
        new_irrep = self.irrep if self.irrep == other.irrep else None
        new_block = self.block if (new_irrep and self.block == other.block) else None
        return Operator(
            cgmat=np.round(self.cgmat - other.cgmat, decimals=13),
            id=None,
            X=self.X,
            irrep=new_irrep,
            block=new_block,
            index_block=None,
        )

    def __mul__(self, coefficient: float) -> Self:
        """Scale the operator by a numeric coefficient."""
        return Operator(
            cgmat=self.cgmat * coefficient,
            id=self.id,
            X=self.X,
            irrep=self.irrep,
            block=self.block,
            index_block=self.index_block,
        )

    def __rmul__(self, coefficient: float) -> Self:
        return self.__mul__(coefficient)

    def __truediv__(self, coefficient: float) -> Self:
        """Divide the operator by a numeric coefficient."""
        if coefficient == 0:
            raise ValueError("Cannot divide by 0.")
        return Operator(
            cgmat=self.cgmat / coefficient,
            id=self.id,
            X=self.X,
            irrep=self.irrep,
            block=self.block,
            index_block=self.index_block,
        )

    def __neg__(self) -> Self:
        return Operator(
            cgmat=self.cgmat * (-1),
            id=self.id,
            X=self.X,
            irrep=self.irrep,
            block=self.block,
            index_block=self.index_block,
        )

    # ------------------------------------------------------------------
    # Kinematic factor evaluation
    # ------------------------------------------------------------------

    def evaluate_K(self, m_value: float, E_value: float, p1_value: float, p2_value: float, p3_value: float) -> complex:
        """Return the complex value of K at the given kinematics.

        Parameters
        ----------
        m_value, E_value, p1_value, p2_value, p3_value : float
            Nucleon mass, energy, and spatial momentum components.

        Returns
        -------
        complex
        """
        return complex(self.K.evalf(subs={mN: m_value, E: E_value, p1: p1_value, p2: p2_value, p3: p3_value}))

    def evaluate_K_real(
        self, m_value: float, E_value: float, p1_value: float, p2_value: float, p3_value: float
    ) -> float:
        """Return the real-valued combination of K that normalises the 3-pt correlator.

        The three-point function ⟨N|O|N⟩ is proportional to (i^n_der) K, and
        by convention is either real or purely imaginary.  This method returns
        the relevant real part or imaginary part accordingly, so that the
        correlator can be processed using only real arithmetic.

        Parameters
        ----------
        m_value, E_value, p1_value, p2_value, p3_value : float

        Returns
        -------
        float
        """
        kin = (1j**self.nder) * complex(
            self.K.evalf(subs={mN: m_value, E: E_value, p1: p1_value, p2: p2_value, p3: p3_value})
        )
        return kin.real if self.p3corr_is_real else kin.imag

    def evaluate_K_gvar(self, m_value, E_value, p1_value, p2_value, p3_value):
        """Return K with Gaussian-variable uncertainty propagation (requires gvar).

        Same convention as ``evaluate_K_real`` but the inputs are ``gvar.GVar``
        objects so that statistical uncertainties on the kinematics are
        propagated analytically through the symbolic expression.

        Parameters
        ----------
        m_value, E_value, p1_value, p2_value, p3_value : gvar.GVar

        Returns
        -------
        gvar.GVar
        """
        if not _GVAR_AVAILABLE:
            raise ImportError("gvar is required for evaluate_K_gvar().")
        # Include the factor i^{n_der} that relates the matrix element to K,
        # and multiply by -i if the correlator is imaginary (so the returned
        # quantity is always real-valued in the gvar sense).
        kin = I**self.nder * self.K
        if self.p3corr_is_real is False:
            kin *= -I
        return lambdify([E, mN, p1, p2, p3], kin)(E_value, m_value, p1_value, p2_value, p3_value)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, folder: str) -> None:
        """Save the operator's cgmat to a .npy file inside *folder*.

        The filename encodes all operator metadata so that the operator can be
        fully reconstructed by ``Operator_from_file`` without any external
        catalogue.

        Parameters
        ----------
        folder : str
            Target directory (created if absent).
        """
        Path(folder).mkdir(parents=True, exist_ok=True)
        filename = (
            f"{folder}/operator_{self.id}_{self.n}_{self.X}"
            f"_{self.irrep[0]}_{self.irrep[1]}_{self.block}_{self.index_block}.npy"
        )
        np.save(filename, self.cgmat)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def display(self) -> None:
        """Render the LaTeX expression of the operator in a Jupyter notebook."""
        if not _IPYTHON_AVAILABLE:
            raise ImportError("IPython is required for display().")
        display(Math(self.latex_O))

    def short_latex_O(self, max_len: int) -> str:
        """Return a truncated LaTeX label, cut at the nearest +/- after *max_len* chars.

        Parameters
        ----------
        max_len : int
            Maximum number of characters before the cut point.

        Returns
        -------
        str
        """
        if type(max_len) is not int or max_len <= 0:
            raise ValueError(f"max_len must be a positive integer, got {max_len=}.")
        if len(self.latex_O) <= max_len:
            return self.latex_O
        short = self.latex_O[:max_len]
        for i in range(max_len, len(self.latex_O)):
            short += self.latex_O[i]
            if self.latex_O[i] in ["+", "-"]:
                short += "..."
                break
        return short


########### Auxiliary Functions #################


def symO_from_Cgmat(cgmat: np.ndarray) -> sym.core.add.Add:
    """Build a symbolic expression for the operator from its CG tensor.

    Constructs the formal sum
        Σ_{i₁…iₙ} c_{i₁…iₙ} O^X_{i₁…iₙ}
    using SymPy's ``ArraySymbol`` for the abstract operator symbol O.
    Indices are shifted from 0-based to 1-based (1234 convention) for
    readability in printed output.

    Parameters
    ----------
    cgmat : ndarray, shape (4,)*n

    Returns
    -------
    sympy.Expr
    """
    n = cgmat.ndim
    O = ArraySymbol("O", (5,) * n)
    operator_symbol = 0
    for indices in it.product(range(4), repeat=n):
        # Shift indices to the 1-based (1,2,3,4) convention used in publications.
        shifted_indices = [sum(x) for x in zip(indices, (1,) * n)]
        operator_symbol += cgmat[indices] * O[shifted_indices]
    return operator_symbol


def diracO_from_cgmat(cgmat: np.ndarray, X: str) -> sym.core.add.Add:
    """Construct the Dirac-space matrix form of the operator.

    For structure X and CG tensor c_{i₁…iₙ} this returns the matrix

        M(p) = Σ_{i₁…iₙ} c_{i₁…iₙ} · Γ^X_{i₁} · p_{i₂} · … · p_{iₙ}

    in 4×4 Dirac space, where:
    * V: Γ^V_i = γ_i
    * A: Γ^A_i = γ_i γ₅
    * T: Γ^T_{ij} = (i/2)(γ_i γ_j − γ_j γ_i) = σ_{ij}

    Parameters
    ----------
    cgmat : ndarray, shape (4,)*n
    X : str
        ``'V'``, ``'A'``, or ``'T'``.

    Returns
    -------
    sympy 4×4 matrix expression
    """
    if X not in ['V', 'A', 'T']:
        raise ValueError("X must be 'V', 'A' or 'T'.")
    n = cgmat.ndim
    op = np.zeros((4, 4))
    for indices in it.product(range(4), repeat=n):
        if X == 'V':
            gamma_prod = gamma_mu[indices[0]]
            start_ind = 1
        elif X == 'A':
            gamma_prod = gamma_mu[indices[0]] @ gamma5
            start_ind = 1
        else:  # 'T'
            # σ_{μν} = (i/2)[γ_μ, γ_ν]; the first two indices label the antisymmetric pair.
            gamma_prod = (
                I / 2 * (gamma_mu[indices[0]] @ gamma_mu[indices[1]] - gamma_mu[indices[1]] @ gamma_mu[indices[0]])
            )
            start_ind = 2
        # The remaining indices each contribute one factor of p_μ (one derivative).
        p_prod = 1
        for ind in indices[start_ind:]:
            p_prod *= p_mu[ind]
        op += cgmat[indices] * p_prod * gamma_prod
    return op


def Kfactor_from_diracO(operator: sym.core.add.Add) -> sym.core.mul.Mul:
    """Derive the symbolic kinematic factor from the Dirac-space operator matrix.

    The kinematic factor is defined as the ratio

        K = Tr[Γ_pol (-i p̸ + m) M(p) (-i p̸ + m)] / Tr[Γ_pol (-i p̸ + m) γ₄ (-i p̸ + m)] / (2E)

    where Γ_pol is the polarisation projector.  The numerator is the tree-level
    spin-averaged three-point function for operator M(p); the denominator is the
    same for the two-point function, chosen so that K → 1 for the simplest
    scalar case.

    After the ratio is formed we use the on-shell dispersion relation E² = p² + m²
    to simplify the expression as far as SymPy can.

    Parameters
    ----------
    operator : sympy matrix expression
        4×4 Dirac-space operator matrix M(p).

    Returns
    -------
    sympy scalar expression
    """
    num_K = sym.trace(Gamma_pol @ (-I * pslash + mN * Id_4) @ operator @ (-I * pslash + mN * Id_4)).simplify(
        rational=True
    )
    return (
        (num_K / den_K)
        .simplify(rational=True)
        .subs({E**2: p1**2 + p2**2 + p3**2 + mN**2, E**3: E * (p1**2 + p2**2 + p3**2 + mN**2)})
        .simplify(rational=True)
        .subs({p1**2 + p2**2 + p3**2 + mN**2: E**2, mN * (p1**2 + p2**2 + p3**2 + mN**2): mN * E**2})
        .simplify(rational=True)
    )


def trace_symm(cgmat: np.ndarray, X: str) -> str:
    """Return ``'= 0'`` or ``'!= 0'`` indicating whether Tr[cgmat] vanishes.

    The trace vanishes only if it is zero for every pair of distinct axes.

    Parameters
    ----------
    cgmat : ndarray
    X : str

    Returns
    -------
    str
    """
    n = cgmat.ndim
    if X != 'T':
        for axis1 in range(n):
            for axis2 in range(axis1 + 1, n):
                tr = np.trace(cgmat, axis1=axis1, axis2=axis2)
                if not np.allclose(tr, 0):
                    return "!= 0"
        return "= 0"
    else:
        # For tensor operators the first two indices carry σ_{μν}, which is antisymmetric.
        # The trace over those two indices vanishes by construction, and there is one
        # less trace we have to check.
        # See https://arxiv.org/pdf/hep-ph/0609231 eqs. 2.7-2.8
        # (To compute the trace the antisymmetrization of the first two indices has to
        # be made explicit)
        cgmat = cgmat - cgmat.swapaxes(0, 1)
        axis1 = 0
        for axis2 in range(2, n):
            tr = np.trace(cgmat, axis1=axis1, axis2=axis2)
            if not np.allclose(tr, 0):
                return "!= 0"
        for axis1 in range(1, n):
            for axis2 in range(axis1 + 1, n):
                tr = np.trace(cgmat, axis1=axis1, axis2=axis2)
                if not np.allclose(tr, 0):
                    return "!= 0"
        return "= 0"


def C_parity(cgmat: np.ndarray, X: str) -> int | str:
    """Return the C-parity of the operator: +1, -1, or ``'mixed'``.

    Charge conjugation acts on a moments operator as

        C O^X_{μ₁…μₙ} C⁻¹ = (−1)^{n + δ_X} O^X_{μ₁ μₙ … μ₂}

    where δ_X = 0 for V and 1 for A, T.  We identify C-parity by comparing
    the original cgmat with the one obtained by reversing the derivative indices.

    Parameters
    ----------
    cgmat : ndarray, shape (4,)*n
    X : str

    Returns
    -------
    int or str
        +1, -1, or ``'mixed'`` if neither eigenstate.
    """
    if X not in ['V', 'A', 'T']:
        raise ValueError("X must be 'V', 'A' or 'T'.")
    cgmat = np.round(cgmat, decimals=13)
    n = cgmat.ndim
    cgmat_C = np.empty(shape=np.shape(cgmat))
    for indices in it.product(range(4), repeat=n):
        # Reverse the derivative indices (all but the first one or two for T).
        if X in ('V', 'A'):
            cgmat_C[indices] = cgmat[(indices[0],) + indices[-1:0:-1]]
        else:
            cgmat_C[indices] = cgmat[(indices[0], indices[1]) + indices[-1:1:-1]]

    if (cgmat == cgmat_C).all():
        Cp = 1
    elif (cgmat == -cgmat_C).all():
        Cp = -1
    else:
        return "mixed"

    # Include the overall sign factor (−1)^{n + δ_X}.
    if X == "V":
        Cp *= (-1) ** n
    elif X in ("A", "T"):
        Cp *= (-1) ** (n + 1)
    return Cp


def index_symm(cgmat: np.ndarray) -> str:
    """Return the symmetry of cgmat under all index permutations.

    Iterates over all non-trivial permutations and checks whether the tensor is
    consistently symmetric or antisymmetric.  Only odd permutations carry
    information (an even permutation can always be decomposed into pairs of
    transpositions, so its sign alone does not distinguish S from A).

    Parameters
    ----------
    cgmat : ndarray, shape (4,)*n

    Returns
    -------
    str
        ``'Symmetric'``, ``'Antisymmetric'``, or ``'Mixed Symmetry'``.
    """
    n = cgmat.ndim
    symm_old = None
    symm_new = "Mixed Symmetry"
    for ip, p in enumerate(it.permutations(range(n))):
        if ip == 0:
            continue  # skip the identity
        cgmat_p = np.empty(shape=np.shape(cgmat))
        for indices in it.product(range(4), repeat=n):
            cgmat_p[indices] = cgmat[tuple(indices[p[i]] for i in range(n))]
        if (cgmat == cgmat_p).all() or (cgmat == -cgmat_p).all():
            if parity(p) == -1:
                symm_new = "Symmetric" if (cgmat == cgmat_p).all() else "Antisymmetric"
        else:
            return "Mixed Symmetry"
        if symm_old is not None and symm_new != symm_old:
            return "Mixed Symmetry"
        symm_old = symm_new
    return symm_new


def index_symm_2exchange(cgmat: np.ndarray, i1: int, i2: int) -> str:
    """Return the symmetry under permutations that swap only indices *i1* and *i2*.

    Parameters
    ----------
    cgmat : ndarray, shape (4,)*n
    i1, i2 : int
        The two index positions to be exchanged.

    Returns
    -------
    str
    """
    n = cgmat.ndim
    symm_old = None
    symm_new = "Mixed Symmetry"
    for ip, p in enumerate(it.permutations(range(n))):
        if ip == 0:
            continue
        if not (p[i1] == i2 and p[i2] == i1):
            continue
        cgmat_p = np.empty(shape=np.shape(cgmat))
        for indices in it.product(range(4), repeat=n):
            cgmat_p[indices] = cgmat[tuple(indices[p[i]] for i in range(n))]
        if (cgmat == cgmat_p).all() or (cgmat == -cgmat_p).all():
            if parity(p) == -1:
                symm_new = "Symmetric" if (cgmat == cgmat_p).all() else "Antisymmetric"
        else:
            return "Mixed Symmetry"
        if symm_old is not None and symm_new != symm_old:
            return "Mixed Symmetry"
        symm_old = symm_new
    return symm_new


def index_symm_index_fixed(cgmat: np.ndarray, *kwargs: int) -> str:
    """Return the symmetry under permutations that keep the specified indices fixed.

    Parameters
    ----------
    cgmat : ndarray, shape (4,)*n
    *kwargs : int
        Index positions that must remain fixed in every permutation considered.

    Returns
    -------
    str
    """
    n = cgmat.ndim
    fixed_indices = kwargs
    symm_old = None
    symm_new = "Mixed Symmetry"
    for ip, p in enumerate(it.permutations(range(n))):
        if ip == 0:
            continue
        if any(p[i] != i for i in fixed_indices):
            continue
        cgmat_p = np.empty(shape=np.shape(cgmat))
        for indices in it.product(range(4), repeat=n):
            cgmat_p[indices] = cgmat[tuple(indices[p[i]] for i in range(n))]
        if (cgmat == cgmat_p).all() or (cgmat == -cgmat_p).all():
            if parity(p) == -1:
                symm_new = "Symmetric" if (cgmat == cgmat_p).all() else "Antisymmetric"
        else:
            return "Mixed Symmetry"
        if symm_old is not None and symm_new != symm_old:
            return "Mixed Symmetry"
        symm_old = symm_new
    return symm_new


def cg_remapping(raw_cg: np.ndarray, n: int) -> np.ndarray:
    """Reshape a flat CG column of length 4ⁿ into an n-index tensor of shape (4,)×n.

    The CG database stores each eigenvector as a column of length dim = ∏ᵢ dᵢ,
    where the composite index runs over all combinations of the input-irrep indices
    in lexicographic order.  This function inverts that flattening so that the
    resulting tensor can be contracted directly with Lorentz indices.

    Parameters
    ----------
    raw_cg : ndarray, shape (4**n,)
    n : int
        Number of Lorentz indices (= number of tensor-product factors).

    Returns
    -------
    ndarray, shape (4,)*n
    """
    cg_remapped = np.zeros(shape=(4,) * n)
    # Build the mapping from flat index i to multi-index (i₁, …, iₙ) using base-4
    # arithmetic.  The +1 offset converts from 0-based to 1-based digits, which
    # then get shifted back by -1 to give the correct 0-based multi-index.
    mapping = (
        np.asarray([tuple(j) for j in [str(int(np.base_repr(i, 4)) + int('1' * n)) for i in range(4**n)]], dtype=int)
        - 1
    )
    for i in range(4**n):
        cg_remapped[tuple(mapping[i])] = raw_cg[i]
    return cg_remapped


def cg_remapping_T(raw_cg: np.ndarray, n: int) -> np.ndarray:
    """Reshape a tensorial CG column of length 6·4^(n−2) into an (n+1)-index tensor.

    For tensorial operators the first two Lorentz indices are antisymmetric,
    so the CG database stores only the 6 independent combinations
    {12, 13, 23, 14, 24, 34} rather than all 16.  This function expands them
    back into the full (4,4) antisymmetric pair.

    Parameters
    ----------
    raw_cg : ndarray, shape (6 * 4**(n-2),)
        Flattened CG column for a tensorial (T) operator with n−1 derivatives.
    n : int
        Number of derivative indices (the tensor accounts for two Dirac indices,
        so the full cgmat will have shape (4,)*(n+1)).

    Returns
    -------
    ndarray, shape (4,)*(n+1)
    """
    cg_remapped = np.zeros(shape=(4,) * (n + 1))
    # The order of the 6 indipendent combinations is fixed by the
    # choice of matrix representation for the irrep (6,1) of H(4)
    for k, ij in enumerate(["12", "13", "23", "14", "24", "34"]):
        i = int(ij[0]) - 1
        j = int(ij[1]) - 1
        mapping = (
            np.asarray(
                [tuple(l) for l in [str(int(np.base_repr(p, 4)) + int('1' * (n - 1))) for p in range(4 ** (n - 1))]],
                dtype=int,
            )
            - 1
        )
        for sub_i in range(4 ** (n - 1)):
            cg_remapped[tuple(np.concatenate([np.array([i, j]), mapping[sub_i]]))] = raw_cg[4 ** (n - 1) * k + sub_i]
    return cg_remapped


def make_operator_database(
    operator_folder: str, max_n: int, verbose: bool = False, cg_database: str | None = None
) -> None:
    """Build a database of H(4)-irreducible lattice operators up to *max_n* indices.

    For each combination of structure X ∈ {V, A, T} and index count n, the
    function loads the relevant CG matrices from the database, loops over all
    irreps and multiplicity blocks, and saves each resulting operator to a
    .npy file whose filename encodes the full operator metadata.

    Parameters
    ----------
    operator_folder : str
        Directory where operator .npy files will be written.
    max_n : int
        Maximum number of indices for V and A operators.  T operators
        automatically get one extra index (σ_{μν} contributes two).
    verbose : bool
        Print progress messages.
    cg_database : str or None
        Path to the CG-coefficient database.  ``None`` → use the bundled data.
    """
    if max_n < 2:
        raise ValueError("max_n must be at least 2.")
    Path(operator_folder).mkdir(parents=True, exist_ok=True)

    X_list = ['V', 'A', 'T']
    n_list = list(range(2, max_n + 1))
    iop = 1  # global operator counter (1-based, matches the printed catalogue)

    for n in n_list:
        if verbose:
            print(f"\nConstructing operators V{n}, A{n} and T{n+1}...\n")

        for X in tqdm(X_list, disable=not verbose):
            # The first irrep is fixed by the Dirac structure; all derivative
            # indices transform in the fundamental (4,1) representation.
            chosen_irreps = []
            if X == 'V':
                chosen_irreps.append((4, 1))
            elif X == 'A':
                chosen_irreps.append((4, 4))
            elif X == 'T':
                chosen_irreps.append((6, 1))
            while len(chosen_irreps) != n:
                chosen_irreps.append((4, 1))

            kwargs: dict = dict(verbose=False, force_computation=False)
            if cg_database is not None:
                kwargs['cgdatabase'] = cg_database
            cg_dict = cg_calc(*chosen_irreps, **kwargs).cg_dict

            for k, v in cg_dict.items():
                for imul, block in enumerate(v):
                    block = np.round(block, decimals=15)
                    # Each column of the CG block is an independent operator.
                    for icol in range(np.shape(block)[1]):
                        cg_mat = (
                            cg_remapping(block[:, icol], len(chosen_irreps))
                            if X in ('V', 'A')
                            else cg_remapping_T(block[:, icol], len(chosen_irreps))
                        )
                        Operator(
                            cgmat=cg_mat,
                            id=iop,
                            X=X,
                            irrep=rep_label_list[k],
                            block=imul + 1,
                            index_block=icol + 1,
                        ).save(operator_folder)
                        iop += 1

    if verbose:
        print(f"\nAll operators saved in {operator_folder}\n")


def Operator_from_file(filename: str) -> Operator:
    """Load an Operator from a .npy file, parsing metadata from the filename.

    Parameters
    ----------
    filename : str
        Path to a file saved by ``Operator.save()``.  The filename format is
        ``operator_{id}_{n}_{X}_{irrep0}_{irrep1}_{block}_{index_block}.npy``.

    Returns
    -------
    Operator
    """
    cgmat = np.load(filename)
    base = filename.split("/")[-1].replace(".npy", "")
    id_, n, X, irrep0, irrep1, block, index_block = base.split("_")[1:]
    return Operator(
        cgmat=cgmat,
        id=int(id_),
        X=X,
        irrep=(int(irrep0), int(irrep1)),
        block=int(block),
        index_block=int(index_block),
    )


def OperatorList_from_database(operator_database: str | None = None) -> list[Operator]:
    """Return a list of all operators in *operator_database*, sorted by id.

    Parameters
    ----------
    operator_database : str, optional
        Path to a directory created by ``make_operator_database``.  Defaults
        to the operator database bundled with the package.

    Returns
    -------
    list[Operator]
    """
    if operator_database is None:
        operator_database = _BUNDLED_OPERATOR_DATABASE
    if not Path(operator_database).is_dir():
        raise ValueError(f"Path does not exist: {operator_database}")
    files = sorted(
        [x for x in Path(operator_database).glob('**/*') if x.is_file()],
        key=lambda x: int(x.name.split("_")[1]),
    )
    return [Operator_from_file(f.as_posix()) for f in files]


def OperatorDict_from_database(operator_database: str | None = None) -> dict:
    """Load all operators into a nested dict keyed by ``(n, X)`` then ``(irrep, block)``.

    The returned structure is::

        operators_dict[(n, X)][(irrep, block)] → list[Operator]

    This layout makes it straightforward to look up all operators of a given
    irrep and multiplicity block for a specific structure and index count.

    Parameters
    ----------
    operator_database : str, optional
        Path to a directory created by ``make_operator_database``.  Defaults
        to the operator database bundled with the package.

    Returns
    -------
    dict
    """
    if operator_database is None:
        operator_database = _BUNDLED_OPERATOR_DATABASE
    if not Path(operator_database).is_dir():
        raise ValueError(f"Path does not exist: {operator_database}")
    files = sorted(
        [x for x in Path(operator_database).glob('**/*') if x.is_file()],
        key=lambda x: int(x.name.split("_")[1]),
    )
    operators_dict: dict = {}
    for file in files:
        op = Operator_from_file(file.as_posix())
        if (op.n, op.X) not in operators_dict:
            operators_dict[(op.n, op.X)] = {}
        if (op.irrep, op.block) not in operators_dict[(op.n, op.X)]:
            operators_dict[(op.n, op.X)][(op.irrep, op.block)] = []
        operators_dict[(op.n, op.X)][(op.irrep, op.block)].append(op)
    return operators_dict


def Operator_from_database(operator_id: int, operator_database: str | None = None) -> Operator:
    """Load a single operator by its integer id from *operator_database*.

    Parameters
    ----------
    operator_id : int
        1-based operator id as listed in the operator catalogue.
    operator_database : str, optional
        Path to a directory created by ``make_operator_database``.  Defaults
        to the operator database bundled with the package.

    Returns
    -------
    Operator
    """
    if operator_database is None:
        operator_database = _BUNDLED_OPERATOR_DATABASE
    if not Path(operator_database).is_dir():
        raise ValueError(f"Path does not exist: {operator_database}")
    files = sorted(
        [x for x in Path(operator_database).glob('**/*') if x.is_file()],
        key=lambda x: int(x.name.split("_")[1]),
    )
    if not isinstance(operator_id, int) or operator_id < 1 or operator_id > len(files):
        raise ValueError(f"operator_id must be between 1 and {len(files)}, got {operator_id}.")
    return Operator_from_file(files[operator_id - 1].as_posix())


def get_OperatorList() -> list[Operator]:
    """Return all operators from the bundled database as a flat sorted list.

    This is a zero-argument convenience wrapper around
    :func:`OperatorList_from_database`.  It always reads from the operator
    database shipped with the h4lat package, so no path configuration is needed.
    The returned list is sorted by the integer operator id, matching the
    printed catalogue order.

    Use :func:`OperatorList_from_database` with an explicit path if you want to
    read from a custom database generated by :func:`make_operator_database`.

    Returns
    -------
    list[Operator]
        All operators in the bundled database, sorted by their integer id.

    Examples
    --------
    >>> ops = get_OperatorList()
    >>> print(len(ops))           # total number of bundled operators
    >>> print(ops[0].X)           # Dirac structure of the first operator
    >>> print(ops[0].irrep)       # H(4) irrep label of the first operator
    >>> print(ops[0].latex_K)     # kinematic factor as a LaTeX string
    """
    # Passing no argument makes OperatorList_from_database fall back to
    # _BUNDLED_OPERATOR_DATABASE, which points at the data/ folder inside the
    # installed package (resolved via importlib.resources).
    return OperatorList_from_database()


def get_OperatorDict() -> dict:
    """Return all operators from the bundled database as a nested dictionary.

    This is a zero-argument convenience wrapper around
    :func:`OperatorDict_from_database`.  It always reads from the operator
    database shipped with the h4lat package, so no path configuration is needed.

    The returned structure is::

        operators_dict[(n, X)][(irrep, block)] -> list[Operator]

    where:

    * *n* is the total number of Lorentz indices (= 1 + number of derivatives
      for V/A, = 2 + number of derivatives for T).
    * *X* is the Dirac structure (``'V'``, ``'A'``, or ``'T'``).
    * *irrep* is an H(4) label ``(k, l)`` (e.g. ``(4, 1)``).
    * *block* is the multiplicity index (1-based) within the irrep.

    This layout makes it straightforward to look up all operators of a given
    irrep and multiplicity block for a specific structure and index count.

    Use :func:`OperatorDict_from_database` with an explicit path if you want to
    read from a custom database generated by :func:`make_operator_database`.

    Returns
    -------
    dict
        Nested dict; see :func:`OperatorDict_from_database` for the full
        key structure.

    Examples
    --------
    >>> d = get_OperatorDict()
    >>> # All 2-index vector operators in irrep (4,1), multiplicity block 1:
    >>> ops = d[(2, 'V')][(4, 1), 1]
    >>> print(ops[0].symm)        # index-permutation symmetry
    >>> print(ops[0].C)           # C-parity eigenvalue
    """
    # Passing no argument makes OperatorDict_from_database fall back to
    # _BUNDLED_OPERATOR_DATABASE, which points at the data/ folder inside the
    # installed package (resolved via importlib.resources).
    return OperatorDict_from_database()


def get_op_selection(n_der: int) -> list[Operator]:
    """Return a curated selection of operators for a given number of derivatives.

    This function returns the standard set of operators used in the
    moments-operator mixing analysis, loaded from the bundled database.
    The operators are the irreducible H(4) representations relevant for
    each dataset (1, 2, or 3 derivatives).

    Parameters
    ----------
    n_der : int
        Number of derivatives.  Must be 1, 2, or 3.

    Returns
    -------
    list[Operator]
        Ordered list: [opV1, opV2, ..., opA1, opA2, ..., opT1, opT2, ...]
        For n_der=3 the list is [opV, opA].

    Raises
    ------
    ValueError
        If n_der is not 1, 2, or 3.
    """
    if n_der == 1:
        # Vector
        # irrep (3,1)
        opV1 = 1/6 * Operator_from_database(2)                                              # = 11 + 22 + 33 - 3 * 44
        opV2 = 1/(3 * np.sqrt(2)) * (Operator_from_database(2) - Operator_from_database(3)) # = 33 - 44
        # irrep (6,3)
        opV3 = 1/np.sqrt(2) * Operator_from_database(14)                                    # = 14 + 41

        # Axial - irrep (6,4)
        opA1 = 1/np.sqrt(2) * Operator_from_database(28) # = 13 + 31
        opA2 = 1/np.sqrt(2) * Operator_from_database(32) # = 34 + 43

        # Tensor
        # irrep (8,1)
        opT1 = Operator_from_database(42) - 1/2 * Operator_from_database(46)  # = 211 - 244
        opT2 = - Operator_from_database(46)                                    # = 233 - 244
        # irrep (8,2)
        opT3 =  Operator_from_database(55) + 1/2 * Operator_from_database(51) # = 124 - 241
        opT4 = 2 * Operator_from_database(55)

        return [opV1, opV2, opV3, opA1, opA2, opT1, opT2, opT3, opT4]

    if n_der == 2:
        # Vector - irrep (4,2)
        opV1 = Operator_from_database(73)  # = {2,3,4}      ( {}=symmetric combination )
        opV2 = Operator_from_database(74)  # = {1,3,4}
        opV3 = Operator_from_database(75)  # = {1,2,4}
        opV4 = Operator_from_database(76)  # = {1,2,3}

        # Axial - irrep (4,3)
        opA1 = Operator_from_database(125) # = {2,3,4}
        opA2 = Operator_from_database(126) # = {1,3,4}
        opA3 = Operator_from_database(127) # = {1,2,4}
        opA4 = Operator_from_database(128) # = {1,2,3}

        # Tensor - irrep (3,2)
        # (operators 197 and 198 cannot be used because their K is proportional to pz*(px-py),
        #  and we have px=py in the 2-der dataset, and pz=0 in the 1-der one)
        opT1 =  2 * Operator_from_database(199) # = 2x 1,2,{3,4} + 1,3,{2,4} + 1,4,{2,3} - 2,3,{1,4} - 2,4,{1,3}

        # Tensor - irrep (3,3)
        # (operator 202 cannot be used because it has K=0 always) (TO DO: look better into it to be sure)
        opT2 =      Operator_from_database(200) # = 1,2,{1,2} - 1,3,{1,3} + 2,3,{2,3}
        opT3 = -2 * Operator_from_database(201) # = 2x 1,2,{1,2} + 1,3,{1,3} -3x 1,4,{1,4} - 2,3,{2,3} +3 2,4,{2,4}

        # Tensor - irrep (6,2) (the block with C=-1)
        opT4 =     Operator_from_database(233) + Operator_from_database(239)   # = 1,3,{1,4} + 1,4,{1,3} - 2,3,{2,4} - 2,4,{2,3}
        opT5 =     Operator_from_database(234) + Operator_from_database(240)   # = 1,2,{1,4} + 1,4,{1,2} + 2,3,{3,4} - 3,4,{2,3}
        opT6 = - ( Operator_from_database(235) + Operator_from_database(241) ) # = 1,2,{2,4} - 1,3,{3,4} - 2,4,{1,2} + 3,4,{1,3}
        opT7 =     Operator_from_database(236) + Operator_from_database(242)   # = 1,2,{1,3} + 1,3,{1,2} + 2,4,{3,4} + 3,4,{2,4}
        opT8 = - ( Operator_from_database(237) + Operator_from_database(243) ) # = 1,2,{2,3} - 1,4,{3,4} - 2,3,{1,2} - 3,4,{1,4}
        opT9 = - ( Operator_from_database(238) + Operator_from_database(244) ) # = 1,3,{2,3} - 1,4,{2,4} + 2,3,{1,3} - 2,4,{1,4}

        # Set custom labels
        opV1.id = "V1"; opV2.id = "V2"; opV3.id = "V3"; opV4.id = "V4"
        opA1.id = "A1"; opA2.id = "A2"; opA3.id = "A3"; opA4.id = "A4"
        opT1.id = "T1"; opT2.id = "T2"; opT3.id = "T3"; opT4.id = "T4"
        opT5.id = "T5"; opT6.id = "T6"; opT7.id = "T7"; opT8.id = "T8"
        opT9.id = "T9"

        return [opV1, opV2, opV3, opV4, opA1, opA2, opA3, opA4,
                opT1, opT2, opT3, opT4, opT5, opT6, opT7, opT8, opT9]

    if n_der == 3:
        # Construct the matrix corresponding to the completely symmetric combination of 4 indices
        matrix = np.zeros((4, 4, 4, 4), dtype=float)
        perms = np.array(list(it.permutations(range(4))))
        matrix[tuple(perms.T)] = 1.0

        # Vector - irrep (1,2)
        opV = Operator(cgmat=matrix,         # = {1,2,3,4} (completely symmetric combination of 4 indices)
                       id="V",
                       X="V",
                       irrep=(1, 2),
                       block=1,
                       index_block=1)

        # Axial - irrep (1,3)
        opA = Operator(cgmat=matrix,         # = {1,2,3,4} (completely symmetric combination of 4 indices)
                       id="A",
                       X="A",
                       irrep=(1, 3),
                       block=1,
                       index_block=1)

        return [opV, opA]

    raise ValueError(f"n_der must be 1, 2, or 3, got {n_der}.")


def latexO_from_diracO(operatorO: sym.core.add.Add, X: str) -> str:
    """Convert the symbolic operator expression to a LaTeX-printable string.

    Monomials are sorted by their multi-index value and formatted using the
    ``numerics_to_latex_conv`` dictionary so that CG coefficients are rendered
    as proper fractions or square-root fractions rather than raw floats.

    Parameters
    ----------
    operatorO : sympy expression
    X : str

    Returns
    -------
    str
        A LaTeX string suitable for display in a Jupyter notebook or PDF.
    """
    if X not in ['V', 'A', 'T']:
        raise ValueError("X must be 'V', 'A' or 'T'.")
    latex_str = ""
    monoms_list = list(sym.Add.make_args(operatorO))
    # Extract the concatenated index string from each monomial for sorting.
    index_list = [
        int(
            ''.join(
                str(monom.as_coeff_mul()[1][1])
                .replace('O', '')
                .replace('[', '')
                .replace(']', '')
                .replace(' ', '')
                .split(',')
            )
        )
        for monom in monoms_list
    ]
    monoms_list = [m for _, m in sorted(zip(index_list, monoms_list))]

    for i, e in enumerate(monoms_list):
        sign = e.as_coeff_mul()[0]
        coeff = e.as_coeff_mul()[1][0]
        symbol = e.as_coeff_mul()[1][1]

        if sign == 1 and i != 0:
            latex_str += "+"
        elif sign == -1:
            latex_str += "-"

        # Look up the coefficient in the conversion dict; fall back to str()
        # if it is not found (e.g. for exotic CG values not in the precomputed table).
        if coeff in numerics_to_latex_conv:
            latex_str += numerics_to_latex_conv[coeff]
        elif float(coeff) in numerics_to_latex_conv:
            latex_str += numerics_to_latex_conv[float(coeff)]
        elif np.round(float(coeff), decimals=13) in numerics_to_latex_conv:
            latex_str += numerics_to_latex_conv[np.round(float(coeff), decimals=13)]
        else:
            latex_str += str(coeff)

        latex_str += str(symbol).replace('[', '^{' + X + '}_{').replace(']', '}')

    return latex_str


def decomposition_analysis(X: str, n_der: int, operator_dict: dict | None = None, verbose: bool = False):
    """Analyse the irrep content of the tensor product irrep(X) ⊗ (4,1)^{n_der}.

    For each irrep appearing in the decomposition the function reports:

    * **Mixing Safe** (Y/N): whether the irrep also appears in any *lower-order*
      tensor product of the same kind.  An irrep is mixing-safe (Y) only if it
      first appears at the current order; otherwise renormalisation can mix it
      with lower-dimensional operators.
    * **Multiplicity**: the number of independent copies of the irrep.
    * **Symmetry** (if *operator_dict* is given): the index-permutation symmetry
      of the operators in the irrep.

    Parameters
    ----------
    X : str
        ``'V'``, ``'A'``, or ``'T'`` — selects the leading irrep.
    n_der : int
        Number of derivative indices.
    operator_dict : dict or None
        If provided (output of ``OperatorDict_from_database``), also reports
        the index-symmetry of the operators.
    verbose : bool
        Print the analysis table to stdout.

    Returns
    -------
    pandas.DataFrame
    """
    if not _PANDAS_AVAILABLE:
        raise ImportError("pandas is required for decomposition_analysis().")
    if X not in ['V', 'A', 'T']:
        raise ValueError("X must be 'V', 'A' or 'T'.")
    if not isinstance(n_der, int) or n_der < 1:
        raise ValueError("n_der must be a positive integer.")

    V = (4, 1)
    starting_irrep = ((4, 1),) if X == 'V' else ((4, 4),) if X == 'A' else ((6, 1),)

    available_irreps = {
        rep_label_list[i]: ["Y", mul, "-"]
        for i, mul in enumerate(get_multiplicities(*(starting_irrep + (V,) * n_der)))
        if mul > 0
    }

    # For each lower-dimensional product of any starting structure, check whether
    # the current irrep already appears.  If it does, renormalisation mixing is
    # possible and we flag it as not mixing-safe.
    starting_irreps_list = [(1, 1), (1, 4), (4, 1), (4, 4), (6, 1)]
    for i_dim in range(n_der):
        for initial_irrep in starting_irreps_list:
            low_dim_irrep = get_multiplicities(*((initial_irrep,) + (V,) * i_dim))
            for key in available_irreps:
                if low_dim_irrep[irrep_index[key]] > 0:
                    available_irreps[key][0] = "N"

    if operator_dict is not None:
        for key in available_irreps:
            if available_irreps[key][1] == 1:
                n_idx = n_der + 1 if X in ('V', 'A') else n_der + 2
                op_list = operator_dict[(n_idx, X)][key, 1]
                symm_list = [op.symm for op in op_list]
                available_irreps[key][2] = symm_list[0][0] if all_equal(symm_list) else 'Mixed'

    df = pd.DataFrame(
        [[k, *v] for k, v in available_irreps.items()],
        index=available_irreps.keys(),
        columns=["Irrep", "Mixing Safe", "Multiplicity", "Symmetry"],
    )
    if operator_dict is None or X == 'T':
        df = df.drop(columns=["Symmetry"])

    if verbose:
        print(f"\nAnalysis of { ' x '.join(str(e) for e in (starting_irrep + (V,) * n_der)) } :\n")
        print("\n".join("|" + "|".join(row.split("|")[2:]) for row in df.to_markdown().split("\n")))

    return df


################## HDF5 I/O #####################


def write_operator(group, operator: Operator) -> None:
    """Serialise an Operator into an HDF5 group.

    Stores the ``cgmat`` as a compressed dataset and all scalar attributes
    (``id``, ``X``, ``irrep``, ``block``, ``index_block``) as group attributes.
    None-valued attributes are recorded via a ``_is_none`` flag so that they
    can be correctly reconstructed by ``read_operator``.

    Parameters
    ----------
    group : h5py.Group
    operator : Operator
    """
    if not _H5PY_AVAILABLE:
        raise ImportError("h5py is required for HDF5 I/O.")
    op_group = group.create_group("operator")
    op_group.attrs["id"] = operator.id
    op_group.attrs["X"] = operator.X
    op_group.create_dataset("cgmat", data=operator.cgmat, compression="gzip")

    if operator.irrep is None:
        op_group.attrs["irrep_is_none"] = True
    else:
        op_group.attrs["irrep"] = operator.irrep

    if operator.block is None:
        op_group.attrs["block_is_none"] = True
    else:
        op_group.attrs["block"] = operator.block

    if operator.index_block is None:
        op_group.attrs["index_block_is_none"] = True
    else:
        op_group.attrs["index_block"] = operator.index_block


def read_operator(group) -> Operator:
    """Reconstruct an Operator from an HDF5 group written by ``write_operator``.

    Parameters
    ----------
    group : h5py.Group

    Returns
    -------
    Operator
    """
    if not _H5PY_AVAILABLE:
        raise ImportError("h5py is required for HDF5 I/O.")
    import numpy as _np

    op_group = group["operator"]

    raw_id = op_group.attrs["id"]
    if isinstance(raw_id, bytes):
        raw_id = raw_id.decode()
    operator_id = int(raw_id) if isinstance(raw_id, (int, _np.integer)) else str(raw_id)

    irrep = None if op_group.attrs.get("irrep_is_none", False) else tuple(op_group.attrs["irrep"])
    block = None if op_group.attrs.get("block_is_none", False) else int(op_group.attrs["block"])
    index_block = None if op_group.attrs.get("index_block_is_none", False) else int(op_group.attrs["index_block"])

    return Operator(
        cgmat=_np.array(op_group["cgmat"]),
        id=operator_id,
        X=str(op_group.attrs["X"]),
        irrep=irrep,
        block=block,
        index_block=index_block,
    )


###################### Execution as Main ############################

if __name__ == "__main__":
    max_n = 2
    if len(sys.argv) > 1:
        try:
            max_n = int(sys.argv[1])
        except ValueError:
            pass
    make_operator_database(operator_folder="operator_database", max_n=max_n, verbose=True)
