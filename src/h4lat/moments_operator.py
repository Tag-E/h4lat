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

where Γ^X is the Dirac structure (V: γ_μ, A: γ_μ γ₅, T: σ_{μν}, S: 𝟙, P: γ₅), the c_{i₁…iₙ}
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

import functools
import itertools as it
import math
import os
import re
import subprocess
import sys
import time
import warnings
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
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
    from pylatex import Alignat, Command, Document, Section, Subsection
    from pylatex.utils import NoEscape

    _PYLATEX_AVAILABLE = True
except ImportError:
    _PYLATEX_AVAILABLE = False

try:
    import h5py  # used directly by operator_list_to_h5/h5_to_operator_list

    _H5PY_AVAILABLE = True
except ImportError:
    _H5PY_AVAILABLE = False

from .cg_calculator import (
    _BUNDLED_OPERATOR_DATABASE,
    cg_calc,
    get_multiplicities,
    irrep_index,
    rep_label_list,
    rep_latex_names,
)
from .kinematic_data import (
    E,
    Gamma_pol_list,
    I,
    Id_4,
    den_K_list,
    gamma5,
    gamma_mu,
    latex_conversion_table,
    mN,
    p1,
    p2,
    p3,
    p_mu,
    pslash,
)
from .utilities import all_equal, parity


def _n_der(n: int, X: str) -> int:
    """Number of derivatives of an operator with *n* Lorentz indices and Dirac structure *X*.

    The Dirac structure takes 0 indices for S/P, 1 for V/A (γ_μ) and 2 for T (σ_{μν});
    every other index belongs to a derivative.
    """
    if X in ('S', 'P'):
        return n
    if X == 'T':
        return n - 2
    return n - 1  # V, A


######################## Main Class #####################################


class Operator:
    """
    A single irreducible lattice moments operator characterised by its
    Clebsch-Gordan coefficient tensor and Dirac structure.

    The ``cgmat`` is an n-index NumPy array of shape (4,)×n, where n is the
    total number of Lorentz indices (one for the Dirac structure plus one per
    derivative).  Each entry c_{i₁…iₙ} is the CG coefficient that selects the
    appropriate linear combination of elementary tensors.

    On construction the class computes:
    * ``C``  — C-parity (+1, -1, or 'mixed').
    * ``tr`` — whether the trace Tr[cgmat] vanishes.
    * ``symm`` — index-permutation symmetry ('Symmetric', 'Antisymmetric',
      'Mixed Symmetry', or '[Invalid Operator]' for tensorial cases).

    The symbolic attributes, which take most of the time, are computed on first
    access and then cached, so that loading many operators stays fast:
    * ``O``  — the symbolic operator expression (for display/LaTeX).
    * ``K``  — the tree-level kinematic factor as a SymPy expression.
    * ``p3corr_is_real``, ``latex_O``, ``latex_K`` — derived from them.
    * ``gamma_pol_index`` — index (0–5) of the polarisation projector used
      for computing K.  Default is 0, which reproduces the original
      Γ_pol = ½(1 + γ₄)(1 − iγ₁γ₂).  Call ``set_polarization_matrix()`` to switch
      to a different projector and recompute K in place.
    """

    def __init__(
        self,
        cgmat: np.ndarray,
        id: int | str,
        X: str,
        irrep: tuple[int, int] | None,
        block: int | None,
        index_block: int | None,
        gamma_pol_index: int = 0,
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
            ``'T'`` (tensor σ_{μν} = i/2 [γ_μ, γ_ν]), ``'S'`` (scalar 𝟙),
            or ``'P'`` (pseudoscalar γ₅).
        irrep : tuple[int,int] or None
            H(4) irrep label, e.g. ``(4,1)``.
        block : int or None
            Multiplicity index (1-based) within the irrep.
        index_block : int or None
            Column index (1-based) within the CG block.
        gamma_pol_index : int, optional
            Index (0–5) selecting the polarisation projector used to compute
            the kinematic factor K (default 0).  The six choices are:
                0: γ₁γ₂  (default — original hardcoded projector)
                1: γ₁γ₃
                2: γ₂γ₃
                3: γ₁γ₄
                4: γ₂γ₄
                5: γ₃γ₄
            Each projector has the form Γ_pol = ½(1 + γ₄)(1 − i γ_a γ_b).
            The stored value can be changed after construction via
            ``set_polarization_matrix()``.
        """
        if not 0 <= gamma_pol_index <= 5:
            raise ValueError(f"gamma_pol_index must be 0–5, got {gamma_pol_index}.")
        self.cgmat = cgmat[:]
        self.id = id
        self.X = X
        self.irrep = irrep
        self.block = block
        self.index_block = index_block

        # Index of the polarisation projector used for computing K.
        self.gamma_pol_index = gamma_pol_index

        # Number of Lorentz indices; number of derivatives depends on how many
        # indices the Dirac structure consumes: 0 for S/P, 1 for V/A, 2 for T.
        self.n = cgmat.ndim
        self.nder = _n_der(self.n, X)

        # The symbolic representations (O, K and what derives from them) are computed
        # on first access: see the cached properties below.

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

    # ------------------------------------------------------------------
    # Symbolic representations, computed on first access and cached
    # ------------------------------------------------------------------

    @functools.cached_property
    def O(self) -> sym.Expr:  # noqa: E743 (the operator is called O throughout)
        """Symbolic operator expression Σ c_{i₁…iₙ} O^X_{i₁…iₙ} (see symO_from_Cgmat)."""
        return symO_from_Cgmat(self.cgmat)

    @functools.cached_property
    def K(self) -> sym.Expr:
        """Tree-level kinematic factor, with the polarisation projector gamma_pol_index."""
        return Kfactor_from_diracO(diracO_from_cgmat(self.cgmat, self.X), self.gamma_pol_index)

    @functools.cached_property
    def p3corr_is_real(self) -> bool:
        """Whether the three-point correlator is real.

        It is real or imaginary depending on the number of derivatives and the Dirac
        structure; we determine this from the presence of the imaginary unit in K.
        """
        return (I in self.K.atoms()) if self.nder % 2 == 1 else (I not in self.K.atoms())

    @functools.cached_property
    def latex_O(self) -> str:
        """LaTeX string of the operator expression O."""
        return latexO_from_diracO(self.O, self.X)

    @functools.cached_property
    def latex_K(self) -> str:
        """LaTeX string of the kinematic factor K."""
        latex_K = str(self.K).replace('**', '^').replace('*', '').replace('I', 'i')
        # K is exact, so square roots of the CG coefficients (e.g. 1/√2 in some
        # get_op_selection operators) appear as sqrt(2): write them as \sqrt{2}.
        latex_K = re.sub(r"sqrt\((\d+)\)", r"\\sqrt{\1}", latex_K)
        if '/' in latex_K:
            latex_K = "\\frac{" + latex_K.split('/')[0] + "}{ " + latex_K.split('/')[1] + "}"
        return latex_K

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
        """Add two operators with matching X, index count, and polarisation index."""
        if self.X != other.X or self.n != other.n:
            raise ValueError("Operator addition requires equal X and number of indices.")
        if self.gamma_pol_index != other.gamma_pol_index:
            raise ValueError(
                "Operator addition requires equal gamma_pol_index "
                f"(got {self.gamma_pol_index} and {other.gamma_pol_index})."
            )
        new_irrep = self.irrep if self.irrep == other.irrep else None
        new_block = self.block if (new_irrep and self.block == other.block) else None
        return Operator(
            cgmat=np.round(self.cgmat + other.cgmat, decimals=13),
            id=None,
            X=self.X,
            irrep=new_irrep,
            block=new_block,
            index_block=None,
            gamma_pol_index=self.gamma_pol_index,
        )

    def __sub__(self, other: Self) -> Self:
        """Subtract two operators with matching X, index count, and polarisation index."""
        if self.X != other.X or self.n != other.n:
            raise ValueError("Operator subtraction requires equal X and number of indices.")
        if self.gamma_pol_index != other.gamma_pol_index:
            raise ValueError(
                "Operator subtraction requires equal gamma_pol_index "
                f"(got {self.gamma_pol_index} and {other.gamma_pol_index})."
            )
        new_irrep = self.irrep if self.irrep == other.irrep else None
        new_block = self.block if (new_irrep and self.block == other.block) else None
        return Operator(
            cgmat=np.round(self.cgmat - other.cgmat, decimals=13),
            id=None,
            X=self.X,
            irrep=new_irrep,
            block=new_block,
            index_block=None,
            gamma_pol_index=self.gamma_pol_index,
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
            gamma_pol_index=self.gamma_pol_index,
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
            gamma_pol_index=self.gamma_pol_index,
        )

    def __neg__(self) -> Self:
        return Operator(
            cgmat=self.cgmat * (-1),
            id=self.id,
            X=self.X,
            irrep=self.irrep,
            block=self.block,
            index_block=self.index_block,
            gamma_pol_index=self.gamma_pol_index,
        )

    # ------------------------------------------------------------------
    # Polarisation projector
    # ------------------------------------------------------------------

    def set_polarization_matrix(self, gamma_pol_index: int) -> None:
        """Change the polarisation projector and recompute the kinematic factor K.

        After this call ``self.gamma_pol_index``, ``self.K``,
        ``self.p3corr_is_real``, and ``self.latex_K`` all reflect the new
        projector (K is recomputed on its next access).  All other attributes (cgmat, O, C, tr,
        symm, …) are unchanged.

        The six projectors, indexed 0–5, all have the form
            Γ_pol = ½(1 + γ₄)(1 − i γ_a γ_b)
        with the following (a, b) pairs:
            0: γ₁γ₂  (default — original hardcoded projector)
            1: γ₁γ₃
            2: γ₂γ₃
            3: γ₁γ₄
            4: γ₂γ₄
            5: γ₃γ₄

        Parameters
        ----------
        gamma_pol_index : int
            New polarisation index, 0–5.

        Raises
        ------
        ValueError
            If *gamma_pol_index* is outside 0–5.

        Examples
        --------
        >>> op = Operator_from_database(1)
        >>> op.gamma_pol_index          # 0 by default
        0
        >>> op.set_polarization_matrix(2)      # switch to γ₂γ₃ projector
        >>> op.gamma_pol_index
        2
        >>> op.K                        # recomputed with the new projector
        """
        if not 0 <= gamma_pol_index <= 5:
            raise ValueError(f"gamma_pol_index must be 0–5, got {gamma_pol_index}.")
        self.gamma_pol_index = gamma_pol_index
        # Forget K and what derives from it: they are recomputed, with the new projector, on next access.
        for name in ("K", "p3corr_is_real", "latex_K"):
            self.__dict__.pop(name, None)

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
    # Only the non-zero coefficients contribute: skipping the zeros gives the same
    # expression, but is much faster for many indices (a T operator with three
    # derivatives has 4⁵ = 1024 entries, almost all of them zero).
    for indices in zip(*np.nonzero(cgmat)):
        # Shift indices to the 1-based (1,2,3,4) convention used in publications.
        shifted_indices = [int(i) + 1 for i in indices]
        operator_symbol += cgmat[indices] * O[shifted_indices]
    return operator_symbol


@functools.lru_cache(maxsize=None)
def _exact_number(x: float, max_denominator: int = 10**4, rtol: float = 1e-10) -> sym.Expr:
    """Return the exact SymPy number that the float *x* approximates.

    The CG coefficients are computed in floating point, so an entry that should be
    exactly 2 can be stored as 2.000000000000001.  In a symbolic computation such
    noise does not cancel: terms that should sum to zero leave a residue of order
    1e-16, which SymPy then keeps (e.g. as the rational 7105427357601/8·10²⁷).
    Converting every coefficient to the exact number it stands for removes it.

    Two forms are tried, in order:

    * a rational p/q with q ≤ *max_denominator* (all CG coefficients of the
      bundled database are of this form: ¼, ⅓, ½, …, 6);
    * ±√(p/q), found from x² (e.g. 1/√2 for operators built as
      ``1/np.sqrt(2) * op``, as in ``get_op_selection``).

    A candidate is accepted only if it agrees with *x* to the relative tolerance
    *rtol* (the noise is ~1e-14, far below it).  If neither form fits, *x* is
    returned as a ``sympy.Float``, i.e. the behaviour before this conversion.

    Parameters
    ----------
    x : float
        A coefficient of the cgmat (pass a Python float, so the cache can be used).
    max_denominator : int, optional
        Largest denominator q tried (default 10⁴).
    rtol : float, optional
        Relative tolerance for accepting a candidate (default 1e-10).

    Returns
    -------
    sympy.Expr
        ``sympy.Rational``, ±``sympy.sqrt(Rational)`` or, as a fallback, ``sympy.Float``.
    """
    if x == 0:
        return sym.S.Zero
    # Absolute tolerance: relative to |x|, but never below rtol itself.
    tol = rtol * max(1.0, abs(x))

    # 1) Rational p/q: the closest fraction with denominator ≤ max_denominator.
    #    Two different fractions with denominators ≤ q_max differ by at least
    #    1/q_max², so the noise cannot make it pick the wrong one.
    r = Fraction(x).limit_denominator(max_denominator)
    if abs(float(r) - x) <= tol:
        return sym.Rational(r.numerator, r.denominator)

    # 2) ±√(p/q): x² is then rational.  SymPy simplifies sqrt(Rational) itself,
    #    e.g. sqrt(1/2) → √2/2.
    r2 = Fraction(x * x).limit_denominator(max_denominator)
    if abs(math.sqrt(r2) - abs(x)) <= tol:
        return (1 if x > 0 else -1) * sym.sqrt(sym.Rational(r2.numerator, r2.denominator))

    # 3) Neither form: keep the float, as before.
    return sym.Float(x)


def diracO_from_cgmat(cgmat: np.ndarray, X: str) -> sym.core.add.Add:
    """Construct the Dirac-space matrix form of the operator.

    For structure X and CG tensor c_{i₁…iₙ} this returns the matrix

        M(p) = Σ_{i₁…iₙ} c_{i₁…iₙ} · Γ^X_{i₁} · p_{i₂} · … · p_{iₙ}

    in 4×4 Dirac space, where:
    * V: Γ^V_i = γ_i            (one Dirac index)
    * A: Γ^A_i = γ_i γ₅         (one Dirac index)
    * T: Γ^T_{ij} = (i/2)(γ_i γ_j − γ_j γ_i) = σ_{ij}  (two Dirac indices)
    * S: Γ^S = 𝟙                (no Dirac index; all indices are derivatives)
    * P: Γ^P = γ₅               (no Dirac index; all indices are derivatives)

    The coefficients are converted to the exact numbers they approximate (see
    ``_exact_number``), so the matrix, and the kinematic factor computed from it,
    is free of floating-point noise: terms that should cancel give exactly 0.

    Parameters
    ----------
    cgmat : ndarray, shape (4,)*n
    X : str
        ``'V'``, ``'A'``, ``'T'``, ``'S'``, or ``'P'``.

    Returns
    -------
    sympy 4×4 matrix expression
    """
    if X not in ['V', 'A', 'T', 'S', 'P']:
        raise ValueError("X must be 'V', 'A', 'T', 'S' or 'P'.")
    # Start from an exact SymPy zero matrix: the 0.0 entries of a NumPy array would
    # turn every exact coefficient added to them back into a float.
    op = sym.zeros(4, 4)
    # Only the non-zero coefficients contribute (most of the 4ⁿ entries are zero).
    for indices in zip(*np.nonzero(cgmat)):
        indices = tuple(int(i) for i in indices)
        if X == 'V':
            gamma_prod = gamma_mu[indices[0]]
            start_ind = 1
        elif X == 'A':
            gamma_prod = gamma_mu[indices[0]] @ gamma5
            start_ind = 1
        elif X == 'S':
            gamma_prod = Id_4
            start_ind = 0
        elif X == 'P':
            gamma_prod = gamma5
            start_ind = 0
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
        # Exact coefficient instead of the stored float (see _exact_number).
        op += _exact_number(float(cgmat[indices])) * p_prod * gamma_prod
    return op


def Kfactor_from_diracO(operator: sym.core.add.Add, gamma_pol_index: int = 0) -> sym.core.mul.Mul:
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
    gamma_pol_index : int, optional
        Index selecting which polarisation projector to use (default 0).
        The six choices are, in order:
            0: γ₁γ₂  (default — reproduces the original hardcoded projector)
            1: γ₁γ₃
            2: γ₂γ₃
            3: γ₁γ₄
            4: γ₂γ₄
            5: γ₃γ₄
        Each projector has the form Γ_pol = ½(1 + γ₄)(1 − i γ_a γ_b).
        See ``kinematic_data.Gamma_pol_list`` for the full list of matrices.

    Returns
    -------
    sympy scalar expression
    """
    if not 0 <= gamma_pol_index <= 5:
        raise ValueError(f"gamma_pol_index must be 0–5, got {gamma_pol_index}.")
    Gamma_pol_choice = Gamma_pol_list[gamma_pol_index]
    den_K_choice = den_K_list[gamma_pol_index]
    num_K = sym.trace(Gamma_pol_choice @ (-I * pslash + mN * Id_4) @ operator @ (-I * pslash + mN * Id_4)).simplify(
        rational=True
    )
    return (
        (num_K / den_K_choice)
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

    where δ_X = 0 for V and S, and δ_X = 1 for A, T and P.  We identify
    C-parity by comparing the original cgmat with the one obtained by
    reversing the derivative indices.

    Parameters
    ----------
    cgmat : ndarray, shape (4,)*n
    X : str

    Returns
    -------
    int or str
        +1, -1, or ``'mixed'`` if neither eigenstate.
    """
    if X not in ['V', 'A', 'T', 'S', 'P']:
        raise ValueError("X must be 'V', 'A', 'T', 'S' or 'P'.")
    cgmat = np.round(cgmat, decimals=13)
    n = cgmat.ndim
    cgmat_C = np.empty(shape=np.shape(cgmat))
    for indices in it.product(range(4), repeat=n):
        # Reverse the derivative indices.
        # S/P carry no Dirac index so all n indices are reversed.
        # V/A keep the first (Dirac) index fixed and reverse the rest.
        # T keeps the first two (σ_{μν}) fixed and reverses the rest.
        if X in ('S', 'P'):
            cgmat_C[indices] = cgmat[indices[::-1]]
        elif X in ('V', 'A'):
            cgmat_C[indices] = cgmat[(indices[0],) + indices[-1:0:-1]]
        else:  # T
            cgmat_C[indices] = cgmat[(indices[0], indices[1]) + indices[-1:1:-1]]

    if (cgmat == cgmat_C).all():
        Cp = 1
    elif (cgmat == -cgmat_C).all():
        Cp = -1
    else:
        return "mixed"

    # Include the overall sign factor (−1)^{n + δ_X}.
    # δ = 0 for V, S;  δ = 1 for A, T, P.
    if X in ("V", "S"):
        Cp *= (-1) ** n
    elif X in ("A", "T", "P"):
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
        Number of derivative indices +1 (= number of tensor-product factors)
        (the tensor accounts for two Dirac indices, so the full cgmat will have shape (4,)*(n+1)).

    Returns
    -------
    ndarray, shape (4,)*(n+1)
    """
    cg_remapped = np.zeros(shape=(4,) * (n + 1))
    # The order of the 6 independent combinations is fixed by the
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
    operator_folder: str,
    max_n: int,
    verbose: bool = False,
    cg_database: str | None = None,
    include_SP: bool = False,
) -> None:
    """Build a database of H(4)-irreducible lattice operators up to *max_n* indices.

    For each combination of structure X and index count n, the function loads
    the relevant CG matrices from the database, loops over all irreps and
    multiplicity blocks, and saves each resulting operator to a .npy file
    whose filename encodes the full operator metadata.

    Parameters
    ----------
    operator_folder : str
        Directory where operator .npy files will be written.
    max_n : int
        Maximum number of indices for V and A operators.  T operators
        automatically get one extra index (σ_{μν} contributes two).
        For S and P (when *include_SP* is True) this is the maximum number
        of derivative indices.
    verbose : bool
        Print progress messages.
    cg_database : str or None
        Path to the CG-coefficient database.  ``None`` → use the bundled data.
    include_SP : bool
        When ``True``, also generate scalar (S, Γ = 𝟙) and pseudoscalar
        (P, Γ = γ₅) operators in addition to the default V, A, T set.
        Defaults to ``False``.
    """
    if max_n < 2:
        raise ValueError("max_n must be at least 2.")
    Path(operator_folder).mkdir(parents=True, exist_ok=True)

    X_list = ['V', 'A', 'T', 'S', 'P'] if include_SP else ['V', 'A', 'T']
    n_list = list(range(2, max_n + 1))
    iop = 1  # global operator counter (1-based, matches the printed catalogue)

    for n in n_list:
        if verbose:
            label = f"V{n}, A{n}, T{n+1}, S{n-1}, P{n-1}" if include_SP else f"V{n}, A{n}, T{n+1}"
            print(f"\nConstructing operators {label}...\n")

        for X in tqdm(X_list, disable=not verbose):
            # The first irrep encodes the Dirac structure; all derivative
            # indices transform in the fundamental (4,1) representation.
            # S and P have a 1-dimensional leading irrep, so their cgmat has
            # one fewer index than the chosen_irreps list.
            chosen_irreps = []
            if X == 'V':
                chosen_irreps.append((4, 1))
            elif X == 'A':
                chosen_irreps.append((4, 4))
            elif X == 'T':
                chosen_irreps.append((6, 1))
            elif X == 'S':
                chosen_irreps.append((1, 1))
            elif X == 'P':
                chosen_irreps.append((1, 4))
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
                        if X in ('V', 'A'):
                            cg_mat = cg_remapping(block[:, icol], len(chosen_irreps))
                        elif X in ('S', 'P'):
                            # Leading irrep is 1-dimensional and contributes no
                            # Lorentz index; remap only the n-1 derivative indices.
                            cg_mat = cg_remapping(block[:, icol], len(chosen_irreps) - 1)
                        else:  # T
                            cg_mat = cg_remapping_T(block[:, icol], len(chosen_irreps))
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

        # Tensor - irrep (6,2), block 1 (C=-1)
        opT4 =   Operator_from_database(233) # = 1,3,{1,4} + 1,4,{1,3} - 2,3,{2,4} - 2,4,{2,3}
        opT5 =   Operator_from_database(234) # = 1,2,{1,4} + 1,4,{1,2} + 2,3,{3,4} - 3,4,{2,3}
        opT6 = - Operator_from_database(235) # = 1,2,{2,4} - 1,3,{3,4} - 2,4,{1,2} + 3,4,{1,3}
        opT7 =   Operator_from_database(236) # = 1,2,{1,3} + 1,3,{1,2} + 2,4,{3,4} + 3,4,{2,4}
        opT8 = - Operator_from_database(237) # = 1,2,{2,3} - 1,4,{3,4} - 2,3,{1,2} - 3,4,{1,4}
        opT9 = - Operator_from_database(238) # = 1,3,{2,3} - 1,4,{2,4} + 2,3,{1,3} - 2,4,{1,4}

        # Set custom labels
        opV1.id = "V1"
        opV2.id = "V2"
        opV3.id = "V3"
        opV4.id = "V4"
        opA1.id = "A1"
        opA2.id = "A2"
        opA3.id = "A3"
        opA4.id = "A4"
        opT1.id = "T1"
        opT2.id = "T2"
        opT3.id = "T3"
        opT4.id = "T4"
        opT5.id = "T5"
        opT6.id = "T6"
        opT7.id = "T7"
        opT8.id = "T8"
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
    if X not in ['V', 'A', 'T', 'S', 'P']:
        raise ValueError("X must be 'V', 'A', 'T', 'S' or 'P'.")
    # The float → LaTeX lookup table, built on the first call (see kinematic_data).
    numerics_to_latex_conv = latex_conversion_table()
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


def irrep_count_table(n_der: int, split: bool = False) -> None:
    """Print a table of cumulative irrep counts across all starting structures.

    For each of the 20 H(4) irreps lists:

    * **Label** — LaTeX name of the irrep.
    * **Count** — cumulative number of appearances summed over all five starting
      irreps ``{(1,1), (1,4), (4,1), (4,4), (6,1)}`` from 0 derivatives up to
      and including *n_der*.
    * **Diff**  — the count added specifically at the *n_der* level (i.e.
      Count(*n_der*) − Count(*n_der* − 1)).  Blank when zero.

    Parameters
    ----------
    n_der : int
        Number of derivatives.
    split : bool
        If ``False`` (default) print a single 3-column table with all 20 irreps.
        If ``True`` split the 20 irreps into two groups of 10 and print them
        side by side as a 6-column table.
    """
    V = (4, 1)
    starting_irreps = [(1, 1), (1, 4), (4, 1), (4, 4), (6, 1)]
    n_irreps = len(rep_label_list)

    def _count_at_level(k: int) -> list[int]:
        c = [0] * n_irreps
        for si in starting_irreps:
            mults = get_multiplicities(*(si,) + (V,) * k)
            for i in range(n_irreps):
                c[i] += mults[i]
        return c

    # Cumulative count up to n_der
    counts = [0] * n_irreps
    for k in range(n_der + 1):
        for i, v in enumerate(_count_at_level(k)):
            counts[i] += v

    # Diff = count added specifically at this level
    diff = _count_at_level(n_der)

    def _diff_str(d: int) -> str:
        return f"+{d}" if d > 0 else (str(d) if d < 0 else "")

    col_w = max(len(name) for name in rep_latex_names) + 2
    print(f"\nIrrep count table for n_der = {n_der}:\n")

    if not split:
        header = f"{'Label':<{col_w}} {'Count':>5}  {'Diff':>5}"
        sep = "-" * len(header)
        print(sep)
        print(header)
        print(sep)
        for i, label in enumerate(rep_latex_names):
            print(f"{label:<{col_w}} {counts[i]:>5}  {_diff_str(diff[i]):>5}")
        print(sep)
    else:
        half = n_irreps // 2
        gap = "    "
        col_block = f"{'Label':<{col_w}} {'Count':>5}  {'Diff':>5}"
        header = col_block + gap + col_block
        sep = "-" * len(header)
        print(sep)
        print(header)
        print(sep)
        for row in range(half):
            left  = row
            right = row + half
            l_label, l_count, l_diff = rep_latex_names[left],  counts[left],  diff[left]
            r_label, r_count, r_diff = rep_latex_names[right], counts[right], diff[right]
            left_col  = f"{l_label:<{col_w}} {l_count:>5}  {_diff_str(l_diff):>5}"
            right_col = f"{r_label:<{col_w}} {r_count:>5}  {_diff_str(r_diff):>5}"
            print(left_col + gap + right_col)
        print(sep)


def decomposition_analysis(X: str, n_der: int, operator_dict: dict | None = None, verbose: bool = False, print_count_table: bool = False):
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
    print_count_table : bool
        If True, call :func:`irrep_count_table` with the default (3-column)
        layout after printing the analysis.

    Returns
    -------
    pandas.DataFrame
    """
    if not _PANDAS_AVAILABLE:
        raise ImportError("pandas is required for decomposition_analysis().")
    if X not in ['V', 'A', 'T', 'S', 'P']:
        raise ValueError("X must be 'V', 'A', 'T', 'S' or 'P'.")
    if not isinstance(n_der, int) or n_der < 1:
        raise ValueError("n_der must be a positive integer.")

    V = (4, 1)
    starting_irrep = (
        ((4, 1),) if X == 'V' else
        ((4, 4),) if X == 'A' else
        ((6, 1),) if X == 'T' else
        ((1, 1),) if X == 'S' else
        ((1, 4),)  # P
    )

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
                if X in ('V', 'A'):
                    n_idx = n_der + 1
                elif X == 'T':
                    n_idx = n_der + 2
                else:  # S, P: all indices are derivatives
                    n_idx = n_der
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

    if print_count_table:
        irrep_count_table(n_der)

    return df


################## PDF Catalogue ################

# Dirac structures in the order in which make_operator_database writes them; also the
# order in which they are listed in the default catalogue titles and file names.
_X_ORDER = ('V', 'A', 'T', 'S', 'P')

# Default selection of operator_show / h4lat-catalogue: V, A and T operators with one
# and two derivatives (IDs 1-280 of the bundled database).  The operators with three
# derivatives take most of the time, so they are written only on request.
_CATALOGUE_DEFAULT_X = ('V', 'A', 'T')
_CATALOGUE_DEFAULT_NDER = (1, 2)


def _derivatives_text(nders: list[int]) -> str:
    """'1 derivative', '3 derivatives', '1, 2 derivatives' (for catalogue titles)."""
    return "1 derivative" if nders == [1] else f"{', '.join(str(d) for d in nders)} derivatives"


def _catalogue_selection(
    X: str | Iterable[str] | None = None,
    nder: int | Iterable[int] | None = None,
    keys: str | Iterable[str] | None = None,
) -> tuple[list[tuple[str, int]], str, str]:
    """Parse the operator selection of :func:`operator_show`.

    The selection is either the product of the Dirac structures *X* and the numbers of
    derivatives *nder* (each defaulting to ``('V', 'A', 'T')`` and ``(1, 2)``), or an
    explicit list of *keys* such as ``['V2', 'V3', 'A3']``: a Dirac structure followed
    by the number of derivatives.  Note that a key counts **derivatives**, not Lorentz
    indices: ``'V2'`` is the vector operators with 2 derivatives, i.e. n = 3 indices.

    Parameters
    ----------
    X : str or iterable of str, optional
        Dirac structures among V, A, T, S, P.  Every letter counts, so ``'VA'``,
        ``'V,A'`` and ``['V', 'A']`` are the same selection (case-insensitive).
    nder : int or iterable of int, optional
        Numbers of derivatives (each ≥ 1).
    keys : str or iterable of str, optional
        Keys ``'<X><nder>'``; a string may hold several, separated by commas or
        spaces.  Cannot be combined with *X* or *nder*.

    Returns
    -------
    selection : list of (X, nder)
        The selected pairs without duplicates, in the order V, A, T, S, P and
        increasing derivatives.
    title : str
        Default title of the catalogue, e.g.
        ``'Available Operators: X = V, A, T with 1, 2 derivatives'`` or
        ``'Available Operators: V with 2, 3 derivatives; A with 3 derivatives'``.
    name : str
        Default file name (without extension), e.g. ``'operator_catalogue_VAT_nder12'``
        or ``'operator_catalogue_V2_V3_A3'``.

    Raises
    ------
    ValueError
        If *keys* is combined with *X* or *nder*, or a value is not valid.
    """
    if keys is not None and (X is not None or nder is not None):
        raise ValueError("Select the operators either with keys or with X and nder, not both.")

    if keys is not None:
        # A single string may hold one key ("V2") or several ("V2,V3" or "V2 V3").
        keys = [keys] if isinstance(keys, str) else list(keys)
        tokens = [token for key in keys for token in re.split(r"[,\s]+", str(key).strip()) if token]
        if not tokens:
            raise ValueError("keys is empty: give at least one key such as 'V2'.")
        pairs = []
        for token in tokens:
            match = re.fullmatch(r"([VATSP])(\d+)", token.upper())
            if match is None or int(match.group(2)) < 1:
                raise ValueError(
                    f"Invalid key {token!r}: expected a Dirac structure (V, A, T, S or P) followed by "
                    "the number of derivatives (at least 1), e.g. 'V2'."
                )
            pairs.append((match.group(1), int(match.group(2))))
    else:
        # Dirac structures: every letter of every item, ignoring commas and spaces.
        X = _CATALOGUE_DEFAULT_X if X is None else X
        letters = X if isinstance(X, str) else "".join(str(x) for x in X)
        X_list = [c for c in letters.upper() if c not in ", "]
        invalid = sorted({c for c in X_list if c not in _X_ORDER})
        if invalid or not X_list:
            raise ValueError(f"X must be made of V, A, T, S and P, got {X!r}.")

        # Numbers of derivatives: positive integers (bool is excluded, although it is an int).
        nder = _CATALOGUE_DEFAULT_NDER if nder is None else nder
        nder_list = [nder] if isinstance(nder, (int, np.integer)) else list(nder)
        if not nder_list or any(
            isinstance(d, bool) or not isinstance(d, (int, np.integer)) or d < 1 for d in nder_list
        ):
            raise ValueError(f"nder must be a positive integer or a list of them, got {nder!r}.")

        pairs = [(x, int(d)) for x in X_list for d in nder_list]

    # Drop duplicates and sort: the same selection then gives the same title and file name
    # however it was written (as keys or as X and nder, in any order).
    selection = sorted(set(pairs), key=lambda pair: (_X_ORDER.index(pair[0]), pair[1]))
    X_sel = list(dict.fromkeys(x for x, _ in selection))
    nder_sel = sorted({d for _, d in selection})

    if set(selection) == {(x, d) for x in X_sel for d in nder_sel}:
        # The selection is a full product X × nder: name it by its structures and derivatives.
        title = f"Available Operators: X = {', '.join(X_sel)} with {_derivatives_text(nder_sel)}"
        name = f"operator_catalogue_{''.join(X_sel)}_nder{''.join(str(d) for d in nder_sel)}"
    else:
        # Otherwise list, for each structure, its numbers of derivatives (title) or the keys (name).
        per_X = {x: [d for y, d in selection if y == x] for x in X_sel}
        title = "Available Operators: " + "; ".join(f"{x} with {_derivatives_text(ds)}" for x, ds in per_X.items())
        name = "operator_catalogue_" + "_".join(f"{x}{d}" for x, d in selection)

    return selection, title, name


def _kfactor_worker(cgmat: np.ndarray, X: str, gamma_pol_index: int) -> sym.Expr:
    """Kinematic factor of one operator, computed in a worker process of operator_show.

    The same computation as ``Operator.K``.  It is a module-level function so that the
    worker processes can find it by name (a lambda or a bound method cannot be pickled).
    """
    return Kfactor_from_diracO(diracO_from_cgmat(cgmat, X), gamma_pol_index)


def _compute_kinematic_factors(operators: list[Operator], n_jobs: int = 1, verbose: bool = True) -> None:
    """Compute the kinematic factor K of every operator, with a progress bar.

    ``Operator.K`` is a cached property, computed on first access; this function fills
    that cache for all *operators* up front, which is the slow part of the catalogue
    (SymPy).  Operators whose K is already cached are skipped.

    Parameters
    ----------
    operators : list[Operator]
    n_jobs : int, optional
        Number of worker processes (default 1: compute in this process).  With
        n_jobs > 1 the operators are spread over a ``ProcessPoolExecutor``, and each
        result is stored in the operator's cache exactly as an access to ``op.K``
        would have stored it.
    verbose : bool, optional
        Show the progress bar (default True).
    """
    # Operators whose K still has to be computed.  functools.cached_property keeps its
    # value in the instance __dict__, under the name of the property.
    todo = [op for op in operators if "K" not in op.__dict__]
    progress = tqdm(
        total=len(operators),
        initial=len(operators) - len(todo),
        desc="Kinematic factors",
        unit="op",
        disable=not verbose,
    )
    try:
        if n_jobs == 1 or len(todo) < 2:
            # Serial: in operator order, the postfix shows the group (e.g. V2) and the ID.
            for op in todo:
                progress.set_postfix_str(f"{op.X}{op.nder}, ID {op.id}", refresh=False)
                _ = op.K  # computed on first access, then cached
                progress.update()
        else:
            # Parallel: the results arrive in the order in which they finish.
            with ProcessPoolExecutor(max_workers=min(n_jobs, len(todo))) as pool:
                futures = {pool.submit(_kfactor_worker, op.cgmat, op.X, op.gamma_pol_index): op for op in todo}
                for future in as_completed(futures):
                    op = futures[future]
                    # Fill the cache of the cached_property K, as op.K would have done.
                    op.__dict__["K"] = future.result()
                    progress.set_postfix_str(f"{op.X}{op.nder}, ID {op.id}", refresh=False)
                    progress.update()
    finally:
        progress.close()


def operator_show(
    operator_database: str | None = None,
    title: str | None = None,
    author: str = "E.T.",
    doc_name: str | None = None,
    verbose: bool = True,
    show: bool = True,
    remove_pdf: bool = False,
    clean_tex: bool = True,
    *,
    X: str | Iterable[str] | None = None,
    nder: int | Iterable[int] | None = None,
    keys: str | Iterable[str] | None = None,
    n_jobs: int | None = 1,
) -> Path:
    """Produce a PDF catalogue of a selection of the operators in *operator_database*
    and, optionally, open it on screen.

    The catalogue is organised by Dirac structure *X* and index count *n*.
    Each section covers one (X, n) combination; each subsection covers one
    H(4) irrep / multiplicity block and lists every operator with its
    symbolic expression and kinematic factor.

    By default only the V, A and T operators with one and two derivatives are
    written; the operators with more derivatives take most of the time (computing
    their kinematic factors with SymPy), so they are written only on request, with
    *X* and *nder* or with *keys*.  Operator IDs are those of the database and never
    change: a partial catalogue simply skips some of them.  Sections always come in
    database order (increasing ID), whatever the order of the selection.

    The command ``h4lat-catalogue`` (installed with the package) calls this
    function from the shell; see ``h4lat-catalogue --help``.

    Parameters
    ----------
    operator_database : str, optional
        Path to an operator database directory produced by
        :func:`make_operator_database`.  When *None* the bundled dataset
        shipped with the package is used.
    title : str, optional
        Title printed on the first page of the PDF.  Defaults to a string
        stating the selected operators, e.g.
        ``"Available Operators: X = V, A, T with 1, 2 derivatives"``.
    author : str, optional
        Author string printed on the title page (default ``"E.T."``).
    doc_name : str, optional
        Path (without extension) of the generated ``.tex`` / ``.pdf``
        files; its folder must exist.  Defaults to a name stating the selected
        operators, in the working directory: ``"operator_catalogue_VAT_nder12"``
        by default, ``"operator_catalogue_V2_V3_A3"`` for ``keys=["V2", "V3", "A3"]``.
    verbose : bool, optional
        Print the selection, progress bars over the operators and progress
        messages to stdout (default ``True``).
    show : bool, optional
        Open the PDF with ``xdg-open`` after generation (default ``True``).
    remove_pdf : bool, optional
        Delete the PDF file after showing it — useful when the user only
        needs a one-off preview (default ``False``).
    clean_tex : bool, optional
        Remove the intermediate ``.tex`` and auxiliary LaTeX files after PDF
        generation (default ``True``).
    X : str or iterable of str, optional (keyword only)
        Dirac structures to include, among ``'V'``, ``'A'``, ``'T'``, ``'S'``,
        ``'P'`` (default ``('V', 'A', 'T')``; ``'VA'`` is the same as ``['V', 'A']``).
    nder : int or iterable of int, optional (keyword only)
        Numbers of derivatives to include (default ``(1, 2)``).  The catalogue
        holds every combination of *X* and *nder*.
    keys : str or iterable of str, optional (keyword only)
        Explicit groups ``'<X><number of derivatives>'``, e.g. ``["V2", "V3", "A3"]``
        (a key counts derivatives, not indices: ``'V2'`` has n = 3).  Cannot be
        combined with *X* or *nder*.
    n_jobs : int or None, optional (keyword only)
        Number of processes computing the kinematic factors (default 1, in this
        process; ``None`` uses all CPU cores).  The result does not depend on it.
        In a script, call ``operator_show`` with ``n_jobs > 1`` under
        ``if __name__ == "__main__":``, as usual with multiprocessing.

    Returns
    -------
    pathlib.Path
        Path of the PDF file (already deleted if *remove_pdf* is True).

    Raises
    ------
    ValueError
        If the selection is invalid or none of the selected operators is in the
        database.  Selected groups missing from the database (e.g. ``'V4'``) are
        only reported with a warning.

    Examples
    --------
    >>> operator_show(show=False)                                # V, A, T with 1, 2 derivatives
    >>> operator_show(X="V", nder=[1, 2, 3], show=False)         # vector operators, up to 3 derivatives
    >>> operator_show(keys=["V2", "V3", "A3"], n_jobs=None, show=False)  # a selection, on all cores
    """

    # pylatex is an optional dependency; raise a helpful error if missing
    if not _PYLATEX_AVAILABLE:
        raise ImportError(
            "pylatex is required for operator_show(). "
            "Install it with: pip install pylatex"
        )
    start = time.time()

    # The selected (X, number of derivatives) groups, and the default title / file name
    # that describe them.  This also validates X, nder and keys.
    selection, default_title, default_name = _catalogue_selection(X=X, nder=nder, keys=keys)
    if title is None:
        title = default_title
    if doc_name is None:
        doc_name = default_name

    # Number of processes for the kinematic factors: None means all CPU cores.
    if n_jobs is None:
        n_jobs = os.cpu_count() or 1
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)) or n_jobs < 1:
        raise ValueError(f"n_jobs must be a positive integer or None, got {n_jobs!r}.")

    # Build the nested operators dict from the chosen (or default) database.
    # OperatorDict_from_database falls back to _BUNDLED_OPERATOR_DATABASE when
    # operator_database is None, so no special-casing is needed here.
    operators_dict = OperatorDict_from_database(operator_database)

    # The (n, X) groups of the database that belong to the selection.  The dict is filled
    # in order of increasing operator ID, so the sections keep the database order.
    selected = set(selection)
    sections = [(n, x) for n, x in operators_dict if (x, _n_der(n, x)) in selected]
    available = [f"{x}{_n_der(n, x)}" for n, x in operators_dict]
    if not sections:
        raise ValueError(
            f"None of the selected operators ({', '.join(f'{x}{d}' for x, d in selection)}) is in the "
            f"operator database, which holds: {', '.join(available)}."
        )
    # Selected groups that the database does not hold are reported, not fatal.
    missing = [f"{x}{d}" for x, d in selection if f"{x}{d}" not in available]
    if missing:
        warnings.warn(f"Not in the operator database, left out of the catalogue: {', '.join(missing)}.", stacklevel=2)

    # The PDF file that will be produced on disk
    file_name = f"{doc_name}.pdf"

    # All the operators that will be printed, in catalogue order.
    operators = [op for key in sections for block in operators_dict[key].values() for op in block]

    if verbose:
        groups = ", ".join(f"{x}{_n_der(n, x)}" for n, x in sections)
        print(f"Operator catalogue: {groups} ({len(operators)} operators) -> {file_name}")
        if n_jobs > 1:
            print(f"Computing the kinematic factors on {n_jobs} processes")

    # --- 1. Kinematic factors: the slow part (SymPy), done for all operators up front ---

    _compute_kinematic_factors(operators, n_jobs=n_jobs, verbose=verbose)

    # --- 2. LaTeX document ---

    # Instantiate the LaTeX document
    doc = Document(
        default_filepath=f'{doc_name}.tex',
        documentclass='article',
        page_numbers=False,
    )

    # Build the document preamble (title, author, date)
    doc.preamble.append(Command("title", title))
    doc.preamble.append(Command("author", author))
    doc.preamble.append(Command("date", NoEscape(r"\today")))
    doc.append(NoEscape(r"\maketitle"))
    doc.append(NoEscape(r"\newpage"))

    # Use a smaller font so that long operator expressions fit on one line
    doc.append(Command('fontsize', arguments=['8', '12']))

    # Second progress bar: writing the rows (the operator expressions are built here;
    # the first one also builds the float → LaTeX lookup table, a few seconds).
    progress = tqdm(total=len(operators), desc="LaTeX rows", unit="op", disable=not verbose)

    # Loop over the selected (n, x) groups (x: Dirac structure) to build one LaTeX section per group
    for n, x in sections:

        # Section title shows the shared Dirac structure, index count and number of derivatives
        n_der = _n_der(n, x)
        section = Section(f"X={x}, n={n} ({n_der} derivative{'s' if n_der != 1 else ''})", numbering=False)

        # Loop over the irrep / multiplicity-block sub-keys within this (n, x)
        for irrep, imul in operators_dict[(n, x)].keys():

            # Collect per-operator properties to determine the block-wide value
            # (or flag it as 'mixed' when operators in the block disagree)
            C_list    = [op.C    for op in operators_dict[(n, x)][(irrep, imul)]]
            tr_list   = [op.tr   for op in operators_dict[(n, x)][(irrep, imul)]]
            symm_list = [op.symm for op in operators_dict[(n, x)][(irrep, imul)]]

            # If every operator shares the same value use it; otherwise mark 'mixed'
            C    = C_list[0]    if all_equal(C_list)    else 'mixed'
            tr   = tr_list[0]   if all_equal(tr_list)   else 'mixed'
            symm = symm_list[0] if all_equal(symm_list) else 'Mixed Symmetry'

            # Subsection title encodes the irrep, block index, and shared symmetries
            subsection = Subsection(
                f"{irrep} Block {imul}: C = {C}, Trace {tr}, {symm} ",
                numbering=False,
            )

            # One alignat environment holds all operator / kinematic-factor rows
            agn = Alignat(numbering=False, escape=False)

            # Append each operator in the block to the math environment
            for op in operators_dict[(n, x)][(irrep, imul)]:

                # Label row: the operator's integer id
                agn.append(r"\text{Operator " + str(op.id) + r"}&\\")

                # Operator expression row
                agn.append(
                    r"\!"*20
                    + r" O_{}^{} &= {} \\".format(
                        op.index_block,
                        '{' + f"{x}{irrep},{imul}" + '}',
                        op,
                    )
                )

                # Kinematic factor row (triple backslash gives extra vertical space)
                agn.append(
                    r"\!"*20
                    + r" K_{}^{} &= {} \\\\\\".format(
                        op.index_block,
                        '{' + f"{x}{irrep},{imul}" + '}',
                        op.latex_K,
                    )
                )
                progress.update()

            # Attach the math block to the subsection, then the subsection to the section
            subsection.append(agn)
            section.append(subsection)

        # Attach the completed section and start a new page
        doc.append(section)
        doc.append(NoEscape(r"\newpage"))

    progress.close()

    # --- 3. PDF generation ---

    if verbose:
        print(f"Compiling {doc_name}.tex with LaTeX ...")

    # Compile the LaTeX source; clean_tex controls removal of auxiliary files
    doc.generate_pdf(doc_name, clean_tex=clean_tex)

    # --- Optional on-screen display ---

    if show:
        # Open the PDF with the system's default viewer
        subprocess.call(["xdg-open", file_name])
        # Brief pause so the viewer has time to render before a potential deletion
        time.sleep(1.5)
        if verbose:
            print("Operator catalogue shown")

    # --- Optional cleanup ---

    if remove_pdf:
        # Delete the PDF from disk
        Path(file_name).unlink()
        if verbose:
            print("Operator catalogue removed from the device")
    elif verbose:
        print(f"Operator catalogue written to {file_name} in {time.time() - start:.0f} s")

    return Path(file_name)


################## HDF5 I/O #####################


def write_operator(group, operator: Operator) -> None:
    """Serialise an Operator into an HDF5 group.

    Stores the ``cgmat`` as a compressed dataset and all scalar attributes
    (``id``, ``X``, ``irrep``, ``block``, ``index_block``, ``gamma_pol_index``)
    as group attributes.  None-valued attributes (``id``, ``irrep``, ``block``,
    ``index_block`` can each legitimately be ``None`` — e.g. for operators
    produced by ``Operator.__add__``/``__sub__``) are recorded via a
    ``_is_none`` flag so that they can be correctly reconstructed by
    ``read_operator``.  ``h5py`` attributes have no native representation for
    ``None``, so writing it directly would raise a ``TypeError``; the flag
    sidesteps that entirely.

    Parameters
    ----------
    group : h5py.Group
    operator : Operator
    """
    if not _H5PY_AVAILABLE:
        raise ImportError("h5py is required for HDF5 I/O.")
    op_group = group.create_group("operator")

    # `id` can be None (see docstring above), so it goes through the same
    # _is_none pattern already used for irrep/block/index_block below.
    if operator.id is None:
        op_group.attrs["id_is_none"] = True
    else:
        op_group.attrs["id"] = operator.id

    op_group.attrs["X"] = operator.X
    op_group.create_dataset("cgmat", data=operator.cgmat, compression="gzip")

    # gamma_pol_index selects which of the six polarisation projectors was
    # used to compute K (see Operator.set_polarization_matrix); it changes
    # the physical content of K, so it must round-trip along with cgmat.
    op_group.attrs["gamma_pol_index"] = operator.gamma_pol_index

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

    op_group = group["operator"]

    if op_group.attrs.get("id_is_none", False):
        operator_id = None
    else:
        raw_id = op_group.attrs["id"]
        if isinstance(raw_id, bytes):
            raw_id = raw_id.decode()
        operator_id = int(raw_id) if isinstance(raw_id, (int, np.integer)) else str(raw_id)

    # Cast each entry to a plain int: h5py returns attrs["irrep"] as a NumPy
    # int64 array, and tuple(...)-ing it directly would leave np.int64
    # elements instead of plain ints (unlike block/index_block below, which
    # were already cast).
    irrep = (
        None
        if op_group.attrs.get("irrep_is_none", False)
        else tuple(int(x) for x in op_group.attrs["irrep"])
    )
    block = None if op_group.attrs.get("block_is_none", False) else int(op_group.attrs["block"])
    index_block = None if op_group.attrs.get("index_block_is_none", False) else int(op_group.attrs["index_block"])

    # gamma_pol_index is optional: files written before it was persisted do
    # not carry it.  A missing attribute is left out of the constructor call,
    # so the Operator default applies.
    kwargs: dict = {}
    if "gamma_pol_index" in op_group.attrs:
        kwargs["gamma_pol_index"] = int(op_group.attrs["gamma_pol_index"])

    return Operator(
        cgmat=np.array(op_group["cgmat"]),
        id=operator_id,
        X=str(op_group.attrs["X"]),
        irrep=irrep,
        block=block,
        index_block=index_block,
        **kwargs,
    )


# Width used to zero-pad the per-operator slot names created by
# operator_list_to_h5 (e.g. "op_000042").  Six digits comfortably covers any
# realistic operator database (up to 999,999 operators) while keeping slot
# names short; it is not a hard limit, just the padding used for sorting.
_H5_SLOT_DIGITS = 6


def operator_list_to_h5(operators: list[Operator], file_path: str) -> None:
    """Serialise a list of Operators into a single HDF5 file.

    Each operator is written into its own top-level group inside *file_path*
    via ``write_operator``.  Groups are named ``"op_000000"``, ``"op_000001"``,
    … using the operator's position in *operators*, NOT ``operator.id`` — id
    can be ``None`` (see ``write_operator``) or repeated across the list, so
    it cannot be relied on as a unique key, whereas the position always is.
    Zero-padding the index keeps the group names sorting lexicographically in
    the same order as *operators*, which is what ``h5_to_operator_list``
    relies on to restore the original list order.

    Parameters
    ----------
    operators : list[Operator]
        Operators to save, in the order they should be restored.
    file_path : str
        Destination path of the ``.h5`` file.  Parent directories are
        created if missing; an existing file at this path is overwritten.
    """
    if not _H5PY_AVAILABLE:
        raise ImportError("h5py is required for HDF5 I/O.")

    # Create parent directories (if any) so callers can pass a path like
    # "my_folder/operators.h5" into a folder that does not exist yet.
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # "w" truncates/creates the file, so re-running this on the same path
    # always produces a fresh, self-consistent snapshot of *operators*.
    with h5py.File(file_path, "w") as f:
        for i, operator in enumerate(operators):
            slot = f.create_group(f"op_{i:0{_H5_SLOT_DIGITS}d}")
            write_operator(slot, operator)


def h5_to_operator_list(file_path: str) -> list[Operator]:
    """Reload a list of Operators previously saved by ``operator_list_to_h5``.

    Parameters
    ----------
    file_path : str
        Path to an ``.h5`` file written by ``operator_list_to_h5``.

    Returns
    -------
    list[Operator]
        The operators, in the same order they were originally saved.
    """
    if not _H5PY_AVAILABLE:
        raise ImportError("h5py is required for HDF5 I/O.")

    if not Path(file_path).is_file():
        raise ValueError(f"Path does not exist: {file_path}")

    with h5py.File(file_path, "r") as f:
        # Sorting the (zero-padded) group names lexicographically recovers
        # the exact save order written by operator_list_to_h5.
        slot_names = sorted(f.keys())
        return [read_operator(f[name]) for name in slot_names]


###################### Execution as Main ############################

if __name__ == "__main__":
    # Default to max_n=4, matching the bundled operator database shipped with the package.
    # Override by passing an integer as the first CLI argument: python moments_operator.py 3
    # Pass "SP" as the second CLI argument to also generate S and P operators.
    max_n = 4
    include_SP = False
    if len(sys.argv) > 1:
        try:
            max_n = int(sys.argv[1])
        except ValueError:
            pass
    if len(sys.argv) > 2 and sys.argv[2].upper() == "SP":
        include_SP = True
    make_operator_database(operator_folder="operator_database", max_n=max_n, verbose=True, include_SP=include_SP)
