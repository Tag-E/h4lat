######################################################
## kinematic_data.py  (h4lat package module)        ##
## created by Emilio Taggi - 2025/01/31             ##
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
Kinematic data for nucleon matrix-element operators on the lattice.

Contains:
  - Dirac gamma matrices (γ₁–γ₄, γ₅) in the Euclidean space,
    satisfying {γ_μ, γ_ν} = 2δ_μν.
  - Symbolic Euclidean 4-momentum p_μ = (p₁, p₂, p₃, iE).
  - A catalogue of six polarisation projectors Γ_pol, one for each pair of
    gamma matrices (γ_a γ_b with a < b). Each has the form
        Γ_pol = ½(1 + γ₄)(1 − i γ_a γ_b).
    The index ordering is:
        0: γ₁γ₂  (default — unpolarised nucleon, momentum along 3-axis)
        1: γ₁γ₃
        2: γ₂γ₃
        3: γ₁γ₄
        4: γ₂γ₄
        5: γ₃γ₄
  - Kinematic denominators den_K_list (one per polarisation choice) used by
    the Operator class.
  - Backward-compatible aliases ``Gamma_pol``, ``Gamma_pol_s``, and ``den_K``
    that point to index 0 (the original hardcoded default).
  - numerics_to_latex_conv: float → LaTeX string lookup table used
    when printing operator coefficients.
"""

from math import gcd

import numpy as np
import sympy as sym
from sympy import I

from .utilities import is_square

######################## Gamma Structures ################################

# Euclidean Dirac gamma matrices.
# The Euclidean metric is δ_μν (all positive), so {γ_μ, γ_ν} = 2δ_μν.
# γ₁, γ₂, γ₃ are anti-Hermitian; γ₄ and γ₅ are Hermitian.

gamma1 = sym.Matrix([[0, 0, 0, I], [0, 0, I, 0], [0, -I, 0, 0], [-I, 0, 0, 0]])
gamma2 = sym.Matrix([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]])
gamma3 = sym.Matrix([[0, 0, I, 0], [0, 0, 0, -I], [-I, 0, 0, 0], [0, I, 0, 0]])
gamma4 = sym.Matrix([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])

gamma1_s = sym.Symbol("gamma_1")
gamma2_s = sym.Symbol("gamma_2")
gamma3_s = sym.Symbol("gamma_3")
gamma4_s = sym.Symbol("gamma_4")

gamma_mu = [gamma1, gamma2, gamma3, gamma4]
gamma_mu_s = [gamma1_s, gamma2_s, gamma3_s, gamma4_s]

gamma5 = sym.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
gamma5_s = sym.Symbol("gamma_5")

Id_4 = sym.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

# ------------------------------------------------------------------
# Polarisation projectors
# ------------------------------------------------------------------
# Each projector has the form  Γ_pol = ½(1 + γ₄)(1 − i γ_a γ_b).
# The first factor ½(1 + γ₄) projects onto positive-energy states.
# The second factor (1 − i γ_a γ_b) selects a spin polarisation state
# characterised by the pair (a, b).
#
# We define one projector for every distinct ordered pair a < b, giving
# six choices indexed 0–5:
#
#   index  pair    Gamma_pol_s label
#   -----  ------  -----------------
#     0    γ₁ γ₂   Gamma_pol_12   ← default (original hardcoded choice)
#     1    γ₁ γ₃   Gamma_pol_13
#     2    γ₂ γ₃   Gamma_pol_23
#     3    γ₁ γ₄   Gamma_pol_14
#     4    γ₂ γ₄   Gamma_pol_24
#     5    γ₃ γ₄   Gamma_pol_34

# Internal helper: the six gamma-matrix pairs (ordered a < b, 1-based labels).
_gamma_pol_pairs = [
    (gamma1, gamma2),  # 0
    (gamma1, gamma3),  # 1
    (gamma2, gamma3),  # 2
    (gamma1, gamma4),  # 3
    (gamma2, gamma4),  # 4
    (gamma3, gamma4),  # 5
]
_gamma_pol_pair_labels = ["12", "13", "23", "14", "24", "34"]

Gamma_pol_list: list = [
    0.5 * (Id_4 + gamma4) @ (Id_4 - I * ga @ gb)
    for ga, gb in _gamma_pol_pairs
]
"""List of six 4×4 polarisation projector matrices, one per gamma pair.

``Gamma_pol_list[k]`` is  ½(1 + γ₄)(1 − i γ_a γ_b)  for pair ``k``.
See the module docstring for the index → pair mapping.
"""

Gamma_pol_s_list: list = [
    sym.Symbol(f"Gamma_pol_{label}")
    for label in _gamma_pol_pair_labels
]
"""Symbolic SymPy ``Symbol`` placeholders for each polarisation projector.

``Gamma_pol_s_list[k]`` is the symbol ``Gamma_pol_<ab>`` corresponding to
``Gamma_pol_list[k]``.  Use these when building symbolic expressions that
should show the projector as an unresolved symbol rather than a 4×4 matrix.
"""

# Backward-compatible aliases — index 0 reproduces the original behaviour.
Gamma_pol = Gamma_pol_list[0]
"""Default polarisation projector: ½(1 + γ₄)(1 − iγ₁γ₂).

This is an alias for ``Gamma_pol_list[0]`` and is kept for backward
compatibility with code that used the single hardcoded projector.
"""
Gamma_pol_s = sym.Symbol("Gamma_pol")
"""Symbolic placeholder for the default polarisation projector.

Kept for backward compatibility; the richer list ``Gamma_pol_s_list``
should be preferred in new code.
"""


######################## Kinematic Symbols ###############################

mN = sym.Symbol("m_N")
E = sym.Symbol("E(p)")

# Euclidean 4-momentum: p_μ = (p₁, p₂, p₃, iE).
# The factor of i in p₄ = iE arises from the Wick rotation p₀^Mink → ip₄^Eucl,
# keeping the on-shell relation p_μ² = −m_N² in Euclidean signature.
p1 = sym.Symbol("p_1")
p2 = sym.Symbol("p_2")
p3 = sym.Symbol("p_3")
p_mu = [p1, p2, p3, I * E]

pslash = np.einsum('ijk,i->jk', gamma_mu, p_mu)
pslash_s = sym.Symbol(r"\cancel{p}")

# ------------------------------------------------------------------
# Kinematic denominators
# ------------------------------------------------------------------
# For each polarisation choice the denominator of K is
#   den_K = 2E Tr[Γ_pol (−i p̸ + m_N)].
# We pre-compute one entry per projector so that Kfactor_from_diracO
# can look up the right denominator without recomputing it.

den_K_list: list = [
    2 * E * sym.trace(gp * (-I * pslash + mN * Id_4)).simplify(rational=True)
    for gp in Gamma_pol_list
]
"""Kinematic denominators, one per polarisation projector.

``den_K_list[k]`` = 2E · Tr[``Gamma_pol_list[k]`` · (−i p̸ + m_N)],
pre-simplified with SymPy's ``rational=True`` flag.
"""

# Backward-compatible alias — identical to the original hardcoded value.
den_K = den_K_list[0]
"""Default kinematic denominator: 2E Tr[Γ_pol (−i p̸ + m_N)] for ``Gamma_pol_list[0]``.

Kept for backward compatibility; prefer ``den_K_list`` in new code.
"""


######################## Numeric → LaTeX Conversion Dict #################

# Pre-build a lookup table mapping floats → LaTeX strings for fractions and
# 1/√n expressions, used by Operator.to_latex() when printing CG coefficients.
# is_square() filters perfect-square denominators since √(n²) = n is already
# covered by the integer/fraction case; gcd filtering keeps only reduced fractions.
max_int = 1000
numerics_to_latex_conv: dict = {}

for num in range(1, max_int + 1):
    for den in range(2, max_int + 1):
        if gcd(num, den) == 1:
            numerics_to_latex_conv[num / den] = r"\frac{" + str(num) + r"}{" + str(den) + r"}"
            numerics_to_latex_conv[np.round(num / den, decimals=13)] = r"\frac{" + str(num) + r"}{" + str(den) + r"}"

        if is_square(den) is False and num / np.sqrt(den) not in numerics_to_latex_conv:
            numerics_to_latex_conv[num / np.sqrt(den)] = r"\frac{" + str(num) + r"}{\sqrt{" + str(den) + r"}}"
            numerics_to_latex_conv[np.round(num / np.sqrt(den), decimals=13)] = (
                r"\frac{" + str(num) + r"}{\sqrt{" + str(den) + r"}}"
            )

    if num != 1:
        numerics_to_latex_conv[num] = str(num)
    else:
        numerics_to_latex_conv[num] = ""
