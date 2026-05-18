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
  - Polarisation projector Γ_pol for an unpolarised moving nucleon.
  - Kinematic denominator den_K used by the Operator class.
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

# Polarisation projector for an unpolarised nucleon moving in the 3-direction:
#   Γ_pol = ½(1 + γ₄)(1 − iγ₁γ₂).
# The first factor projects onto positive energy; the second selects
# unpolarised states with definite helicity along the 3-axis.
Gamma_pol = 0.5 * (Id_4 + gamma4) @ (Id_4 - I * gamma1 @ gamma2)
Gamma_pol_s = sym.Symbol("Gamma_pol")


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

# Denominator of the kinematic normalisation factor K:
#   den_K = 2E Tr[Γ_pol (−i p_slash + m_N)].
den_K = 2 * E * sym.trace(Gamma_pol * (-I * pslash + mN * Id_4)).simplify(rational=True)


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
