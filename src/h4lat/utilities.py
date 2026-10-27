######################################################
## utilities.py  (h4lat package module)             ##
## created by Emilio Taggi - 2025/03/04             ##
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
Utility functions shared across h4lat modules.

These are elementary combinatorial helpers needed by the CG-coefficient
and operator-construction machinery.  They are intentionally kept free of
heavy dependencies so that the module can be imported in any context.
"""

import itertools as it
import math


def parity(permutation: tuple) -> int:
    """Return the parity of a permutation: +1 (even) or -1 (odd).

    The parity is computed via cycle counting:
        parity = (-1)^(n - c)
    where n is the length and c is the number of disjoint cycles (including
    fixed points).  This avoids the O(n²) transposition-counting approach.

    Parameters
    ----------
    permutation : tuple
        A permutation of {0, 1, …, n-1}, as produced by itertools.permutations.

    Returns
    -------
    int
        +1 for an even permutation, -1 for an odd one.

    References
    ----------
    Algorithm credit: https://stackoverflow.com/a/1504565
    """
    permutation = list(permutation)
    length = len(permutation)
    elements_seen = [False] * length
    cycles = 0
    for index, already_seen in enumerate(elements_seen):
        if already_seen:
            continue
        cycles += 1
        current = index
        # Traverse the cycle starting at `current`.
        while not elements_seen[current]:
            elements_seen[current] = True
            current = permutation[current]
    return (-1) ** ((length - cycles) % 2)


def is_square(apositiveint: int) -> bool:
    """Return True if *apositiveint* is a perfect square, False otherwise.

    ``math.isqrt(n)`` is the exact integer square root ⌊√n⌋, computed with integer
    arithmetic only: unlike ``math.sqrt``, which goes through a float and can round
    the wrong way for large n, it is exact for integers of any size.  n is a perfect
    square exactly when squaring ⌊√n⌋ gives n back.  Needed when deciding whether a
    coefficient ``num / sqrt(den)`` already appears in simplified form in the
    LaTeX-conversion dictionary.

    Parameters
    ----------
    apositiveint : int
        An integer strictly greater than 1.

    Returns
    -------
    bool

    Raises
    ------
    ValueError
        If the input is not an integer > 1.
    """
    # `type(...) is not int` (not isinstance) also rejects bool, a subclass of int.
    if type(apositiveint) is not int or apositiveint < 2:
        raise ValueError("Input must be an integer > 1.")
    return math.isqrt(apositiveint) ** 2 == apositiveint


def all_equal(iterable) -> bool:
    """Return True if all elements of *iterable* are equal.

    Uses itertools.groupby for a short-circuit, O(1)-extra-space check.

    Parameters
    ----------
    iterable : iterable
        Any iterable.

    Returns
    -------
    bool

    References
    ----------
    Algorithm credit: https://stackoverflow.com/a/3844832
    """
    g = it.groupby(iterable)
    return next(g, True) and not next(g, False)
