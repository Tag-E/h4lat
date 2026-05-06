"""
Utility functions shared across h4lat modules.

These are elementary combinatorial helpers needed by the CG-coefficient
and operator-construction machinery.  They are intentionally kept free of
heavy dependencies so that the module can be imported in any context.
"""

import itertools as it


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

    Uses Newton's method (integer square-root iteration) to avoid the
    floating-point inaccuracies of ``math.sqrt``.  Needed when deciding
    whether a coefficient ``num / sqrt(den)`` already appears in simplified
    form in the LaTeX-conversion dictionary.

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

    References
    ----------
    Algorithm credit: https://stackoverflow.com/a/2489519
    """
    if type(apositiveint) is not int or apositiveint < 2:
        raise ValueError("Input must be an integer > 1.")
    x = apositiveint // 2
    seen = set([x])
    while x * x != apositiveint:
        x = (x + (apositiveint // x)) // 2
        if x in seen:
            return False
        seen.add(x)
    return True


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
