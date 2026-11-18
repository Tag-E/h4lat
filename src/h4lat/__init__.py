"""
h4lat — Clebsch-Gordan coefficients and lattice operators for the hypercubic group H(4)
and for the group W(F₄) of the 16-cell honeycomb lattice.

The group is chosen with the argument ``group`` of the functions that depend on it: by
default H(4), with ``group='honeycomb'`` W(F₄) (see ``get_group``).  The module-level
constants (``rep_label_list``, ``char_table``, …) are those of H(4); those of any group are
attributes of ``get_group(group)``.

Quick start
-----------
>>> from h4lat import cg_calc, get_multiplicities
>>> muls = get_multiplicities((4, 1), (4, 1))   # returns a 20-element list (one entry per H(4) irrep)
>>> cg = cg_calc((4, 1), (4, 1))                # loads bundled CG database
>>> cg.cg_dict                                  # {irrep_index: [cgmat, ...], ...}
>>> get_multiplicities((4, 1), (4, 1), group='honeycomb')   # 25 entries, one per W(F₄) irrep
"""

import importlib.metadata as _metadata

from .cg_calculator import (
    _BUNDLED_OPERATOR_DATABASE as OPERATOR_DATABASE,
)
from .cg_calculator import (
    CGmat_from_block,
    cg_calc,
    char_table,
    class_orders,
    get_multiplicities,
    irrep_dim,
    irrep_index,
    irrep_texname,
    latex_print_multiplicities,
    n_ele_h4,
    n_ele_s4,
    n_rep,
    print_CGmat,
    rep_dim_list,
    rep_label_list,
)
from .groups import AVAILABLE_GROUPS, LatticeGroup, get_group
from .kinematic_data import (
    Gamma_pol,
    Gamma_pol_list,
    Gamma_pol_s,
    Gamma_pol_s_list,
    den_K,
    den_K_list,
)
from .moments_operator import (
    AVAILABLE_PAPERS,
    C_parity,
    Kfactor_from_diracO,
    Operator,
    Operator_from_database,
    Operator_from_file,
    OperatorDict_from_database,
    OperatorList_from_database,
    cg_remapping,
    cg_remapping_T,
    decomposition_analysis,
    diracO_from_cgmat,
    get_op_selection,
    get_OperatorDict,
    get_OperatorList,
    get_paper_operators,
    h5_to_operator_list,
    index_symm,
    index_symm_2exchange,
    index_symm_index_fixed,
    latexO_from_diracO,
    make_operator_database,
    operator_list_to_h5,
    operator_show,
    read_operator,
    symO_from_Cgmat,
    trace_symm,
    write_operator,
)

# The version set in pyproject.toml, read from the metadata of the installed package.
try:
    __version__ = _metadata.version("h4lat")
except _metadata.PackageNotFoundError:  # a source tree that is not installed
    __version__ = "unknown"
__author__ = "Emilio Taggi"

__all__ = [
    # CG calculator
    "cg_calc",
    "get_multiplicities",
    "latex_print_multiplicities",
    "CGmat_from_block",
    "print_CGmat",
    # Lattice groups (H(4) and the 16-cell honeycomb group W(F₄))
    "get_group",
    "LatticeGroup",
    "AVAILABLE_GROUPS",
    # H(4) group constants
    "n_rep",
    "rep_dim_list",
    "rep_label_list",
    "irrep_index",
    "irrep_dim",
    "irrep_texname",
    "char_table",
    "class_orders",
    "n_ele_h4",
    "n_ele_s4",
    # Operator
    "Operator",
    "make_operator_database",
    "Operator_from_file",
    "OperatorList_from_database",
    "OperatorDict_from_database",
    "Operator_from_database",
    "operator_show",
    # Convenience getters for the bundled operator database
    "get_OperatorList",
    "get_OperatorDict",
    "get_op_selection",
    # Operators used in published works
    "get_paper_operators",
    "AVAILABLE_PAPERS",
    "cg_remapping",
    "cg_remapping_T",
    "symO_from_Cgmat",
    "diracO_from_cgmat",
    "Kfactor_from_diracO",
    "trace_symm",
    "C_parity",
    "index_symm",
    "index_symm_2exchange",
    "index_symm_index_fixed",
    "latexO_from_diracO",
    "decomposition_analysis",
    "write_operator",
    "read_operator",
    "operator_list_to_h5",
    "h5_to_operator_list",
    # bundled data paths (H(4); the folders of any group: get_group(group).operator_database_folder)
    "OPERATOR_DATABASE",
    # Kinematic / polarisation data
    "Gamma_pol",
    "Gamma_pol_list",
    "Gamma_pol_s",
    "Gamma_pol_s_list",
    "den_K",
    "den_K_list",
]
