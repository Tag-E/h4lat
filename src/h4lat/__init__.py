"""
h4lat — Clebsch-Gordan coefficients and lattice operators for the hypercubic group H(4).

Quick start
-----------
>>> from h4lat import cg_calc, get_multiplicities
>>> muls = get_multiplicities((4, 1), (4, 1))   # returns a 20-element list
>>> cg = cg_calc((4, 1), (4, 1))                # loads bundled CG database
>>> cg.cg_dict                                  # {irrep_index: [cgmat, ...], ...}
"""

from .cg_calculator import (
    _BUNDLED_OPERATOR_DATABASE as OPERATOR_DATABASE,
)
from .cg_calculator import (
    CGmat_from_block,
    cg_calc,
    char_table,
    class_orders,
    force_symmetry_alpha,
    force_symmetry_beta,
    force_symmetry_gamma,
    get_multiplicities,
    irrep_dim,
    irrep_index,
    irrep_texname,
    latex_print_multiplicities,
    n_ele_h4,
    n_ele_s4,
    # constants
    n_rep,
    print_CGmat,
    rep_dim_list,
    rep_label_list,
)
from .moments_operator import (
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
    index_symm,
    index_symm_2exchange,
    index_symm_index_fixed,
    latexO_from_diracO,
    make_operator_database,
    read_operator,
    symO_from_Cgmat,
    trace_symm,
    write_operator,
)

__version__ = "0.1.0"
__author__ = "Emilio Taggi"

__all__ = [
    # CG calculator
    "cg_calc",
    "get_multiplicities",
    "latex_print_multiplicities",
    "CGmat_from_block",
    "force_symmetry_gamma",
    "force_symmetry_alpha",
    "force_symmetry_beta",
    "print_CGmat",
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
    # bundled data paths
    "OPERATOR_DATABASE",
]
