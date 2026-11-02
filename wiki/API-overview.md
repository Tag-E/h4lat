All the names below are imported by `import h4lat` (they are listed in `h4lat.__all__`). Every function and class has
a detailed docstring: use `help(h4lat.cg_calc)`, or `h4lat.cg_calc?` in Jupyter.

## Modules

| Module | Contents |
|--------|----------|
| `h4lat.cg_calculator` | `cg_calc`, `get_multiplicities`, the $`{H(4)}`$ group constants and generator matrices |
| `h4lat.moments_operator` | `Operator`, the operator database, kinematic factors, C-parity and symmetry analysis, HDF5 I/O, the PDF catalogue |
| `h4lat.kinematic_data` | Euclidean gamma matrices, symbolic momenta, polarisation projectors |
| `h4lat.utilities` | Permutation parity, perfect-square test, all-equal check |
| `h4lat.cli` | `catalogue_main`, the entry point of the `h4lat-catalogue` command |

## Group theory and Clebsch-Gordan coefficients

| Name | Description |
|------|-------------|
| `cg_calc` | Clebsch-Gordan coefficients of a tensor product of irreps, loaded or computed ([guide](https://github.com/Tag-E/h4lat/wiki/Computing-Clebsch-Gordan-coefficients)) |
| `get_multiplicities` | Multiplicity of each irrep in a tensor product |
| `latex_print_multiplicities` | The decomposition of a tensor product as a LaTeX string |
| `CGmat_from_block` | Fixes the free parameters of a symbolic Clebsch-Gordan block ([theory](https://github.com/Tag-E/h4lat/wiki/Clebsch-Gordan-coefficients#fixing-the-free-parameters)) |
| `print_CGmat` | Pretty-prints a Clebsch-Gordan matrix in Jupyter |
| `rep_label_list`, `rep_dim_list`, `n_rep` | The 20 irrep labels `(k, l)`, their dimensions, and their number |
| `irrep_index`, `irrep_dim`, `irrep_texname` | Position in `rep_label_list`, dimension and LaTeX name of an irrep label |
| `char_table`, `class_orders` | The character table (irreps × classes) and the number of elements of each class |
| `n_ele_h4`, `n_ele_s4` | The orders of $`{H(4)}`$ (384) and $`{S_4}`$ (24) |

## Operators

| Name | Description |
|------|-------------|
| `Operator` | One irreducible lattice operator ([guide](https://github.com/Tag-E/h4lat/wiki/Building-operators)) |
| `get_OperatorList`, `get_OperatorDict` | The bundled operator database as a list or a nested dict ([guide](https://github.com/Tag-E/h4lat/wiki/Operator-database)) |
| `Operator_from_database` | One operator of a database, by ID |
| `OperatorList_from_database`, `OperatorDict_from_database` | The same as the getters, for any database folder |
| `Operator_from_file` | One operator from its `.npy` file |
| `make_operator_database` | Builds an operator database with up to a given number of derivatives |
| `OPERATOR_DATABASE` | The path of the bundled operator database |
| `operator_show` | Writes a PDF catalogue of the operators ([guide](https://github.com/Tag-E/h4lat/wiki/Operator-catalogue)) |
| `get_paper_operators`, `AVAILABLE_PAPERS` | Operators used in published works ([guide](https://github.com/Tag-E/h4lat/wiki/Operators-from-the-literature)) |
| `get_op_selection` | A curated selection of operators free of mixing, for one to three derivatives |
| `decomposition_analysis` | Irrep content of an operator product, with multiplicities and mixing with fewer derivatives |
| `operator_list_to_h5`, `h5_to_operator_list` | Save and load a list of operators in an HDF5 file ([guide](https://github.com/Tag-E/h4lat/wiki/Saving-and-loading-operators)) |
| `write_operator`, `read_operator` | Save and load one operator in an HDF5 group |

## Building blocks of an operator

| Name | Description |
|------|-------------|
| `cg_remapping`, `cg_remapping_T` | Reshape a Clebsch-Gordan column into the `cgmat` tensor of an operator (`_T` for the tensor structure) |
| `symO_from_Cgmat` | Symbolic expression of an operator |
| `diracO_from_cgmat` | Dirac matrix $`{M(p)}`$ of an operator ([theory](https://github.com/Tag-E/h4lat/wiki/Lattice-operators#kinematic-factor)) |
| `Kfactor_from_diracO` | Kinematic factor from the Dirac matrix |
| `latexO_from_diracO` | LaTeX string of an operator |
| `C_parity` | C-parity of an operator |
| `trace_symm` | Whether the traces of an operator vanish |
| `index_symm`, `index_symm_2exchange`, `index_symm_index_fixed` | Symmetry under permutations of all, two, or the non-fixed indices |
| `Gamma_pol_list`, `Gamma_pol_s_list`, `den_K_list` | The six polarisation projectors, their symbols, and the denominators of K |
| `Gamma_pol`, `Gamma_pol_s`, `den_K` | The same for the default projector (index 0) |

## Command line

| Command | Description |
|---------|-------------|
| `h4lat-catalogue` | Writes a PDF catalogue of the operators; see `h4lat-catalogue --help` and [Operator catalogue](https://github.com/Tag-E/h4lat/wiki/Operator-catalogue) |
| `python scripts/regenerate_data.py` | Regenerates the bundled data (in a clone of the repository; see [CONTRIBUTING.md](https://github.com/Tag-E/h4lat/blob/main/CONTRIBUTING.md)) |
