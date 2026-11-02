## HDF5 files

`operator_list_to_h5` / `h5_to_operator_list` save an arbitrary `list[Operator]` to a single portable `.h5` file and
reload it later, preserving every attribute needed to reconstruct each operator exactly, including
`gamma_pol_index`, so a custom polarisation matrix set via `set_polarization_matrix()` survives the round trip. This is
a lighter-weight alternative to `make_operator_database` for saving an ad-hoc selection of operators (e.g. the output
of `get_paper_operators`, or operators built by hand) rather than a full database:

```python
from h4lat import get_paper_operators, operator_list_to_h5, h5_to_operator_list

ops = get_paper_operators("Rodekamp:2023wpe")  # any list of Operator objects
operator_list_to_h5(ops, "my_operators/selection.h5")

# ... later, possibly in a different session ...
reloaded = h5_to_operator_list("my_operators/selection.h5")
assert [op.id for op in reloaded] == [op.id for op in ops]
```

Requires `h5py` (`pip install "h4lat[full]"` or `"h4lat[operators]"`). Under the hood, each operator is stored in its
own group and (de)serialised with `write_operator` / `read_operator`, which can be used directly if you need to embed a
single operator inside a larger, custom HDF5 layout of your own. Fields that are `None` (e.g. the `id` of a sum of
operators) are stored too.

## One file per operator

`op.save(folder)` writes the `cgmat` of an operator to a `.npy` file whose name holds all the other information, in
the format of the [operator database](https://github.com/Tag-E/h4lat/wiki/Operator-database#building-an-operator-database),
and `Operator_from_file(path)` reads it back. Since the file name cannot store `None`, `save` refuses operators without
`id`, `irrep`, `block` or `index_block`, such as the sums built with `+` or `-`: save those with
`operator_list_to_h5`.
