This page shows how to decompose tensor products of $`{H(4)}`$ irreps and how to get their Clebsch-Gordan (CG)
coefficients, from the bundled database or by computing them. The method is explained on the theory pages
[The hypercubic group](https://github.com/Tag-E/h4lat/wiki/The-hypercubic-group) and
[Clebsch-Gordan coefficients](https://github.com/Tag-E/h4lat/wiki/Clebsch-Gordan-coefficients). Irreps are given by
their labels `(k, l)`, the irrep $`{\tau^{(k)}_l}`$ of dimension `k`.

## Tensor-product decomposition

```python
from h4lat import get_multiplicities, latex_print_multiplicities, rep_label_list

# Decompose (4,1) ⊗ (4,1)
muls = get_multiplicities((4, 1), (4, 1))
for i, m in enumerate(muls):
    if m > 0:
        print(f"  {rep_label_list[i]}  ×{m}")

# Get a LaTeX string
print(latex_print_multiplicities((4, 1), (4, 1)))
```

`get_multiplicities` takes any number of irreps and returns the 20 multiplicities, in the order of `rep_label_list`.

## Loading CG coefficients

```python
from h4lat import cg_calc

# Load from the bundled database (no computation)
cg = cg_calc((4, 1), (4, 1))

# cg.cg_dict   : dict[irrep_index -> list[ndarray]]
# cg.mul_list  : list of multiplicities (length 20)

for irep_idx, matrices in cg.cg_dict.items():
    from h4lat import rep_label_list
    print(f"Irrep {rep_label_list[irep_idx]}:  {len(matrices)} matrix/matrices, "
          f"shape {matrices[0].shape}")
```

`cg.cg_dict[irep][m]` is the CG block of copy `m` of the irrep `rep_label_list[irep]`: one row per basis vector of the
product, one column per basis vector of the irrep. Its normalisation is explained in
[Clebsch-Gordan coefficients](https://github.com/Tag-E/h4lat/wiki/Clebsch-Gordan-coefficients#fixing-the-free-parameters).
Pass `verbose=False` to `cg_calc` to silence its messages.

The bundled database holds the products of X = (1,1), (1,4), (4,1), (4,4) or (6,1) (scalar, pseudoscalar, vector,
axial, tensor) with zero, one, two or three (4,1) factors (the derivatives). With no derivatives the "product" is the
irrep of X alone, e.g. `cg_calc((6, 1))`, whose CG matrix is the identity. The raw symbolic blocks from which the CG
matrices are derived (`cg.raw_cg`, read on first access, needed by `prescription_changed=True`) are bundled only for
products of up to three irreps (up to two derivatives): for the products of four irreps they would take tens of MB,
so recompute them with `force_computation=True` (about a minute) if you need them.

## Computing new CG coefficients

```python
# Compute the (6,1) ⊗ (4,1) ⊗ (4,1) tensor product (tensor operator, with two derivatives)
cg = cg_calc((6, 1), (4, 1), (4, 1))

# Products of four irreps take about a minute; parallel=True sums over the group on all cores
cg = cg_calc((6, 1), (4, 1), (4, 1), (4, 1), parallel=True)

# ... or on 8 threads only (giving n_threads implies parallel=True)
cg = cg_calc((6, 1), (4, 1), (4, 1), (4, 1), n_threads=8)
```

These products are bundled, so the calls above load them; to compute a bundled product anyway, pass
`force_computation=True`. Any other product is computed on the first call.

The projection formula sums over the 384 group elements numerically, on the coefficients of the free symbols, so a
product of four irreps such as (6,1) ⊗ (4,1)³ takes about a minute (it took days while the sum was done
symbolically, until October 2026; the result is the same to the last bit). With `parallel=True` the sum runs on all
CPU cores with [numba](https://numba.pydata.org) (`pip install "h4lat[parallel]"`); use fewer threads with
`n_threads=` (between 1 and the number of cores, or the environment variable `NUMBA_NUM_THREADS` if set). The result
does not depend on `parallel` or on the number of threads. By default (`parallel=None`) products of three or more
irreps are computed in parallel when numba is installed. Most of the remaining time goes into the symbolic raw blocks
saved in `{product}_raw/`, which are built serially.

## Where the results are stored

New results are written to `./cg_database` in the working directory (H(4) matrices regenerated with
`force_h4gen=True` go to `./h4_ele`), never into the installed package. Later calls read the bundled data first, then
`./cg_database`. Pass `cgdatabase="/path/to/custom/db"` to read and write a custom database instead.

To update the bundled data, copy these folders into `src/h4lat/data/`, or regenerate them with
`python scripts/regenerate_data.py` (see `--help`), then refresh the PDF catalogue with
[`h4lat-catalogue`](https://github.com/Tag-E/h4lat/wiki/Operator-catalogue). The rules for changing the bundled data
are in [CONTRIBUTING.md](https://github.com/Tag-E/h4lat/blob/main/CONTRIBUTING.md).

## From CG blocks to operators

A column of a CG block, reshaped into a tensor with one index per factor, gives the coefficients of an operator:
`cg_remapping(column, n)` reshapes a column of length $`{4^n}`$ into shape `(4,)*n`, and `cg_remapping_T` does the same
for the tensor structure, whose first factor (6,1) is the antisymmetric pair of indices of $`{\sigma_{\mu\nu}}`$. See
[Building operators](https://github.com/Tag-E/h4lat/wiki/Building-operators).
