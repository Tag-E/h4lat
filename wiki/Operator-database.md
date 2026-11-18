h4lat ships with a database of 1360 $`{H(4)}`$-irreducible operators $`{\bar\psi \, \Gamma \, D \cdots D \, \psi}`$: every
Dirac structure S ($`{\Gamma = \mathbf{1}}`$), P ($`{\gamma_5}`$), V ($`{\gamma_\mu}`$), A ($`{\gamma_\mu\gamma_5}`$) and T
($`{\sigma_{\mu\nu}}`$), with zero to three derivatives. How they are built is explained on the theory page
[Lattice operators](https://github.com/Tag-E/h4lat/wiki/Lattice-operators).

A second database holds the same operators sorted by the irreps of the group of the 16-cell honeycomb lattice (see
[The 16-cell honeycomb group](https://github.com/Tag-E/h4lat/wiki/The-16-cell-honeycomb-group)): 1360 operators
again, since every tensor product holds as many operators as its dimension, which are combined into fewer, larger
irreps. All the functions below read it with `group="honeycomb"`.

## Operator IDs

The IDs run over the number of derivatives and, for each, over S, P, V, A, T (the 16 operators without derivatives
first, then 64 with one, 256 with two, 1024 with three). Within each group they follow the order of the irreps, then
the copy of the irrep, then the column of the Clebsch-Gordan block.

| Derivatives | S | P | V | A | T |
|-------------|---|---|---|---|---|
| 0 | 1 | 2 | 3–6 | 7–10 | 11–16 |
| 1 | 17–20 | 21–24 | 25–40 | 41–56 | 57–80 |
| 2 | 81–96 | 97–112 | 113–176 | 177–240 | 241–336 |
| 3 | 337–400 | 401–464 | 465–720 | 721–976 | 977–1360 |

The ranges are the same in the honeycomb database; within each range the operators follow the irreps of the honeycomb
group, so the same ID is in general a different operator in the two databases. The IDs refer to the bundled database
of a given h4lat version and group: when citing operators by ID, give the h4lat version (`h4lat.__version__`) and the
group.

## Loading operators

The two convenience getters below load the database without any path configuration. The symbolic expression `op.O`,
the kinematic factor `op.K` and their LaTeX forms are computed when first accessed, so loading the whole database takes
about a second:

```python
from h4lat import get_OperatorList, get_OperatorDict

# --- Flat list, sorted by operator id ---
ops = get_OperatorList()

print(f"Total operators in the bundled database: {len(ops)}")

# Inspect the first operator
op = ops[0]
print(f"id={op.id}  X={op.X}  irrep={op.irrep}  block={op.block}")
print(f"Kinematic factor K = {op.latex_K}")
print(f"C-parity  = {op.C}")
print(f"Trace     = {op.tr}")
print(f"Symmetry  = {op.symm}")

# Evaluate K numerically at given kinematics
K_val = op.evaluate_K(m_value=0.939, E_value=1.0, p1_value=0.0, p2_value=0.0, p3_value=0.3)
print(f"K (numerical) = {K_val}")

# The polarisation matrix used to compute K can be changed (indices 0–5, default 0 = γ₁γ₂):
op.set_polarization_matrix(2)   # switch to Γ_pol = ½(1 + γ₄)(1 − i γ₂γ₃)

# --- A selection, by Dirac structure X and number of derivatives nder (or keys "<X><nder>") ---
local = get_OperatorList(nder=0)              # ψ̄ψ, ψ̄γ₅ψ, ψ̄γ_μψ, ψ̄γ_μγ₅ψ, ψ̄σ_μνψ (IDs 1–16)
scalars = get_OperatorList(X="S", nder=2)     # ψ̄ D_μ D_ν ψ (IDs 81–96)
some = get_OperatorList(keys="V1 A1")         # V and A with one derivative (IDs 25–56)

# --- The operators of the 16-cell honeycomb group ---
honeycomb = get_OperatorList(keys="V1", group="honeycomb")   # irreps (1,1), (6,1) and (9,1)
print(honeycomb[0].group)                                     # 'honeycomb'
```

A single operator is loaded with `Operator_from_database(id)`, or `Operator_from_database(id, group="honeycomb")`.
Every operator records its group in `op.group` (`'H4'` or `'honeycomb'`), since the label `op.irrep` refers to it.

Operators without Lorentz indices ($`{\bar\psi\psi}`$ and $`{\bar\psi\gamma_5\psi}`$) have a 0-index `op.cgmat`, of shape
`()`.

```python
# --- Nested dict, keyed by (n, X) then (irrep, block) ---
# n is the number of Lorentz indices: nder for S/P, nder + 1 for V/A, nder + 2 for T
d = get_OperatorDict()

# All 2-index vector operators in irrep (6,1), multiplicity block 1:
# (4,1)⊗(4,1) decomposes into (1,1)⊕(3,1)⊕(6,1)⊕(6,3); (6,1) is the antisymmetric piece.
ops_V2 = d[(2, 'V')][(6, 1), 1]
for op in ops_V2:
    print(f"  id={op.id}  symm={op.symm}  C={op.C}")

# All 3-index axial operators in irrep (8,1), block 1:
if (3, 'A') in d and ((8, 1), 1) in d[(3, 'A')]:
    ops_A3 = d[(3, 'A')][(8, 1), 1]
    print(f"Number of A3 operators in (8,1) block 1: {len(ops_A3)}")

# The same for the honeycomb group: the 9 vector operators with one derivative in the irrep (9,1),
# which joins the irreps (3,1) and (6,3) of H(4)
d_honeycomb = get_OperatorDict(keys="V1", group="honeycomb")
ops_v2 = d_honeycomb[(2, 'V')][(9, 1), 1]
```

Both functions are thin wrappers that call the more general `OperatorList_from_database` /
`OperatorDict_from_database` without a path, so they always read from the bundled database of the group
(`OPERATOR_DATABASE` for $`{H(4)}`$). Pass an explicit path to the underlying functions (which take the same `X`, `nder`
and `keys` selection) if you have generated a custom database with `make_operator_database`; the group of its operators
is then read from their file names (and checked against `group`, if you give it).

## Building an operator database

```python
from h4lat import make_operator_database

make_operator_database(
    operator_folder="my_operators",
    max_nder=3,     # S, P, V, A and T with 0 to 3 derivatives (as the bundled database)
    verbose=True,
)

# the same for the honeycomb group (the bundled honeycomb database)
make_operator_database(operator_folder="my_honeycomb_operators", max_nder=3, group="honeycomb")
```

The operators are numbered as in the bundled database (by number of derivatives, then S, P, V, A, T). The folder must
not already hold operator files. Products beyond the bundled ones (four or more derivatives) are computed by
`cg_calc`, which for four derivatives takes a lot of memory.

Each operator is stored in its own file, `operator_{id}_{n}_{X}_{k}_{l}_{block}_{index_block}.npy`, which holds its
`cgmat`; all the other information is in the file name. `n` is the number of Lorentz indices, not the number of
derivatives. The file names of the operators of the honeycomb group end in `_honeycomb` (before `.npy`), and a name
without a group is an $`{H(4)}`$ operator, as in the databases written before h4lat knew other groups.
`Operator_from_file` reads one file, and `op.save(folder)` writes one.

## Operator IDs before October 2026

Until October 2026 the bundled database held only the V, A and T operators with one to three derivatives (IDs
1–1176). Adding the S and P operators and the operators without derivatives shifted every ID; the operators themselves
are unchanged. An old ID of an operator with one, two or three derivatives maps to a new one as follows:

| Old IDs | Derivatives | New ID |
|---------|-------------|--------|
| 1–56 | 1 | old + 24 |
| 57–280 | 2 | old + 56 |
| 281–1176 | 3 | old + 184 |

Also in October 2026, the C-parity of the P operators was corrected (see
[Lattice operators](https://github.com/Tag-E/h4lat/wiki/Lattice-operators#c-parity)), and the copies of the irreps
with multiplicity greater than 1 started to be chosen with a definite C-parity, so the operator IDs of those irreps
refer to other copies than before (see
[Clebsch-Gordan coefficients](https://github.com/Tag-E/h4lat/wiki/Clebsch-Gordan-coefficients#fixing-the-free-parameters)).
The [changelog](https://github.com/Tag-E/h4lat/blob/main/CHANGELOG.md) lists all the changes of each release.
