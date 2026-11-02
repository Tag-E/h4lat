*Theory, part 3 of 3.* Previous: [The hypercubic group](https://github.com/Tag-E/h4lat/wiki/The-hypercubic-group) ·
[Clebsch-Gordan coefficients](https://github.com/Tag-E/h4lat/wiki/Clebsch-Gordan-coefficients)

## Operators and their irreps

We can now turn to the problem of building all the operators that are relevant for physics applications, in
particular for the moments of parton distribution functions. A generic operator of interest has the form

```math
\mathcal{O}^{X}_{\nu \mu_1 \cdots \mu_n} = \bar\psi \, \Gamma^X_\nu \, \overleftrightarrow{D}_{\mu_1} \cdots
\overleftrightarrow{D}_{\mu_n} \, \psi , \tag{1}
```

where $`{\overleftrightarrow{D} = \overrightarrow{D} - \overleftarrow{D}}`$ is the covariant derivative acting to the
right minus the one acting to the left (a factor $`{\tfrac{1}{2}}`$ is also common: constant factors play no role here),
and $`{n}`$ is the number of derivatives. The Dirac structure $`{\Gamma^X}`$ is one of those in the table below, and
$`{\nu}`$ stands for its Lorentz indices: none for S and P, one for V and A, and the two indices of
$`{\sigma_{\nu_1 \nu_2}}`$ for T.

| X | Dirac structure $`{\Gamma^X}`$ | Irrep of $`{\Gamma^X}`$ | Code label | C-parity $`{\eta_X}`$ of $`{\bar\psi \Gamma^X \psi}`$ |
|---|------------------------------|-----------------------|------------|-------------------------------------------------|
| S | $`{\mathbf{1}}`$ | $`{\tau^{(1)}_1}`$ | `(1, 1)` | $`{+1}`$ |
| P | $`{\gamma_5}`$ | $`{\tau^{(1)}_4}`$ | `(1, 4)` | $`{+1}`$ |
| V | $`{\gamma_\nu}`$ | $`{\tau^{(4)}_1}`$ | `(4, 1)` | $`{-1}`$ |
| A | $`{\gamma_\nu \gamma_5}`$ | $`{\tau^{(4)}_4}`$ | `(4, 4)` | $`{+1}`$ |
| T | $`{\sigma_{\nu_1 \nu_2} = \tfrac{i}{2} [\gamma_{\nu_1}, \gamma_{\nu_2}]}`$ | $`{\tau^{(6)}_1}`$ | `(6, 1)` | $`{-1}`$ |

To each covariant derivative $`{\overleftrightarrow{D}_{\mu_i}}`$ we associate the irrep $`{\tau^{(4)}_1}`$. The operators
of Eq. (1) therefore span the tensor product of the irrep of $`{\Gamma^X}`$ with $`{n}`$ copies of $`{\tau^{(4)}_1}`$, and
the [Clebsch-Gordan coefficients](https://github.com/Tag-E/h4lat/wiki/Clebsch-Gordan-coefficients) of this product,
$`{c_{\nu \mu_1 \cdots \mu_n}}`$, give the operators that transform according to definite irreps:

```math
\mathcal{O} = \sum_{\nu, \mu_1, \dots, \mu_n} c_{\nu \mu_1 \cdots \mu_n} \, \mathcal{O}^{X}_{\nu \mu_1 \cdots \mu_n} ,
\tag{2}
```

one operator for each column of a Clebsch-Gordan block. For example, the vector operators with one derivative,
$`{\mathcal{O}_{\mu_1\mu_2} = \bar\psi \gamma_{\mu_1} \overleftrightarrow{D}_{\mu_2} \psi}`$, are those of the
[explicit example](https://github.com/Tag-E/h4lat/wiki/Clebsch-Gordan-coefficients#an-explicit-example) of the
previous page.

The complete catalogue of these operators, for S, P, V, A and T with up to three derivatives (1360 operators), is
bundled with the library. It can be accessed with `get_OperatorList`, `get_OperatorDict` and `Operator_from_database`
(see [Operator database](https://github.com/Tag-E/h4lat/wiki/Operator-database)), or browsed as a PDF (see
[Operator catalogue](https://github.com/Tag-E/h4lat/wiki/Operator-catalogue)). Operators with more derivatives are not
precomputed, but they can be computed with the techniques outlined above, with `make_operator_database`.

For every operator, h4lat also gives its index symmetry (`op.symm`), whether its traces vanish (`op.tr`), its
C-parity (`op.C`) and its kinematic factor (`op.K`), described below.

## Kinematic factor

The kinematic factor $`{K}`$ of an operator is the tree-level factor that relates the nucleon matrix element of the
operator to the corresponding moment. To compute it, h4lat first builds the Dirac matrix obtained by replacing every
derivative with a component of the nucleon momentum,

```math
M(p) = \sum_{\nu, \mu_1, \dots, \mu_n} c_{\nu \mu_1 \cdots \mu_n} \, \Gamma^X_\nu \, p_{\mu_1} \cdots p_{\mu_n} , \tag{3}
```

and then

```math
K(p) = \frac{1}{2 E(p)} \,
\frac{\mathrm{Tr}\left[ \Gamma_{\text{pol}} \left( -i \gamma_\rho p_\rho + m_N \right) M(p)
\left( -i \gamma_\rho p_\rho + m_N \right) \right]}
{\mathrm{Tr}\left[ \Gamma_{\text{pol}} \left( -i \gamma_\rho p_\rho + m_N \right) \right]} , \tag{4}
```

where:

- the $`{c_{\nu \mu_1 \cdots \mu_n}}`$ are the Clebsch-Gordan coefficients of the operator, Eq. (2);
- the $`{\gamma_\mu}`$ are the Euclidean gamma matrices, Hermitian, with $`{\{\gamma_\mu, \gamma_\nu\} = 2\delta_{\mu\nu}}`$,
  and the repeated index $`{\rho}`$ is summed over;
- $`{p_\mu = (p_1, p_2, p_3, iE(p))}`$ is the Euclidean four-momentum of the nucleon, with
  $`{E(p) = \sqrt{m_N^2 + p_1^2 + p_2^2 + p_3^2}}`$, so that $`{p_\rho p_\rho = -m_N^2}`$;
- $`{\Gamma_{\text{pol}} = \tfrac{1}{2}(1 + \gamma_4)(1 - i \gamma_a \gamma_b)}`$ is the polarisation projector:
  $`{\tfrac{1}{2}(1 + \gamma_4)}`$ projects onto positive energies, and the pair $`{(a, b)}`$ selects the spin
  polarisation. The default is $`{(a, b) = (1, 2)}`$; the six choices are listed in `Gamma_pol_list`, and
  `op.set_polarization_matrix(k)` switches an operator to choice `k`.

With this normalisation $`{K = 1}`$ for the vector charge $`{\bar\psi \gamma_4 \psi}`$ (operator 6 of the database), and
$`{K = m_N / E(p)}`$ for $`{\bar\psi \psi}`$. Each derivative contributes only the momentum component $`{p_\mu}`$ to $`{M(p)}`$:
constant factors per derivative, such as the $`{i}`$ in $`{\partial_\mu e^{ipx} = i p_\mu e^{ipx}}`$, are not included in
$`{K}`$ (`op.evaluate_K_real` adds the factor $`{i^n}`$). $`{K}`$ is computed in exact arithmetic, as a SymPy expression in
$`{m_N}`$, $`{E(p)}`$ and $`{p_1, p_2, p_3}`$.

## C-parity

Under charge conjugation $`{\mathcal{C}}`$, the operators of Eq. (1) transform as

```math
\mathcal{C} \, \mathcal{O}^{X}_{\nu \mu_1 \cdots \mu_n} \, \mathcal{C}^{-1}
= \eta_X \, (-1)^n \, \mathcal{O}^{X}_{\nu \mu_n \cdots \mu_1} ,
```

i.e. charge conjugation reverses the order of the derivative indices and leaves those of $`{\Gamma^X}`$ in place. The
sign $`{\eta_X}`$ is the C-parity of the local operator $`{\bar\psi \Gamma^X \psi}`$ (see the table above), and each
derivative adds a factor $`{-1}`$. Since h4lat chooses the copies of every irrep even or odd under the reversal of the
derivative indices (see [Clebsch-Gordan coefficients](https://github.com/Tag-E/h4lat/wiki/Clebsch-Gordan-coefficients#fixing-the-free-parameters)),
every operator of the bundled database has a definite C-parity, $`{C = \pm 1}`$ (`op.C`). For more context on C-parity
we refer the reader to Göckeler et al. (1996).

In October 2026 the C-parity of the P operators was corrected: the pseudoscalar density $`{\bar\psi \gamma_5 \psi}`$ is
C-even (as the $`{\pi^0}`$), so `C_parity` now gives $`{\eta_X (-1)^n}`$ with $`{\eta_P = +1}`$. P operators had not been
bundled before, so no bundled operator changed.

## Mixing under renormalisation

The classification by irrep and C-parity is important because of renormalisation. On the lattice, an operator can mix
under renormalisation only with operators of equal or lower dimension (the dimension of
$`{\bar\psi \, \Gamma \, D^n \psi}`$ is $`{3 + n}`$) that transform according to the same irrep and have the same
C-parity. If the pair (irrep, C-parity) of an operator appears only once among the operators
$`{\bar\psi \, \Gamma \, D \cdots D \, \psi}`$ of equal and lower dimension, the operator cannot mix with any of them,
and its renormalisation is multiplicative. (Flavour-singlet operators can, in addition, mix with gluon operators.)
Mixing with operators of lower dimension is particularly harmful, since its coefficients diverge as inverse powers of
the lattice spacing.

The operators relevant for the moments of the unpolarised, helicity and transversity PDFs are those with
$`{\Gamma = \gamma_\mu, \gamma_\mu \gamma_5, \sigma_{\mu\nu}}`$ (V, A and T). With the complete catalogue provided, it is
easy to assess which irreps are free of mixing and to obtain explicit expressions for their operators:

- `decomposition_analysis(X, n_der)` lists the irreps of the product for the Dirac structure `X` with `n_der`
  derivatives, with their multiplicities, and marks an irrep as safe from mixing when it does not appear among the
  operators with fewer derivatives, for any of S, P, V, A and T. It does not take the C-parity into account, nor the
  operators of the same dimension.
- `get_op_selection(n_der)` returns a curated selection of V, A and T operators free of mixing, for one to three
  derivatives.
- `get_paper_operators` returns the operators used in published works (see
  [Operators from the literature](https://github.com/Tag-E/h4lat/wiki/Operators-from-the-literature)).

## References

- M. Göckeler, R. Horsley, E.-M. Ilgenfritz, H. Perlt, P. E. L. Rakow, G. Schierholz and A. Schiller, *Lattice
  operators for moments of the structure functions and their transformation under the hypercubic group*,
  Phys. Rev. D **54**, 5705–5714 (1996), [doi:10.1103/PhysRevD.54.5705](https://doi.org/10.1103/PhysRevD.54.5705),
  [arXiv:hep-lat/9602029](https://arxiv.org/abs/hep-lat/9602029).

All the references of the wiki are collected on the [References](https://github.com/Tag-E/h4lat/wiki/References)
page.
