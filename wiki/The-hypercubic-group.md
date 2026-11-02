*Theory, part 1 of 3.* Next: [Clebsch-Gordan coefficients](https://github.com/Tag-E/h4lat/wiki/Clebsch-Gordan-coefficients) ·
[Lattice operators](https://github.com/Tag-E/h4lat/wiki/Lattice-operators)

This page introduces the hypercubic group $`H(4)`$, its irreducible representations and how they combine in tensor
products. Throughout the wiki, the irrep written $`\tau^{(k)}_l`$ in the literature is the label `(k, l)` in the code:
for example, $`\tau^{(4)}_1`$ is `(4, 1)`.

## Elements of H(4)

In four-dimensional Euclidean space, the group of rotations and reflections is the continuous group $`O(4)`$. On a
hypercubic lattice only the elements of $`O(4)`$ that map the lattice onto itself survive. They form a finite
subgroup, the hypercubic group $`H(4)`$.

Each of these elements maps every axis onto plus or minus another axis: it is a *signed permutation* of the four
axes, i.e. a permutation of the axis labels combined with a change of sign of some of the axes. The permutations
form the symmetric group $`S_4`$, of order $`4! = 24`$. The sign changes form the group $`(\mathbb{Z}_2)^4`$, of order
$`2^4 = 16`$, where $`\mathbb{Z}_2`$ is the cyclic group of order two. Conjugating a sign change with a permutation
gives another sign change, so the sign changes form a normal subgroup and $`H(4)`$ is the semidirect product

```math
H(4) = (\mathbb{Z}_2)^4 \rtimes S_4 , \qquad |H(4)| = 2^4 \cdot 4! = 384 .
```

Not all the elements are rotations: the swap of two axes and the reflection of one axis have determinant $`-1`$. The
rotations are the 192 elements with determinant $`+1`$.

## Generators

All the elements of $`H(4)`$ can be generated from three of them, $`\alpha`$, $`\beta`$ and $`\gamma`$. We adopt the
notation of [Baake et al. (1982)](https://doi.org/10.1063/1.525461) and
[Göckeler et al. (1996)](https://doi.org/10.1103/PhysRevD.54.5705):

- $`\alpha`$ swaps the first two axes: it is the transposition $`(12)`$;
- $`\beta`$ permutes the four axes cyclically, $`1 \to 2 \to 3 \to 4 \to 1`$: it is the 4-cycle $`(1234)`$;
- $`\gamma`$ reflects the first axis, $`x_1 \to -x_1`$.

The first two generate the permutations $`S_4`$ and satisfy its defining relations
$`\alpha^2 = \beta^4 = (\alpha\beta)^3 = 1`$ (the product $`\alpha\beta`$ is a 3-cycle). The last one, $`\gamma`$,
generates the $`\mathbb{Z}_2`$ structure: combined with the permutations, it gives the reflection of every axis. In
the vector representation $`\tau^{(4)}_1`$, where the group acts on the four axes, the generators are the matrices

```math
\alpha = \begin{pmatrix} 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}, \qquad
\beta = \begin{pmatrix} 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{pmatrix}, \qquad
\gamma = \begin{pmatrix} -1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix} .
```

For each of the 20 irreps of $`H(4)`$ (see [below](https://github.com/Tag-E/h4lat/wiki/The-hypercubic-group#irreducible-representations)),
Baake et al. (1982) give explicit matrices for the three generators.

> **Note on the irrep $`\tau^{(2)}_2`$.** In Table IXB of Baake et al. (1982), the matrix given for $`\alpha`$ in the
> irrep $`\tau^{(2)}_2`$ is the identity. This cannot be right: the matrices of every representation must satisfy
> $`(\alpha\beta)^3 = 1`$, because $`\alpha\beta`$ is a 3-cycle of $`S_4`$, while with $`\alpha = 1`$ one gets
> $`(\alpha\beta)^3 = \beta^3 \neq 1`$. Moreover $`\tau^{(2)}_2 = \tau^{(2)}_1 \otimes \tau^{(1)}_2`$, and
> $`\tau^{(1)}_2`$ is trivial on the permutations, so $`\alpha`$ must be the same as in $`\tau^{(2)}_1`$:
> h4lat uses $`\alpha = \mathrm{diag}(1, -1)`$. (Until October 2026, h4lat used the identity, as in the reference.)

## The 384 matrices

From these $`20 \cdot 3`$ generator matrices, h4lat builds the explicit matrices of all the elements in every irrep,
$`20 \cdot 384`$ matrices in total. Starting from the identity, $`\alpha`$ and $`\beta`$ are applied iteratively until
the $`4! = 24`$ distinct elements of $`S_4`$ have been generated (new elements are recognised by their matrix in
$`\tau^{(4)}_1`$, and the same products are formed in every irrep). Looping over these 24 elements $`g`$, we obtain all
the remaining ones as the products

```math
g \left(\gamma^{(1)}\right)^{i_1} \left(\gamma^{(2)}\right)^{i_2} \left(\gamma^{(3)}\right)^{i_3}
\left(\gamma^{(4)}\right)^{i_4}, \qquad i_1, i_2, i_3, i_4 \in \{0, 1\}, \tag{1}
```

where $`\gamma^{(j)}`$ reflects the $`j`$-th axis. It is obtained from $`\gamma`$ by conjugation with $`\beta`$:

```math
\gamma^{(1)} \equiv \gamma, \qquad \gamma^{(j)} \equiv \beta^{\,j-1} \, \gamma \, \beta^{-(j-1)} \quad (j = 2, 3, 4):
```

$`\beta^{-(j-1)}`$ maps the $`j`$-th axis onto the first one, $`\gamma`$ reflects it, and $`\beta^{j-1}`$ maps it back.
In this way, $`24 \cdot 2^4 = 384`$ elements are generated for each irrep.

**In the code.** The generator matrices are the lists `alpha_list`, `beta_list` and `gamma_list` of the module
`h4lat.cg_calculator`, in the order of `rep_label_list`. The matrices of all the elements are shipped with the
package (folder `src/h4lat/data/h4_ele/`) and are held by the attribute `h4_mat` of any `cg_calc` object:
`h4_mat[ir][e]` is the matrix of element `e` in the irrep `rep_label_list[ir]`, where
`e = 16*s + 8*i1 + 4*i2 + 2*i3 + i4` for the `s`-th element of $`S_4`$ and the exponents of Eq. (1).

```python
from h4lat import cg_calc, irrep_index
from h4lat.cg_calculator import alpha_list, beta_list, gamma_list

h4_mat = cg_calc((4, 1), (4, 1), verbose=False).h4_mat  # a bundled product: nothing is computed
vector = h4_mat[irrep_index[(4, 1)]]                    # the 384 matrices of the vector irrep (4,1)
print(vector[8])                                        # γ⁽¹⁾ = diag(-1, 1, 1, 1)
print(alpha_list[irrep_index[(2, 2)]])                  # α in (2,2): diag(1, -1)
```

## Irreducible representations

$`H(4)`$ has 20 conjugacy classes, and therefore 20 irreducible representations (irreps). Writing $`\tau^{(k)}_l`$ for
the $`l`$-th irrep of dimension $`k`$, they are

```math
\tau^{(1)}_1,\ \tau^{(1)}_2,\ \tau^{(1)}_3,\ \tau^{(1)}_4,\quad
\tau^{(2)}_1,\ \tau^{(2)}_2,\quad
\tau^{(3)}_1,\ \tau^{(3)}_2,\ \tau^{(3)}_3,\ \tau^{(3)}_4,\quad
\tau^{(4)}_1,\ \tau^{(4)}_2,\ \tau^{(4)}_3,\ \tau^{(4)}_4,\quad
\tau^{(6)}_1,\ \tau^{(6)}_2,\ \tau^{(6)}_3,\ \tau^{(6)}_4,\quad
\tau^{(8)}_1,\ \tau^{(8)}_2 .
```

As for every finite group, the squares of the dimensions add up to the order of the group:
$`4 \cdot 1^2 + 2 \cdot 2^2 + 4 \cdot 3^2 + 4 \cdot 4^2 + 4 \cdot 6^2 + 2 \cdot 8^2 = 384`$. Some of them have a direct
physical meaning:

| Irrep | Code label | Dim. | What transforms in it |
|-------|------------|------|-----------------------|
| $`\tau^{(1)}_1`$ | `(1, 1)` | 1 | scalars (the trivial irrep), e.g. $`\bar\psi\psi`$ |
| $`\tau^{(1)}_4`$ | `(1, 4)` | 1 | pseudoscalars (the determinant of the signed permutation), e.g. $`\bar\psi\gamma_5\psi`$ |
| $`\tau^{(3)}_1`$ | `(3, 1)` | 3 | traceless diagonal symmetric tensors, e.g. $`T_{11} - T_{22}`$ |
| $`\tau^{(4)}_1`$ | `(4, 1)` | 4 | vectors (the action on the four axes), e.g. $`\gamma_\mu`$ and the derivative $`D_\mu`$ |
| $`\tau^{(4)}_4`$ | `(4, 4)` | 4 | axial vectors, $`\tau^{(4)}_1 \otimes \tau^{(1)}_4`$, e.g. $`\gamma_\mu\gamma_5`$ |
| $`\tau^{(6)}_1`$ | `(6, 1)` | 6 | antisymmetric tensors, e.g. $`\sigma_{\mu\nu}`$ |
| $`\tau^{(6)}_3`$ | `(6, 3)` | 6 | off-diagonal symmetric tensors, e.g. $`T_{\mu\nu} + T_{\nu\mu}`$ with $`\mu \neq \nu`$ |

In the code, `rep_label_list` lists the 20 labels in this order, `rep_dim_list` their dimensions, and the
dictionaries `irrep_index`, `irrep_dim` and `irrep_texname` map a label to its position, its dimension and its LaTeX
name.

## Tensor products and the Clebsch-Gordan series

The tensor product of two or more irreps of $`H(4)`$ is a representation of $`H(4)`$, but in general it is not
irreducible (a few products are, e.g. those with a one-dimensional irrep, or $`\tau^{(2)}_1 \otimes \tau^{(4)}_1`$,
which is an 8-dimensional irrep). It can be decomposed into a direct sum of irreps. This decomposition, called the
**Clebsch-Gordan series**, has the form

```math
D^{(\mu)} \otimes D^{(\nu)} = \bigoplus_\sigma a_\sigma \, D^{(\sigma)}, \tag{2}
```

where the symbol $`D^{(\mu)}`$ denotes a generic irrep $`\tau^{(k)}_l`$. The coefficient $`a_\sigma`$ is the number of
times the irrep $`D^{(\sigma)}`$ appears in the decomposition of the tensor product, i.e. the **multiplicity** of
$`D^{(\sigma)}`$.

For a given tensor product, the multiplicities can be computed from the characters of the irreps,
$`\chi^{(\sigma)}(g) = \operatorname{Tr} D^{(\sigma)}(g)`$, which take the same value on all the elements of a
conjugacy class:

```math
a_\sigma = \left\langle \chi^{(\sigma)}, \chi^{(\mu)} \chi^{(\nu)} \right\rangle
= \frac{1}{384} \sum_{i=1}^{20} k_i \, \chi^{(\sigma)*}_i \, \chi^{(\mu)}_i \, \chi^{(\nu)}_i , \tag{3}
```

where $`384`$ is the number of elements of the group, the index $`i`$ runs over the 20 conjugacy classes of $`H(4)`$,
$`k_i`$ is the number of elements in class $`i`$, and $`\chi^{(\sigma)}_i`$ is the character of the irrep $`\sigma`$
on class $`i`$. All the characters of $`H(4)`$ are real, so the complex conjugation can be dropped. For a product of
more irreps, $`\chi^{(\mu)}_i \chi^{(\nu)}_i`$ is replaced by the product of all their characters. For the derivation
of the formula and the related group theory, we refer the reader to chapter 4 of Jones (1990).

**In the code.** `get_multiplicities` gives the multiplicities $`a_\sigma`$ of a tensor product of any number of
irreps, as a list in the order of `rep_label_list`, and `latex_print_multiplicities` gives the decomposition as a
LaTeX string. The character table and the class orders (the number of elements in each class) are copied from Baake
et al. (1982), and are available as `char_table` (one row per irrep, one column per class) and `class_orders`.

```python
from h4lat import get_multiplicities, latex_print_multiplicities, rep_label_list

muls = get_multiplicities((4, 1), (4, 1))
print({rep_label_list[i]: m for i, m in enumerate(muls) if m})  # {(1, 1): 1, (3, 1): 1, (6, 1): 1, (6, 3): 1}
print(latex_print_multiplicities((4, 1), (4, 1)))
```

The next page, [Clebsch-Gordan coefficients](https://github.com/Tag-E/h4lat/wiki/Clebsch-Gordan-coefficients), shows
how to find the basis in which this decomposition is explicit.

## References

- M. Baake, B. Gemünden and R. Oedingen, *Structure and representations of the symmetry group of the
  four-dimensional cube*, J. Math. Phys. **23**, 944–953 (1982); Erratum: J. Math. Phys. **23**, 2595 (1982),
  [doi:10.1063/1.525461](https://doi.org/10.1063/1.525461).
- M. Göckeler, R. Horsley, E.-M. Ilgenfritz, H. Perlt, P. E. L. Rakow, G. Schierholz and A. Schiller, *Lattice
  operators for moments of the structure functions and their transformation under the hypercubic group*,
  Phys. Rev. D **54**, 5705–5714 (1996), [doi:10.1103/PhysRevD.54.5705](https://doi.org/10.1103/PhysRevD.54.5705),
  [arXiv:hep-lat/9602029](https://arxiv.org/abs/hep-lat/9602029).
- H. F. Jones, *Groups, Representations and Physics* (Adam Hilger, Bristol, 1990).

All the references of the wiki are collected on the [References](https://github.com/Tag-E/h4lat/wiki/References)
page.
