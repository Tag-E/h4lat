*Theory, part 2 of 3.* Previous: [The hypercubic group](https://github.com/Tag-E/h4lat/wiki/The-hypercubic-group) ·
Next: [Lattice operators](https://github.com/Tag-E/h4lat/wiki/Lattice-operators)

## From the decomposition to the coefficients

With Eq. (3) of the [previous page](https://github.com/Tag-E/h4lat/wiki/The-hypercubic-group#tensor-products-and-the-clebsch-gordan-series)
we know how to decompose a product of irreps into a direct sum. For reasons explained on the
[Lattice operators](https://github.com/Tag-E/h4lat/wiki/Lattice-operators) page, a case of direct physical interest
is $`\tau^{(4)}_1 \otimes \tau^{(4)}_1`$, whose Clebsch-Gordan series is

```math
\tau^{(4)}_1 \otimes \tau^{(4)}_1 = \tau^{(1)}_1 \oplus \tau^{(3)}_1 \oplus \tau^{(6)}_1 \oplus \tau^{(6)}_3 . \tag{1}
```

For an application to physics, however, the decomposition alone is not enough. We also need to know how the basis
vectors of the tensor product can be recombined so that the decomposition on the right-hand side of Eq. (1) becomes
manifest. In other words, we need the change-of-basis matrix, whose entries are the **Clebsch-Gordan coefficients**
of $`H(4)`$.

A general method to obtain the Clebsch-Gordan coefficients of a finite group was given by
[Sakata (1974)](https://doi.org/10.1063/1.1666528). In the next section we explain it for $`H(4)`$; then we clarify
its meaning with the example of Eq. (1).

## Sakata's formula

For $`H(4)`$, the recipe of Sakata (1974) is as follows.

1. Consider $`\mu \otimes \nu`$, the product of two irreps of $`H(4)`$. (The method, and the code, work in the same
   way for a product of any number of irreps.)
2. Compute the decomposition of $`\mu \otimes \nu`$ with the character table of $`H(4)`$ and Eq. (3) of the
   [previous page](https://github.com/Tag-E/h4lat/wiki/The-hypercubic-group#tensor-products-and-the-clebsch-gordan-series).
3. For every element $`h`$ of $`H(4)`$, consider the following matrices:
   - $`D^{(\mu)}(h)`$ and $`D^{(\nu)}(h)`$, the matrices of $`h`$ in the irreps $`\mu`$ and $`\nu`$, and their tensor
     (Kronecker) product $`T(h) = D^{(\mu)}(h) \otimes D^{(\nu)}(h)`$, the matrix of $`h`$ on the product space;
   - $`D(h)`$, the block-diagonal matrix with one block $`D^{(\sigma)}(h)`$ for each irrep $`\sigma`$ in the
     decomposition of $`\mu \otimes \nu`$, repeated as many times as its multiplicity.
4. Take an arbitrary matrix $`A`$ of the same size as $`T(h)`$, i.e. a matrix whose entries are free symbols that are
   fixed at the end, and compute

   ```math
   M^{(\mu \otimes \nu)} = \sum_{h \in H(4)} T(h) \, A \, D(h)^\dagger . \tag{2}
   ```

   All the matrices of $`H(4)`$ used by h4lat are real and orthogonal, so $`D(h)^\dagger = D(h)^T = D(h)^{-1}`$.
5. Examine the columns of $`M^{(\mu \otimes \nu)}`$. By normalising them appropriately (i.e. by fixing the values of
   the entries of $`A`$ that appear in the columns under consideration), one can read off the Clebsch-Gordan
   coefficients of interest. Each column gives the coefficients of one operator.

The formula works because the sum runs over the whole group: for every element $`g`$, the matrix
$`M = M^{(\mu \otimes \nu)}`$ satisfies

```math
T(g) \, M = \sum_{h} T(gh) \, A \, D(h)^{-1} = \sum_{h'} T(h') \, A \, D(g^{-1}h')^{-1}
= \sum_{h'} T(h') \, A \, D(h')^{-1} D(g) = M \, D(g) ,
```

where $`h' = gh`$ runs over the whole group as $`h`$ does. Therefore the columns of $`M`$ above a block $`D^{(\sigma)}`$ transform exactly as the irrep $`\sigma`$, whatever the
values of the entries of $`A`$ (as long as the block is not zero). The free symbols only parametrise the freedom that
is left: the normalisation of each block and, when $`\sigma`$ appears more than once, the choice of its copies.

**In the code.** The matrices $`D^{(\mu)}(h)`$ are available for every irrep $`\mu`$ and every element $`h`$ (see
[The hypercubic group](https://github.com/Tag-E/h4lat/wiki/The-hypercubic-group#the-384-matrices)). The class
`cg_calc` implements Eq. (2) and gives the Clebsch-Gordan coefficients of any tensor product. The sum over the group is
done numerically, on the coefficients of the symbols of $`A`$, optionally in parallel; the result does not depend on
the number of threads. The entries of $`A`$ are then fixed by `CGmat_from_block`, as explained
[below](https://github.com/Tag-E/h4lat/wiki/Clebsch-Gordan-coefficients#fixing-the-free-parameters), so that, among
other things, each copy of an irrep has a definite C-parity.

## An explicit example

To make things more transparent, let us introduce the lattice operators related to the decomposition of Eq. (1),

```math
\mathcal{O}_{\mu_1 \mu_2} = \bar\psi \, \gamma_{\mu_1} \overleftrightarrow{D}_{\mu_2} \, \psi .
```

More details are given on the [Lattice operators](https://github.com/Tag-E/h4lat/wiki/Lattice-operators) page, and
the complete physical background can be found, for example, in Göckeler et al. (1996). The important fact here is
that each index $`\mu_i`$ transforms according to $`\tau^{(4)}_1`$, the vector irrep of $`H(4)`$.

Since this irrep is four-dimensional, the operators form a 16-dimensional space. The basis one would naively consider
is $`\mathcal{O}_{11}, \mathcal{O}_{12}, \dots, \mathcal{O}_{43}, \mathcal{O}_{44}`$. However, this basis is not
suitable for physics applications, because its operators do not belong to definite irreps of $`H(4)`$. From Eq. (1)
we already know that the operators can be recombined so that one of them transforms according to $`\tau^{(1)}_1`$,
three according to $`\tau^{(3)}_1`$, six according to $`\tau^{(6)}_1`$ and the last six according to
$`\tau^{(6)}_3`$.

To get an explicit expression for the operators of the new basis, we use Sakata's formula, Eq. (2), and compute
$`M^{(\tau^{(4)}_1 \otimes \tau^{(4)}_1)}`$. This is a $`16 \times 16`$ change-of-basis matrix containing the
Clebsch-Gordan coefficients of interest. Its rows are numbered according to the old basis,
$`\mathcal{O}_{11}, \mathcal{O}_{12}, \dots, \mathcal{O}_{44}`$ (the operator $`\mathcal{O}_{\mu_1\mu_2}`$ is row
$`4(\mu_1 - 1) + \mu_2`$), and its columns according to the new one. Its first column is

```math
\begin{pmatrix}
f(A) \\ 0 \\ 0 \\ 0 \\ 0 \\ f(A) \\ 0 \\ 0 \\ 0 \\ 0 \\ f(A) \\ 0 \\ 0 \\ 0 \\ 0 \\ f(A)
\end{pmatrix} ,
```

where $`f(A)`$ is a linear function of the entries of $`A`$ whose precise form does not matter (it is
$`96\,(A_{1,1} + A_{6,1} + A_{11,1} + A_{16,1})`$). We are free to choose its value, and we choose the simplest one,
$`f(A) = 1`$. Remembering how the 16 rows are numbered, we can read the first operator of the new basis from the first
column:

```math
\mathcal{O}_{11} + \mathcal{O}_{22} + \mathcal{O}_{33} + \mathcal{O}_{44} .
```

This operator, the trace, transforms according to the irrep $`\tau^{(1)}_1`$.

The next three columns of $`M^{(\tau^{(4)}_1 \otimes \tau^{(4)}_1)}`$ form the block of $`\tau^{(3)}_1`$:

```math
\begin{pmatrix}
f(A) & \sqrt{2}\, f(A) & \sqrt{6}\, f(A) \\
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0 \\
f(A) & \sqrt{2}\, f(A) & -\sqrt{6}\, f(A) \\
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0 \\
f(A) & -2\sqrt{2}\, f(A) & 0 \\
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0 \\
-3 f(A) & 0 & 0
\end{pmatrix} ,
```

where $`f(A)`$ is again a linear function of the entries of $`A`$, but a different one: each block depends on its own
columns of $`A`$. Within a block, instead, the same $`f(A)`$ appears in every column, with only a fixed numerical
coefficient in front (here $`1`$, $`\sqrt{2}`$ and $`\sqrt{6}`$). This is a consequence of Schur's lemma: a copy of an
irrep is fixed by a single number, its normalisation.

> **Multiplicity greater than one.** When an irrep appears $`m > 1`$ times in the product, its blocks depend on $`m`$
> different linear combinations $`f_1(A), \dots, f_m(A)`$, which can be fixed separately. They give rise to $`m`$
> linearly independent copies of the irrep, which all transform with the same matrices $`D^{(\sigma)}(h)`$.

Fixing the normalisation of each column (h4lat makes its first non-zero entry equal to $`\pm 1`$, see
[below](https://github.com/Tag-E/h4lat/wiki/Clebsch-Gordan-coefficients#fixing-the-free-parameters)), these columns
provide us with the three operators

```math
\mathcal{O}_{11} + \mathcal{O}_{22} + \mathcal{O}_{33} - 3\,\mathcal{O}_{44}, \qquad
\mathcal{O}_{11} + \mathcal{O}_{22} - 2\,\mathcal{O}_{33}, \qquad
\mathcal{O}_{11} - \mathcal{O}_{22} ,
```

which transform according to the irrep $`\tau^{(3)}_1`$. Continuing in the same way, we obtain the antisymmetric
operators of $`\tau^{(6)}_1`$ and the symmetric ones of $`\tau^{(6)}_3`$,

```math
\mathcal{O}_{\mu\nu} - \mathcal{O}_{\nu\mu} \quad \text{and} \quad \mathcal{O}_{\mu\nu} + \mathcal{O}_{\nu\mu},
\qquad \mu < \nu ,
```

with the pairs $`(\mu, \nu)`$ in the order 12, 13, 23, 14, 24, 34.

This procedure can be used for a general tensor product. In particular, applying it to
$`\tau^{(4)}_1 \otimes \tau^{(4)}_1`$, $`\tau^{(4)}_1 \otimes \tau^{(4)}_1 \otimes \tau^{(4)}_1`$ and
$`\tau^{(4)}_1 \otimes \tau^{(4)}_1 \otimes \tau^{(4)}_1 \otimes \tau^{(4)}_1`$, we reproduce the irrep decomposition
and the operators given in Göckeler et al. (1996), up to the choice of basis within each irrep. In h4lat this basis is
fixed in the same way for every product and every copy of an irrep: the Clebsch-Gordan coefficients of every irrep
transform with the same explicit matrices $`D^{(\sigma)}(h)`$, and the copies of an irrep that appears more than once
have a definite C-parity.

## Fixing the free parameters

The function `CGmat_from_block` turns each symbolic block of Eq. (2) into numbers. It fixes the free symbols in a
deterministic way, so that the result is the same at every run.

- **Normalisation.** Each column is divided by the absolute value of its first non-zero entry, which therefore becomes
  $`\pm 1`$ ($`+1`$ in the first column of a block), as in the example above. This fixes the normalisation of every
  operator separately, so the columns of a block no longer have the same norm. Once rescaled to unit norm, they
  transform exactly with the orthogonal matrices of the irrep: $`T(h)\, C = C\, D^{(\sigma)}(h)`$ for every element
  $`h`$, where $`C`$ is the block with unit columns. The test suite checks this for every bundled block.
- **Copies.** By Schur's lemma, a copy of an irrep is fixed by the first column of its block. When an irrep appears
  $`m > 1`$ times, h4lat takes as reference rows the first $`m`$ rows of the first column whose entries are linearly
  independent functions of $`A`$. Copy $`j`$ is the block for the $`A`$ of minimal norm that makes reference row $`j`$
  equal to 1 and the other reference rows equal to 0. The copies are therefore linearly independent, and each one is
  an exact Clebsch-Gordan block, since it is $`M`$ for one value of $`A`$.
- **C-parity.** For an operator $`\bar\psi \, \Gamma \, D \cdots D \, \psi`$, the factors of the product after the
  first one are the derivative indices, and charge conjugation reverses their order. When all those factors are the
  same irrep (for the operators of h4lat they are all $`\tau^{(4)}_1`$), this reversal commutes with the action of
  $`H(4)`$, so it maps the copies of an irrep into each other. Since it squares to one, the copies can be chosen even
  or odd under it, and h4lat chooses them in this way, the even copies first. Every operator built from them
  therefore has a definite C-parity (see [Lattice operators](https://github.com/Tag-E/h4lat/wiki/Lattice-operators#c-parity)).
  An irrep with multiplicity 1 is automatically even or odd.

Until October 2026, the copies were chosen without the C-parity condition, so some operators had mixed C-parity, and
the operator IDs of irreps with multiplicity greater than 1 referred to other copies.

## Bundled products

The Clebsch-Gordan coefficients of the following products are computed in advance and bundled with h4lat:

```math
X \otimes \underbrace{\tau^{(4)}_1 \otimes \cdots \otimes \tau^{(4)}_1}_{k \text{ factors}}, \qquad
X \in \left\{ \tau^{(1)}_1, \tau^{(1)}_4, \tau^{(4)}_1, \tau^{(4)}_4, \tau^{(6)}_1 \right\}, \qquad k = 0, 1, 2, 3 .
```

These are the products needed for the operators $`\bar\psi \, \Gamma \, D \cdots D \, \psi`$ with up to three
derivatives (see [Lattice operators](https://github.com/Tag-E/h4lat/wiki/Lattice-operators)), e.g.
$`\tau^{(4)}_1 \otimes \tau^{(4)}_1`$ and $`\tau^{(4)}_4 \otimes \tau^{(4)}_1`$. For $`k = 0`$ the "product" is the irrep
$`X`$ alone, whose Clebsch-Gordan matrix is the identity. The symbolic blocks of Eq. (2) are bundled only for
$`k \le 2`$, because for $`k = 3`$ they would take tens of MB. Any other product is computed on demand: see
[Computing Clebsch-Gordan coefficients](https://github.com/Tag-E/h4lat/wiki/Computing-Clebsch-Gordan-coefficients).

```python
from h4lat import cg_calc, irrep_index

cg = cg_calc((4, 1), (4, 1), verbose=False)  # loaded from the bundled database
C = cg.cg_dict[irrep_index[(3, 1)]][0]       # the block of the first (here the only) copy of (3,1): shape (16, 3)
print(C[:, 0])                               # 1 at the rows of O_11, O_22, O_33 and -3 at the row of O_44
```

## References

- I. Sakata, *A general method for obtaining Clebsch-Gordan coefficients of finite groups. I. Its application to point
  and space groups*, J. Math. Phys. **15**, 1702–1709 (1974),
  [doi:10.1063/1.1666528](https://doi.org/10.1063/1.1666528).
- M. Göckeler, R. Horsley, E.-M. Ilgenfritz, H. Perlt, P. E. L. Rakow, G. Schierholz and A. Schiller, *Lattice
  operators for moments of the structure functions and their transformation under the hypercubic group*,
  Phys. Rev. D **54**, 5705–5714 (1996), [doi:10.1103/PhysRevD.54.5705](https://doi.org/10.1103/PhysRevD.54.5705),
  [arXiv:hep-lat/9602029](https://arxiv.org/abs/hep-lat/9602029).

All the references of the wiki are collected on the [References](https://github.com/Tag-E/h4lat/wiki/References)
page.
