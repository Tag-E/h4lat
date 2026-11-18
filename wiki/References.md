## Group theory and Clebsch-Gordan coefficients

- M. Baake, B. Gemünden and R. Oedingen, *Structure and representations of the symmetry group of the
  four-dimensional cube*, J. Math. Phys. **23**, 944–953 (1982); Erratum: J. Math. Phys. **23**, 2595 (1982),
  [doi:10.1063/1.525461](https://doi.org/10.1063/1.525461).
  The irreps, character table and generator matrices of $`{H(4)}`$ used by h4lat. For the irrep $`{\tau^{(2)}_2}`$, h4lat
  uses a different generator $`{\alpha}`$: see
  [The hypercubic group](https://github.com/Tag-E/h4lat/wiki/The-hypercubic-group#generators).
- I. Sakata, *A general method for obtaining Clebsch-Gordan coefficients of finite groups. I. Its application to point
  and space groups*, J. Math. Phys. **15**, 1702–1709 (1974),
  [doi:10.1063/1.1666528](https://doi.org/10.1063/1.1666528).
  The projection formula used to compute the Clebsch-Gordan coefficients.
- H. F. Jones, *Groups, Representations and Physics* (Adam Hilger, Bristol, 1990).
  Background on finite groups, representations and characters.
- The GAP Group, *GAP – Groups, Algorithms, and Programming*, [www.gap-system.org](https://www.gap-system.org).
  The character table of the 16-cell honeycomb group $`{W(F_4)}`$ was computed with GAP (h4lat builds the matrices of
  the group itself, and its tests check the table without GAP): see
  [The 16-cell honeycomb group](https://github.com/Tag-E/h4lat/wiki/The-16-cell-honeycomb-group#irreducible-representations).

## The 16-cell honeycomb lattice

- S. Katz and D. Nogradi, *QCD on the 16-cell honeycomb*, Phys. Rev. D **114**, 054504 (2026),
  [arXiv:2512.10604](https://arxiv.org/abs/2512.10604).
  The 16-cell honeycomb lattice for lattice QCD, and its smaller cut-off effects.
- S. Katz and D. Nogradi, *Hadron structure on the 16-cell honeycomb*,
  [arXiv:2611.21543](https://arxiv.org/abs/2611.21543).
  The group $`{W(F_4)}`$: the generators, the labels and definitions of the 25 irreps used by h4lat, how they restrict
  to $`{H(4)}`$, and the classification of the operators for the moments of the parton distributions, whose tables
  the tests of h4lat reproduce.

## Lattice operators

- M. Göckeler, R. Horsley, E.-M. Ilgenfritz, H. Perlt, P. E. L. Rakow, G. Schierholz and A. Schiller, *Lattice
  operators for moments of the structure functions and their transformation under the hypercubic group*,
  Phys. Rev. D **54**, 5705–5714 (1996), [doi:10.1103/PhysRevD.54.5705](https://doi.org/10.1103/PhysRevD.54.5705),
  [arXiv:hep-lat/9602029](https://arxiv.org/abs/hep-lat/9602029).
  The classification of the operators $`{\bar\psi \, \Gamma \, D \cdots D \, \psi}`$ under $`{H(4)}`$ and C-parity, and
  their mixing under renormalisation.

## Operators used in published works

The papers whose operators `get_paper_operators` returns are listed, with their DOIs and arXiv numbers, on the page
[Operators from the literature](https://github.com/Tag-E/h4lat/wiki/Operators-from-the-literature#available-papers).

## Citing h4lat

If you use h4lat in your research, please cite it, and give the version you used (`h4lat.__version__`). The DOI
[10.5281/zenodo.23598681](https://doi.org/10.5281/zenodo.23598681) covers all the versions and leads to the latest one.
The BibTeX entry is in the [README](https://github.com/Tag-E/h4lat#how-to-cite), and the "Cite this repository"
button on the GitHub page gives the reference in APA and BibTeX form.
