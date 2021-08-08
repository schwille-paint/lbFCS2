[dna-paint]:https://www.nature.com/articles/nprot.2017.024
[lbfcs-git]: https://github.com/schwille-paint/lbFCS
[lbfcs-paper]: https://pubs.acs.org/doi/abs/10.1021/acs.nanolett.9b03546
[conda]:https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
[picasso]: https://github.com/jungmannlab/picasso
[picasso_addon-git]: https://github.com/schwille-paint/picasso_addon
[picasso_addon-doi]: https://doi.org/10.5281/zenodo.4792396
[picasso_addon-installation]: https://picasso-addon.readthedocs.io/en/latest/installation.html
[spt-git]: https://github.com/schwille-paint/SPT
[spt-paper]: https://www.nature.com/articles/s41467-021-24223-4

<!--- Comments -->

# lbFCS2

This python package lbFCS2 allows to determine absolute docking strand numbers and local hyridization rates
in individual [DNA-PAINT][dna-paint] clusters requiring only a single [DNA-PAINT][dna-paint] image aquistion.
It is a revised framework of the previously published [lbFCS][lbfcs-git] package 
as published in ['Towards absolute molecular numbers in DNA-PAINT'][lbfcs-paper].

<img src="/docs/figures/fig01.png" alt="principle" width="1000">

It is suited for the application to well-separated [DNA-PAINT][dna-paint] localization clusters containing only low 
molecular numbers of up to six docking strands.

For ... 
* ... automated localization of the raw images,
* ... automated undrifting,
* ... and automated detection and isolation of all localization clusters within a DNA-PAINT image ...

it requires the [picasso_addon][picasso_addon-git] based on the [picasso][picasso] package.
A fixed release [doi.org/10.5281/zenodo.4792396][picasso_addon-doi] of [picasso_addon][picasso_addon-git] 
was already used in ['Tracking single particles for hours via continuous DNA-mediated fluorophore exchange'][spt-paper]
(corresponding python package can be found at [SPT][spt-git]).

The complete installation instructions of the [picasso_addon][picasso_addon-git] and [picasso][picasso] package 
and creation of the necessary [conda][conda] environment compatible with [picasso_addon][picasso_addon-git],
[picasso][picasso], [SPT][spt-git] and lbFCS2 can be found here [picasso_addon: Installation instructions][picasso_addon-installation].

All scripts for simple evaluation of data can be found in [/scripts](/scripts/).



