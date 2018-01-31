# GaussianRandomFields
[![Build Status](https://travis-ci.org/PieterjanRobbe/GaussianRandomFields.jl.png)](https://travis-ci.org/PieterjanRobbe/GaussianRandomFields.jl)
[![Coverage Status](https://coveralls.io/repos/github/PieterjanRobbe/GaussianRandomFields.jl/badge.svg?branch=master)](https://coveralls.io/github/PieterjanRobbe/GaussianRandomFields.jl?branch=master)

A Julia package to compute and sample from Gaussian random fields.

<p align="center">
<img align="middle" src="https://github.com/PieterjanRobbe/GaussianRandomFields.jl/blob/master/figures/front.png" width="800">
</p>

## Key Features

* Generation of stationary (isotropic and anisotropic) and separable non-stationary covariance functions. 
* We provide most standard covariance functions such as Gaussian, Exponential and Mat&eacute;rn covariances. Adding a user-defined covariance function is very easy.
* Implementation of most common methods to generate Gaussian random fields: Cholesky factorization, Karhunen-Lo&egrave;ve expansion and circulant embedding.
* Easy generation of Gaussian random fields defined on a Finite Element mesh.
* Versatile plotting features for easy visualisation of Gaussian random fields.

## Examples

Read the [tutorial](tutorial/tutorial.ipynb) for details and examples on how to use this package.

## References

[1] Lord, G. J., Powell, C. E. and Shardlow, T. *An introduction to computational stochastic PDEs*. No. 50. Cambridge University Press, 2014.

[2] Graham, I. G., Kuo, F. Y., Nuyens, D., Scheichl, R. and Sloan, I.H. *Analysis of circulant embedding methods for sampling stationary random fields*. [ArXiv preprint](https://arxiv.org/abs/1710.00751), 2017.

[3] Le Maître, O. and Knio, M. O. *Spectral methods for uncertainty quantification: with applications to computational fluid dynamics*. Springer Science & Business Media, 2010.

[4] Baker, C. T. *The numerical treatment of integral equations*. Clarendon Press, 1977.

[5] Betz, W., Papaioannou I. and Straub, D. *Numerical methods for the discretization of random fields by means of the Karhunen–Loève expansion.* Computer Methods in Applied Mechanics and Engineering 271, pp. 109-129, 2014.