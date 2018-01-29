# GaussianRandomFields
[![Build Status](https://travis-ci.org/PieterjanRobbe/GaussianRandomFields.jl.png)](https://travis-ci.org/PieterjanRobbe/GaussianRandomFields.jl)
[![Coverage Status](https://coveralls.io/repos/github/PieterjanRobbe/GaussianRandomFields.jl/badge.svg?branch=master)](https://coveralls.io/github/PieterjanRobbe/GaussianRandomFields.jl?branch=master)

A Julia package to compute and sample from Gaussian random fields.

<p align="center">
<img align="middle" src="https://github.com/PieterjanRobbe/GaussianRandomFields.jl/blob/master/figures/front.png" width="800">
</p>

## Key Features

* Generation of stationary (isotropic and anisotropic) and separable non-stationary covariance functions. 
* We provide most standard covariance functions such as Gaussian, Exponential and Mat&eacute;rn. Adding a custom covariance function is very easy.
* Implementation of most common methods to generate Gaussian random fields: Cholesky factorization, Karhunen-Lo&egrave;ve expansion, circulant embedding.
* Easy generation of Gaussian random fields defined on a Finite Element mesh.
* Versatile plotting features for easy visualisation of Gaussian random fields.

## Examples

Read the [tutorial](tutorial/tutorial.ipynb) for details and examples on how to use this package.

## References

[1] Lord, Gabriel J., Catherine E. Powell, and Tony Shardlow. An introduction to computational stochastic PDEs. No. 50. Cambridge University Press, 2014.

