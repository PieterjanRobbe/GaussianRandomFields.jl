---
title: 'GaussianRandomFields.jl: A Julia package to generate and sample from Gaussian random fields'
tags:
  - Julia
  - random fields
  - uncertainty quantification
  - statistics
authors:
  - name: Pieterjan Robbe
    orcid: 0000-0002-6254-8245
    affiliation: "1"
affiliations:
 - name: KU Leuven, Belgium
   index: 1
date: 27 May 2023
bibliography: paper.bib
---

# Summary

Random fields are used to represent spatially-varying uncertainty, and are commonly used as training data in uncertainty quantification and machine learning applications. GaussianRandomFields.jl is a Julia [@bezanson2017] software package to generate and sample from Gaussian random fields. It offers support for well-known covariance functions, such as Gaussian, exponential and Matérn covariances [@bishop2006; @chiles2012; @montero2015], as well as user-defined covariance structures defined on arbitrary domains. The package implements most common methods to generate samples from these random fields, including the Cholesky factorization, the Karhunen-Loève expansion, and the circulant embedding method [@lord2014]. GaussianRandomFields.jl makes use of [Plots.jl](https://github.com/JuliaPlots/Plots.jl) [@PlotsJL] to quickly visualize samples of the random fields.

# Statement of need

Gaussian random fields are used by scientists to describe complex patterns and structures emerging in nature. They provide a statistical tool for describing a vast amount of different structures found in various applications such as electronics [@cui2018], geostatistics [@pirot2015], machine learning [@stephenson2006] and cosmology [@chiang2000]. Random fields can be viewed as the extension from random variables to random functions, taking random values at different points Gaussian random fields are particularly attractive, because they only require two parameters to be fully defined: a mean value and a covariance function. GaussianRandomFields.jl provides Julia implementations of stationary separable and non-separable isotropic and anisotropic Gaussian random fields. It has been used in a number of recent works, including [@blondeel2020], [@robbe2021] and [@wu2023].

There are already some packages for Gaussian random field generation available in R [@randomfields] and Python [@GSTools]. GaussianRandomFields.jl offers a native Julia implementation. As such, it benefits from the performance advantage of Julia, and provides a convenient unified API for different covariance functions by leveraging multiple dispatch. A particular example are the covariance functions from [KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl), which can easily be linked to the Gaussian random field generators implemented in this package.

# Usage

The following example is an excerpt from the [tutorial](https://pieterjanrobbe.github.io/GaussianRandomFields.jl/stable/) (see \autoref{fig:samples}).

```julia
using GaussianRandomFields, Plots

cov = CovarianceFunction(2, Exponential(.5))
pts = range(0, stop=1, length=1001)
grf = GaussianRandomField(cov, CirculantEmbedding(), pts, pts, minpadding=2001)
heatmap(grf)
```

![Three realizations of a two-dimensional Gaussian random field with exponential covariance function.\label{fig:samples}](exponential_grf.jpg)

# Acknowledgements

The author acknowledges the help of David Widmann implementing a dimension-independent variant of the circulant embedding method using `@generated` functions.

# References