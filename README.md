# <img src="docs/src/assets/logo.png" alt="alt text" width="75" height="75" align="center"> GaussianRandomFields

| **Documentation** | **Build Status** | **Coverage** | **PkgEval** |
|-------------------|------------------|--------------|-------------|
| [![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://PieterjanRobbe.github.io/GaussianRandomFields.jl/stable) [![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://PieterjanRobbe.github.io/GaussianRandomFields.jl/dev) | [![Build Status](https://github.com/PieterjanRobbe/GaussianRandomFields.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/PieterjanRobbe/GaussianRandomFields.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Build Status](https://ci.appveyor.com/api/projects/status/github/PieterjanRobbe/GaussianRandomFields.jl?svg=true)](https://ci.appveyor.com/project/PieterjanRobbe/GaussianRandomFields-jl) | [![Coverage](https://codecov.io/gh/PieterjanRobbe/GaussianRandomFields.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/PieterjanRobbe/GaussianRandomFields.jl) [![Coverage](https://coveralls.io/repos/github/PieterjanRobbe/GaussianRandomFields.jl/badge.svg?branch=main)](https://coveralls.io/github/PieterjanRobbe/GaussianRandomFields.jl?branch=main) | [![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/G/GaussianRandomFields.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/report.html) |

| **Paper** |
|-----------|
| [![DOI](https://joss.theoj.org/papers/10.21105/joss.05595/status.svg)](https://doi.org/10.21105/joss.05595) |

`GaussianRandomFields` is a Julia package to compute and sample from Gaussian random fields.

<p align="center">
<img align="middle" src="docs/src/assets/examples.jpg" style="width:100%">
</p>

## Key Features

* Support for stationary separable and non-separable isotropic and anisotropic Gaussian random fields.
* We provide most standard covariance functions such as Gaussian, Exponential and Matérn covariances. Adding a user-defined covariance function is very easy.
* Implementation of most common methods to generate Gaussian random fields: Cholesky factorization, eigenvalue decomposition, Karhunen-Loève expansion and circulant embedding.
* Easy generation of Gaussian random fields defined on a Finite Element mesh.
* Versatile plotting features for easy visualisation of Gaussian random fields using [Plots.jl](https://github.com/JuliaPlots/Plots.jl).

## Installation

```
pkg> add GaussianRandomFields  # Press ']' to enter the Pkg REPL mode
```

## Testing

```
pkg> test GaussianRandomFields
```

## Usage

- See the [Tutorial](https://PieterjanRobbe.github.io/GaussianRandomFields.jl/dev/tutorial) for an introduction on how to use this package (including fancy pictures!)
- See the [API](https://PieterjanRobbe.github.io/GaussianRandomFields.jl/dev/API) for a detailed manual

## Contributing

Feel free to open an issue for bug reports, feature requests, or general questions. We encourage new feature additions as pull requests, preferably in a new feature branch.

## Citing

If you find this package useful in your work, feel free to cite

```
@article{robbe2023gaussianrandomfields,
  title={GaussianRandomFields.jl: A Julia package to generate and sample from Gaussian random fields},
  author={Robbe, Pieterjan},
  journal={Journal of Open Source Software},
  volume={8},
  number={89},
  pages={5595},
  year={2023}
}
```

## References

[1] Lord, G. J., Powell, C. E. and Shardlow, T. *An introduction to computational stochastic PDEs*. No. 50. Cambridge University Press, 2014.

[2] Graham, I. G., Kuo, F. Y., Nuyens, D., Scheichl, R. and Sloan, I.H. *Analysis of circulant embedding methods for sampling stationary random fields*. SIAM Journal on Numerical Analysis 56(3), pp. 1871-1895, 2018.

[3] Le Maître, O. and Knio, M. O. *Spectral methods for uncertainty quantification: with applications to computational fluid dynamics*. Springer Science & Business Media, 2010.

[4] Baker, C. T. *The numerical treatment of integral equations*. Clarendon Press, 1977.

[5] Betz, W., Papaioannou I. and Straub, D. *Numerical methods for the discretization of random fields by means of the Karhunen–Loève expansion.* Computer Methods in Applied Mechanics and Engineering 271, pp. 109-129, 2014.
