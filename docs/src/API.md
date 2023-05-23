# API

## Covariance Structures

```@docs
CovarianceStructure
```

### Isotropic Covariance Structures

```@docs
IsotropicCovarianceStructure
Linear
Spherical
Exponential
Whittle
SquaredExponential
Gaussian
Matern
```

### Anisotropic Covariance Structures

```@docs
AnisotropicCovarianceStructure
AnisotropicExponential
```

## Covariance Functions

```@docs
AbstractCovarianceFunction
CovarianceFunction
SeparableCovarianceFunction
apply
```

## Gaussian Random Field Generators

```@docs
GaussianRandomFieldGenerator
```

### Cholesky Factorization

```@docs
Cholesky
```

### Spectral Decomposition

```@docs
Spectral
```

### Karhunen Loève Decomposition

```@docs
KarhunenLoeve
rel_error
```

### Circulant Embedding

```@docs
CirculantEmbedding
```

## Gaussian Random Fields

```@docs
GaussianRandomField
sample
randdim
```

## Plotting

Standard plotting functions such as `plot` and `plot!` (for one-dimensional Gaussian random fields), and `heatmap`, `surface`, `contour` and `contourf` (for two-dimensional Gaussian random fields) are implemented. There are also some convenience plotting functions defined:

```@docs
plot_eigenvalues
plot_eigenfunction
plot_covariance_matrix
```

## Unstructured Meshes

```@docs
star
Lshape
```

## Utilities

```@docs
QuadratureRule
Midpoint
Trapezoidal
Simpson
GaussLegendre
EOLE
AbstractEigenSolver
EigenSolver
EigsSolver
```
