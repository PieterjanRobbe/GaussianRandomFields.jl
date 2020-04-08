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

```@docs
plot
plot!
surface
contour
contourf
heatmap
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
