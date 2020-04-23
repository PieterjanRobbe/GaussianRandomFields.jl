module GaussianRandomFields

# dependencies
using SpecialFunctions, RecipesBase, FastGaussQuadrature, Arpack, FFTW

using LinearAlgebra, Statistics

using Base.Cartesian

# export statements
export CovarianceStructure, IsotropicCovarianceStructure, AnisotropicCovarianceStructure, AbstractCovarianceFunction, CovarianceFunction, apply # from covariance_functions.jl

export SeparableCovarianceFunction # from separable.jl

export Matern # from matern.jl

export Exponential # from exponential.jl

export Linear # from linear.jl

export Spherical # from spherical.jl

export Whittle # from whittle.jl

export SquaredExponential, Gaussian # from squaredexponential.jl

export AnisotropicExponential # from anisotropic_exponential.jl

export GaussianRandomFieldGenerator, GaussianRandomField, sample, randdim # from gaussian_random_fields.jl

export Cholesky # from cholesky.jl

export Spectral # from spectral.jl

export KarhunenLoeve, rel_error # from karhunen_loeve.jl

export CirculantEmbedding # from circulant_embedding.jl

export QuadratureRule, GaussLegendre, EOLE, Simpson, Midpoint, Trapezoidal, AbstractEigenSolver, EigsSolver, EigenSolver # from quadrature.jl

#export plot, plot!, heatmap, surface, contour, contourf, plot_eigenvalues, plot_eigenfunction, plot_covariance_matrix # from plots.jl

export star, Lshape # from fem_data/

# include statements
include("covariance_functions/covariance_functions.jl")

include("covariance_functions/matern.jl")

include("covariance_functions/exponential.jl")

include("covariance_functions/linear.jl")

include("covariance_functions/spherical.jl")

include("covariance_functions/whittle.jl")

include("covariance_functions/squared_exponential.jl")

include("covariance_functions/anisotropic_exponential.jl")

include("gaussian_random_fields.jl")

include("fem.jl")

include("generators/cholesky.jl")

include("generators/spectral.jl")

include("generators/quadrature.jl")

include("generators/karhunen_loeve.jl")

include("generators/circulant_embedding.jl")

include("analytic.jl")

include("separable.jl")

include("../fem_data/star.jl")

include("../fem_data/Lshape.jl")

include("plot.jl")

end # module
