module GaussianRandomFields

# dependencies
using SpecialFunctions, Plots, FastGaussQuadrature, Arpack, FFTW

using LinearAlgebra, Statistics

using Base.Cartesian

# import statements
import Plots: plot, plot!, surface, heatmap, contour, contourf

# export statements
export IsotropicCovarianceStructure, AnisotropicCovarianceStructure, CovarianceFunction, apply # from covariance_functions.jl

export SeparableCovarianceFunction # from separable.jl

export Matern # from matern.jl

export Exponential # from exponential.jl

export SquaredExponential, Gaussian # from squaredexponential.jl

export AnisotropicExponential # from anisotropic_exponential.jl

export GaussianRandomField, sample, randdim # from gaussian_random_fields.jl

export Cholesky # from cholesky.jl

export Spectral # from spectral.jl

export KarhunenLoeve, rel_error # from karhunen_loeve.jl

export CirculantEmbedding # from circulant_embedding.jl

export GaussLegendre, EOLE, Simpson, Midpoint, Trapezoidal, AbstractEigenSolver, EigsSolver, EigenSolver, compute # from quadrature.jl

export plot, plot!, heatmap, surface, contour, contourf, plot_eigenvalues, plot_eigenfunction, plot_covariance_matrix # from plots.jl

export star, Lshape # from data/

# include statements
include("covariance_functions/covariance_functions.jl")

include("covariance_functions/matern.jl")

include("covariance_functions/exponential.jl")

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

include("../data/star.jl")

include("../data/Lshape.jl")

include("plot.jl")

end # module
