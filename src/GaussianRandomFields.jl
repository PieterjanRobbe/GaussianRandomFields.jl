# TODO plotting ; docs ; test ; examples
# TODO make notebook with interact
# TODO make CirculantEmbedding
# TODO make TurningBand
####################
#
#  --> code: FEM (zie spanos?)
#  --> docs: alles met @Ref's
#  --> jldoctest
#  --> analytical
#  --> make readme; make notebook with plots
#  --> other methods are future work
#  --> automated testing: have series of kernels + methods ready
#
####################
module GaussianRandomFields

# dependencies
using SpecialFunctions

using PyPlot

using FastGaussQuadrature

# import statements
import Base: show, -

import PyPlot: plot, surf, contour, contourf

# export statements
export CovarianceFunction, SeparableCovarianceFunction # from covariance_functions.jl

export Matern # from matern.jl

export Exponential # from exponential.jl

export SquaredExponential # from squaredexponential.jl

export GaussianRandomField, sample, randdim # from gaussian_random_field.jl

export Cholesky # from cholesky.jl

export Spectral # from spectral.jl

export KarhunenLoeve, GaussLegendre, EOLE, rel_error # from karhunen_loeve.jl

export plot, surf, contour, contourf, plot_eigenvalues, plot_eigenfunction # from plots.jl

# include statements
include("covariance_functions.jl")

include("covariance_functions/matern.jl")

include("covariance_functions/exponential.jl")

include("covariance_functions/squared_exponential.jl")

include("gaussian_random_field_generators.jl")

include("gaussian_random_fields.jl")

include("gaussian_random_field_generators/cholesky.jl")

include("gaussian_random_field_generators/spectral.jl")

include("gaussian_random_field_generators/karhunen_loeve.jl")

include("separable.jl")

include("plot.jl")

end # module
