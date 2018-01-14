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

####################
####################
####################
###
### DONE: covariance_functions // generators
### all docs OK
###
### TODO : implement FE for KL
### TODO Circulant
### TODO analytical
### TODO docs
### TODO readme with highlights, tut
### TODO separable
### TODO gaussian_random_fields
### TODO automatic testing
###
###
####################
####################
####################
module GaussianRandomFields

# dependencies
using SpecialFunctions

using PyPlot

using FastGaussQuadrature

# import statements
import Base: show, -

import PyPlot: plot, surf, contour, contourf, plot_trisurf, tricontourf

# export statements
export CovarianceFunction, SeparableCovarianceFunction # from covariance_functions.jl

export Matern # from matern.jl

export Exponential # from exponential.jl

export SquaredExponential, Gaussian # from squaredexponential.jl

export GaussianRandomField, sample, randdim # from gaussian_random_field.jl

export Cholesky # from cholesky.jl

export Spectral # from spectral.jl

export KarhunenLoeve, rel_error # from karhunen_loeve.jl

export GaussLegendre, EOLE, Simpson, Midpoint, Trapezoidal # from quadrature.jl

export plot, surf, contour, contourf, tricontourf, plot_trisurf, plot_eigenvalues, plot_eigenfunction # from plots.jl

# include statements
include("covariance_functions/covariance_functions.jl")

include("covariance_functions/matern.jl")

include("covariance_functions/exponential.jl")

include("covariance_functions/squared_exponential.jl")

include("gaussian_random_fields.jl")

include("generators/cholesky.jl")

include("generators/spectral.jl")

include("generators/quadrature.jl")

include("generators/karhunen_loeve.jl")

include("separable.jl")

include("plot.jl")

end # module
