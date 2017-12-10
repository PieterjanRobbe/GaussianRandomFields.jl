# TODO plotfield
# TODO make  GaussianRandomField(::CovarianceFunction,::RandomFieldSampler)
# TODO make notebook with interact
# TODO make RandomFieldSampler
# TODO make KarhunenLoeveExpansion
# TODO make CirculantEmbedding
# TODO make CholeskyFactorization
# TODO make TurningBand
# TODO add variance and mean to CovarianceFunction ! See slides
# TODO spectral methods???
module GaussianRandomFields

# dependencies
using SpecialFunctions
using Plots
plotly()

# import satements
import Base.show

import Plots.plot

# export statements
export CovarianceFunction, SeparableCovarianceFunction, Matern, Exponential, SquaredExponential # from covariance_functions.jl

export GaussianRandomField, sample # from gaussian_random_field.jl

export Cholesky # from cholesky.jl

export plot # from plots.jl

# include statements
include("covariance_functions.jl")

include("gaussian_random_field_generators.jl")

include("gaussian_random_fields.jl")

include("cholesky.jl")

include("plot.jl")

end # module
