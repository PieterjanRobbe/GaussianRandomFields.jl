## runtests.jl : run all test files

# using PyPlot   # *****
using GaussianRandomFields
using Test
using Suppressor

# test indexsets
include("test_covariance_functions.jl")
include("test_cholesky.jl")
include("test_spectral.jl")
include("test_karhunen_loeve.jl")
# include("test_circulant_embedding.jl")    # *****
include("test_gaussian_random_fields.jl")
include("test_separable.jl")
# include("test_fem.jl")     # *****
# include("test_plot.jl")    # *****
include("test_show.jl")
