## runtests.jl : run all test files

using GaussianRandomFields
using Base.Test

# toggle printing mode
verbose = true

# test indexsets
include("test_covariance_functions.jl")
