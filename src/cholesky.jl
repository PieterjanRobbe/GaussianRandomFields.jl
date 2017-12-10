## cholesky_factorization.jl ##
# Gaussian random field sampler based on the Cholesky decomposition of the covariance matrix

"""
`struct Cholesky`

Implements a Gaussiand random field sampler based on the Cholesky factorization of the covariance matrix.

Examples:
```
```
"""
struct Cholesky <: GaussianRandomFieldGenerator end 

# TODO C must be SPD??
function GaussianRandomField(cov::CovarianceFunction,method::Cholesky,pts::Array{T,N}) where {T<:AbstractFloat,N}
    C = apply(cov,pts,pts)
    ishermitian(C) || warn("To use a Cholesky factorization, the covariance matrix must be symmetric/hermitian. Consider using a spectral (eigenvalue) decomposition using the Spectral() method. I will try to continue without guarantees...")
    L = chol(Hermitian(C))'
    GaussianRandomField{typeof(cov),Cholesky}(cov,pts,L)
end

function sample(grf::GaussianRandomField{C,Cholesky} where {C}; xi::Vector{T} where {T<:AbstractFloat}=randn(size(grf.data,1)))
    grf.data*xi
end
