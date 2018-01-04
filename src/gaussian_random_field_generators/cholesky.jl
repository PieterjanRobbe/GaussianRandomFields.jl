# TODO jldoctest
## cholesky.jl : Gaussian random field sampler based on the Cholesky decomposition of the covariance matrix

"""
`struct Cholesky`

Implements a Gaussiand random field sampler based on the Cholesky factorization of the covariance matrix.

Examples:
```
```
"""
struct Cholesky <: GaussianRandomFieldGenerator end 

function _GaussianRandomField(mean,cov,method::Cholesky,pts...)

    # evaluate covariance function
    C = apply(cov.cov,pts,pts)
    
    # error checking
    if !ishermitian(C)
        warn("to use a Cholesky factorization, the covariance matrix must be symmetric/hermitian")
        C = Hermitian(C)
    end
    if !isposdef(C)
        throw(ArgumentError("to use a Cholesky factorization, the covariance matrix must be positive definite (SPD)"))
    end

    # compute Cholesky factorization
    L = chol(C)'

    GaussianRandomField{typeof(cov),Cholesky}(mean,cov,pts,L)
end

# returns the required dimension of the random points
randdim(grf::GaussianRandomField{C,Cholesky} where {C}) = size(grf.data,1) 

function _sample(grf::GaussianRandomField{C,Cholesky} where {C}, xi)
    grf.mean + grf.cov.cov.σ*reshape(grf.data*xi,length.(grf.pts))
end
