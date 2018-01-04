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
    C = apply(cov,pts,pts)
	L = choleskyfy(C)    

	GaussianRandomField{typeof(cov),Cholesky,typeof(pts)}(mean,cov,pts,L)
end

# note: transpose is for efficiency
function _GaussianRandomField(mean,cov,method::Cholesky,p::Matrix{T},t::Matrix{T}) where {T}
	C = apply(cov,p,p)
	L = choleskyfy(C)

	pts = (p,t)
	GaussianRandomField{typeof(cov),Cholesky,typeof(pts)}(mean,cov,pts,L)
end

function choleskyfy(C)
    issymmetric(C) || warn("to use a Cholesky factorization, the covariance matrix must be symmetric/hermitian")
    isposdef(C) || throw(ArgumentError("to use a Cholesky factorization, the covariance matrix must be positive definite"))

	chol(Symmetric(C))'
end

# returns the required dimension of the random points
randdim(grf::GaussianRandomField{C,Cholesky} where {C}) = size(grf.data,1) 

function _sample(grf::GaussianRandomField{C,Cholesky,NTuple{N,T}} where {C,N,T<:AbstractVector}, xi)
    grf.mean + grf.cov.cov.σ*reshape(grf.data*xi,length.(grf.pts))
end

function _sample(grf::GaussianRandomField{C,Cholesky,NTuple{N,T}} where {C,N,T<:AbstractMatrix}, xi)
    grf.mean + grf.cov.cov.σ* ( grf.data*xi )
end
