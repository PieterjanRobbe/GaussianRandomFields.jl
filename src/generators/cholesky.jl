## cholesky.jl : Gaussian random field generator based on the Cholesky decomposition of the covariance matrix

"""
    Cholesky <: GaussianRandomFieldGenerator

A [`GaussiandRandomFieldGenerator`](@ref) based on a Cholesky factorization of the covariance matrix.

# Examples
```jldoctest
julia> m = Matern(0.1,1.0)
Matérn (λ=0.1, ν=1.0, σ=1.0, p=2.0)

julia> c = CovarianceFunction(2,m)
2d Matérn covariance function (λ=0.1, ν=1.0, σ=1.0, p=2.0)

julia> pts1 = 0:0.02:1; pts2 = 0:0.02:1 
0.0:0.02:1.0

julia> grf = GaussianRandomField(c,Cholesky(),pts1,pts2)
Gaussian random field with 2d Matérn covariance function (λ=0.1, ν=1.0, σ=1.0, p=2.0) on a 51x51 structured grid, using a Cholesky decomposition

julia> plot(grf) 
[...]

```
Note that the Cholesky factorization requires the covariance matrix to be `Symmetric` and positive definite. If the covariance matrix is not `Symmetric`, a warning will be thrown but the method will try to continue with an approximate symmetric matrix. If the covariance matrix is not positive definite, an error will be thrown. Try using the `Spectral` method in that case.

See also: [`Spectral`](@ref), [`KarhunenLoeve`](@ref), [`CirculantEmbedding`](@ref)
"""
struct Cholesky <: GaussianRandomFieldGenerator end 

const CholeskyGRF = GaussianRandomField{C,Cholesky} where {C}

# compose a GaussianRandomField using Cholesky factorization
function _GaussianRandomField(mean,cov,method::Cholesky,pts...)
    C = apply(cov,pts,pts)

    # error checking
    issymmetric(C) || warn("to use a Cholesky factorization, the covariance matrix must be symmetric/hermitian")
    isposdef(C) || throw(ArgumentError("to use a Cholesky factorization, the covariance matrix must be positive definite"))

    # compute Cholesky factorization
    L = (cholesky(Symmetric(C))).U'
    
    GaussianRandomField{typeof(cov),Cholesky,typeof(pts)}(mean,cov,pts,L)
end


# returns the required dimension of the random points
randdim(grf::CholeskyGRF) = size(grf.data,1) 

# sample from the GaussianRandomField using Cholesky factorization
function _sample(grf::CholeskyGRF, xi)
	grf.mean + std(grf.cov)*reshape(grf.data*xi,size(grf.mean))
end

show(io::IO,::Cholesky) = print(io,"Cholesky decomposition")
