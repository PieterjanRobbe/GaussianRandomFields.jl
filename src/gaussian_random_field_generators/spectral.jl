# TODO jldoctest
## spectral.jl : Gaussian random field sampler based on the spectral (eigenvalue) decomposition of the covariance matrix
 
"""
`struct Spectral`

Implements a Gaussiand random field sampler based on the spectral (eigenvalue) decomposition of the covariance matrix. Usefull when the covariance matrix is not SPD. 

Examples:
```
```
"""
struct Spectral <: GaussianRandomFieldGenerator end 

mutable struct SpectralData
    eigenval
    eigenfunc
end

function _GaussianRandomField(mean,cov,method::Spectral,pts...)
    C = apply(cov,pts,pts)
	data = spectralify(C)

	GaussianRandomField{typeof(cov),Spectral,typeof(pts)}(mean,cov,pts,data)
end

function _GaussianRandomField(mean,cov,method::Spectral,p::Matrix,t::Matrix)
	C = apply(cov,p,p)
	data = spectralify(C)

	pts = (p,t)
	GaussianRandomField{typeof(cov),Spectral,typeof(pts)}(mean,cov,pts,data)
end

function spectralify(C)
    # compute eigenvalue decomposition
    F = eigfact(C)
    idx = sortperm(F[:values],rev=true)
    Λ = F[:values][idx]
    U = F[:vectors][:,idx]

    # if negative eigenvalues detected, remove them
    n = length(Λ)+1
    if Λ[end] < 0.
        found = false
        n = 0
        while !found
            n += 1
            if Λ[n] < 0
                found = true
            end
        end
		warn("negative eigenvalue $(Λ[n]) detected, Gaussian random field will be approximated (ignoring all negative eigenvalues)")
    end

    data = SpectralData(sqrt.(Λ[1:n-1]),U[:,1:n-1]) # note: store sqrt of eigenval for more efficient sampling
end

# returns the required dimension of the random points
randdim(grf::GaussianRandomField{C,Spectral} where {C<:CovarianceFunction}) = length(grf.data.eigenval) 

function _sample(grf::GaussianRandomField{C,Spectral,Tuple{T1,T2}} where {C,N,T1<:AbstractVector,T2<:AbstractVector}, xi)
    grf.mean + grf.cov.cov.σ*reshape(( grf.data.eigenfunc*diagm(grf.data.eigenval) )*xi,length.(grf.pts))
end

function _sample(grf::GaussianRandomField{C,Spectral,Tuple{T1,T2}} where {C,N,T1<:AbstractMatrix,T2<:AbstractMatrix}, xi)
	grf.mean + grf.cov.cov.σ* ( ( grf.data.eigenfunc*diagm(grf.data.eigenval) )*xi )
end
