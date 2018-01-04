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

    # evaluate covariance matrix
    C = apply(cov,pts,pts)

    # compute eigenvalue decomposition
    F = eigfact(C)
    idx = sortperm(F[:values],rev=true)
    Λ = F[:values][idx]
    U = F[:vectors][:,idx]

    # if negstive eigenvalues detected, remove them
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
        warn("negative eigenvalue $(Λ[n]) detected, ignoring all negative eigenvalues")
    end
    data = SpectralData(sqrt.(Λ[1:n-1]),U[:,1:n-1]) # note: store sqrt of eigenval
    GaussianRandomField{typeof(cov),Spectral}(mean,cov,pts,data)
end

# returns the required dimension of the random points
randdim(grf::GaussianRandomField{C,Spectral} where {C<:CovarianceFunction}) = length(grf.data.eigenval) 

function _sample(grf::GaussianRandomField{C,Spectral} where {C}, xi)
    grf.mean + grf.cov.cov.σ*reshape(( grf.data.eigenfunc*diagm(grf.data.eigenval) )*xi,length.(grf.pts))
end
