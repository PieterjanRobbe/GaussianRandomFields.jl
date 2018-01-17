## spectral.jl : Gaussian random field generator based on the spectral (eigenvalue) decomposition of the covariance matrix
 
"""
    Spectral <: GaussianRandomFieldGenerator

A [`GaussiandRandomFieldGenerator`](@ref) based on a spectral (eigenvalue) decomposition of the covariance matrix.

# Examples
```jldoctest
julia> m = Matern(0.1,1.0)
Matérn (λ=0.1, ν=1.0, σ=1.0, p=2.0)

julia> c = CovarianceFunction(2,m)
2d Matérn covariance function (λ=0.1, ν=1.0, σ=1.0, p=2.0)

julia> pts1 = 0:0.02:1; pts2 = 0:0.02:1 
0.0:0.02:1.0

julia> grf = GaussianRandomField(c,Spectral(),pts1,pts2)
Gaussian random field with 2d Matérn covariance function (λ=0.1, ν=1.0, σ=1.0, p=2.0) on a 51x51 structured grid, using a Spectral decomposition

julia> plot(grf) 
[...]

```
See also: [`Cholesky`](@ref), [`KarhunenLoeve`](@ref)
"""
struct Spectral <: GaussianRandomFieldGenerator end 

const SpectralGRF = GaussianRandomField{C,Spectral} where {C}

# container type for eigenvalues and eigenfunctions
mutable struct SpectralData{X,Y}
    eigenval::X
    eigenfunc::Y
end

function _GaussianRandomField(mean,cov,method::Spectral,pts...)
    C = apply(cov,pts,pts)

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

    # store eigenvalues and eigenfunctions
    data = SpectralData(sqrt.(Λ[1:n-1]),U[:,1:n-1]) # note: store sqrt of eigenval for more efficient sampling

    GaussianRandomField{typeof(cov),Spectral,typeof(pts)}(mean,cov,pts,data)
end

# returns the required dimension of the random points
randdim(grf::SpectralGRF) = length(grf.data.eigenval) 

# see KarhunenLoeve.jl for sample function

show(io::IO,::Spectral) = print(io,"Spectral decomposition")
