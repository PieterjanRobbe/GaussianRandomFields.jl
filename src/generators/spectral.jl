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
This is also useful when computing Gaussian random fields on a Finite Element mesh using a truncated KL expansion. Here's an example that computes the first 10 eigenfunctions on an L-shaped domain.

```jldoctest
julia> p = readdlm(Pkg.dir("GaussianRandomFields")*"/data/Lshape.p");

julia> t = readdlm(Pkg.dir("GaussianRandomFields")*"/data/Lshape.t",Int64);

julia> grf = grf = GaussianRandomField(CovarianceFunction(2,Matern(0.2,1.0)),Spectral(),p,t,n=10)
Gaussian random field with 2d Matérn covariance function (λ=0.2, ν=1.0, σ=1.0, p=2.0) on a mesh with 998 points and 1861 elements, using a spectral decomposition

julia> tricontourf(p[:,1],p[:,2],grf.data.eigenfunc[:,1],triangles=t-1,cmap=get_cmap("viridis"))
[...]

```
See also: [`Cholesky`](@ref), [`KarhunenLoeve`](@ref), [`CirculantEmbedding`](@ref)
"""
struct Spectral <: GaussianRandomFieldGenerator end 

const SpectralGRF = GaussianRandomField{C,Spectral} where {C}

# container type for eigenvalues and eigenfunctions
mutable struct SpectralData{X,Y}
	eigenval::X
	eigenfunc::Y
end

function _GaussianRandomField(mean,cov,method::Spectral,pts...;n::N=0) where {N<:Integer}
	C = apply(cov,pts,pts)

	# compute eigenvalue decomposition
	if n == 0
		F = eigfact(C)
		idx = sortperm(F[:values],rev=true)
		Λ = F[:values][idx]
		U = F[:vectors][:,idx]
	else # thanks to #4
		(eigenval,eigenfunc) = eigs(C,nev=n,ritzvec=true,which=:LM)
		idx = sortperm(eigenval,rev=true)
		Λ = eigenval[idx]
		U = eigenfunc[:,idx]
	end

	# if negative eigenvalues detected, remove them
	n = find_last_positive(Λ)
	n == length(Λ) || warn("negative eigenvalue $(Λ[n+1]) detected, Gaussian random field will be approximated (ignoring all negative eigenvalues)")

	# store eigenvalues and eigenfunctions
	data = SpectralData(sqrt.(Λ[1:n]),U[:,1:n]) # note: store sqrt of eigenval for more efficient sampling

	GaussianRandomField{typeof(cov),Spectral,typeof(pts)}(mean,cov,pts,data)
end

# find last positive entry in vector, assumes sorted from high to low
function find_last_positive(x::Vector{T} where {T})
	found = false
	n = 0
	while !found
		n += 1
		if ( n > length(x) ) || ( x[n] < 0 )
			found = true
		end
	end
	return n-1
end

# returns the required dimension of the random points
randdim(grf::SpectralGRF) = length(grf.data.eigenval) 

# see KarhunenLoeve.jl for sample function

show(io::IO,::Spectral) = print(io,"spectral decomposition")
