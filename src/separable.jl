# separable.jl : separable Gaussian random fields

## SeparableCovarianceFunction ##
struct SeparableCovarianceFunction{d,V} <: AbstractCovarianceFunction{d}
    cov::V
end

"""
	SeparableCovarianceFunction(cov...)

	Create a separable covariance function in `length(cov)` dimensions for the covariance structures `cov`. Usefull for defining anisotropic covariance functions, or if the usual KL expansion is too expensive.

# Examples
```
julia> e = Exponential(0.1)
exponential (λ=0.1, σ=1.0, p=2.0)

julia> m = Matern(0.01,1.0)
Matérn (λ=0.01, ν=1.0, σ=1.0, p=2.0)

julia> c = SeparableCovarianceFunction([e,m])
2d separable covariance function [ exponential (λ=0.1, σ=1.0, p=2.0), Matérn (λ=0.01, ν=1.0, σ=1.0, p=2.0) ]

```
See also: [`CovarianceFunction`](@ref), [`KarhunenLoeve`](@ref) 
"""
SeparableCovarianceFunction(cov::Vector{<:CovarianceStructure}) = SeparableCovarianceFunction{length(cov),typeof(cov)}(cov)
SeparableCovarianceFunction(cov::CovarianceStructure...) = SeparableCovarianceFunction([cov...])

Base.eltype(cov::SeparableCovarianceFunction) = mapreduce(eltype, promote_type, cov.cov)

function GaussianRandomField(mean::Array{<:Real}, cov::SeparableCovarianceFunction{d}, kl::KarhunenLoeve{n}, pts::Vararg{AbstractVector,d}; kwargs...) where {d,n}
    size(mean) == length.(pts) || throw(DimensionMismatch("size of the mean does not correspond to the dimension of the points"))

    # generate the 1d grf's
    data = SpectralData[]
    for i in 1:length(cov.cov)
        cov_ = CovarianceFunction(1,cov.cov[i])
		if ( typeof(cov.cov[i]) <: Exponential && cov.cov[i].p == 1.)
			push!(data,compute_analytic(cov_,n,pts[i]))
		else
        	push!(data,_GaussianRandomField(mean,cov_,kl,pts[i];kwargs...).data)
		end
    end
    eigenval = [data[i].eigenval for i in 1:d]

    # determine n-d eigenvalues
    p = Base.product([data[i].eigenval for i in 1:d]...)
    m = map(prod,p)
    idx = sortperm(m[:],rev=true)
    pidx = Base.product(broadcast(:,1,length.(eigenval))...)
    alldata = (collect(pidx)[idx], data)

	GaussianRandomField{typeof(kl),typeof(cov),typeof(pts),typeof(mean),typeof(alldata)}(mean,cov,pts,alldata)
end

# zero-mean GRF
GaussianRandomField(cov::SeparableCovarianceFunction{d,T} where {T},kl,pts::V...;kwargs...) where {d,V<:AbstractVector} = GaussianRandomField(zeros(eltype(pts[1]),length.(pts)...),cov,kl,pts...;kwargs...)

# constant mean GRF
GaussianRandomField(mean::R where {R<:Real},cov::SeparableCovarianceFunction{d,T} where {T},kl,pts::V...;kwargs...) where {d,V<:AbstractVector} = GaussianRandomField(mean*ones(eltype(pts[1]),length.(pts)...),cov,kl,pts...;kwargs...)

# return number of dimension
ndims(::SeparableCovarianceFunction{d}) where {d} = d

# sample function
function sample(grf::GaussianRandomField{KarhunenLoeve{n},<:SeparableCovarianceFunction}; xi::Vector{<:Real} = randn(n)) where n
    length(xi) == n || throw(DimensionMismatch("length of random points vector must be equal to $(n)"))
    order, data = grf.data
	x = zeros(eltype(xi), length(grf.mean))
	d = length(grf.cov.cov)
	for i in 1:n
        ev = prod([data[j].eigenval[order[i][j]] for j = 1:d])
        ef = kron([data[j].eigenfunc[:,order[i][j]] for j = 1:d]...)
		x += xi[i]*ev.*ef
	end
	grf.mean + prod([grf.cov.cov[i].σ for i = 1:d])*reshape(x, size(grf.mean))
end

function sample(grf::GaussianRandomField{KarhunenLoeve{n},<:SeparableCovarianceFunction{1}}; xi::Vector{<:Real} = randn(n)) where n
    length(xi) == n || throw(DimensionMismatch("length of random points vector must be equal to $(n)"))
    order, data = grf.data
	x = data[1].eigenfunc * (data[1].eigenval .* xi)
	grf.mean + grf.cov.cov[1].σ*x
end

function show(io::IO, s::SeparableCovarianceFunction{d}) where {d}
	str = "$(d)d separable covariance function [ $(s.cov[1])"
	for i in 2:length(s.cov)
		str *= ", $(s.cov[i])"
    end
	str *= " ]"
	print(io, str) 
end
