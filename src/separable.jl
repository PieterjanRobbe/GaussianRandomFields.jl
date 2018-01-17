# separable.jl : separable Gaussian random fields

## SeparableCovarianceFunction ##
struct SeparableCovarianceFunction{d,V}
    cov::V
end

"""
	SeparableCovarianceFunction(d, cov)

Create a separable covariance function in `d` dimensions for the covariance structures `cov`. Usefull for defining anisotropic covariance functions.

# Examples
```
julia> e = Exponential(0.1)
exponential (λ=0.1, σ=1.0, p=2.0)

julia> m = Matern(0.01,1.0)
Matérn (λ=0.01, ν=1.0, σ=1.0, p=2.0)

julia> c = SeparableCovarianceFunction(e,m)
2d separable covariance function [ exponential (λ=0.1, σ=1.0, p=2.0), Matérn (λ=0.01, ν=1.0, σ=1.0, p=2.0) ]

```
See also: [`CovarianceFunction`](@ref) 
"""
SeparableCovarianceFunction(cov::Vector{T}) where {T<:CovarianceStructure} = length(cov) > 1 ? SeparableCovarianceFunction{length(cov),Vector{T}}(cov) : throw(ArgumentError("cannot generate a separable covariance function in 1d, use CovarianceFunction instead"))
SeparableCovarianceFunction(cov...) = SeparableCovarianceFunction([cov...])

function GaussianRandomField(mean::Array{T} where {T<:Real},cov::SeparableCovarianceFunction{d,T} where {T},kl::KarhunenLoeve{n},pts::V...;kwargs...) where {d,n,V<:AbstractVector}
    all(size(mean).==length.(pts)) || throw(DimensionMismatch("size of the mean does not correspond to the dimension of the points"))
    length(pts) == d || throw(DimensionMismatch("number of point ranges must be equal to the dimension of the covariance function"))

    # generate the 1d grf's
    data = SpectralData[]
    for i in 1:length(cov.cov)
        cov_ = CovarianceFunction(1,cov.cov[i])
        push!(data,_GaussianRandomField(mean,cov_,kl,pts[i];kwargs...).data)
    end
    eigenval = [data[i].eigenval for i in 1:d]

    # determine n-d eigenvalues
    p = Base.product([data[i].eigenval for i in 1:d]...)
    m = map(prod,p)
    idx = sortperm(m[:],rev=true)
    pidx = Base.product(range.(1,length.(eigenval))...)

	GaussianRandomField{typeof(cov),typeof(kl),typeof(pts)}(mean,cov,pts,(collect(pidx)[idx],data))
end

# zero-mean GRF
GaussianRandomField(cov::SeparableCovarianceFunction{d,T} where {T},kl,pts::V...;kwargs...) where {d,V<:AbstractVector} = GaussianRandomField(zeros(eltype(pts[1]),length.(pts)...),cov,kl,pts...;kwargs...)

# constant mean GRF
GaussianRandomField(mean::R where {R<:Real},cov::SeparableCovarianceFunction{d,T} where {T},kl,pts::V...;kwargs...) where {d,V<:AbstractVector} = GaussianRandomField(mean*ones(eltype(pts[1]),length.(pts)...),cov,kl,pts...;kwargs...)

# sample function
function sample(grf::GaussianRandomField{C,KarhunenLoeve{n}} where {C<:SeparableCovarianceFunction}; xi::Vector{T} = randn(n)) where {n,T<:Real}
    length(xi) == n || throw(DimensionMismatch("length of random points vector must be equal to $(n)"))
    (order,data) = grf.data
	x = zeros(T,prod(size(grf.mean)))
	d = length(grf.cov.cov)
	for i in 1:n
        ev = prod([data[j].eigenval[order[i][j]] for j = 1:d])
        ef = kron([data[j].eigenfunc[:,order[i][j]] for j = 1:d]...)
		x += xi[i]*ev.*ef
	end
	grf.mean + prod([grf.cov.cov[i].σ for i = 1:d])*reshape(x, size(grf.mean))
end

function show(io::IO, s::SeparableCovarianceFunction{d}) where {d}
	str = "$(d)d separable covariance function [ $(s.cov[1])"
	for i in 2:length(s.cov)
		str *= ", $(s.cov[i])"
    end
	str *= " ]"
	print(io, str) 
end
