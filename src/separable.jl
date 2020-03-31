# separable.jl : separable Gaussian random fields

## SeparableCovarianceFunction ##
struct SeparableCovarianceFunction{d,V} <: AbstractCovarianceFunction{d}
    cov::V
end

"""
    SeparableCovarianceFunction(cov...)

Separable covariance function in `length(cov)` dimensions for the covariance structures `cov`. Usefull for defining anisotropic covariance functions, or if the non-seperable `KarhunenLoeve` expansion is too expensive.

# Examples
```
julia> SeparableCovarianceFunction(Exponential(0.1), Matern(0.01, 1.0))
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
    data = ntuple(length(cov.cov)) do i
        cov_ = CovarianceFunction(1, cov.cov[i])
        if isa(cov.cov[i], Exponential) && isone(cov.cov[i].p)
            compute_analytic(cov_,n,pts[i])
        else
            _GaussianRandomField(mean,cov_,kl,pts[i];kwargs...).data
        end
    end

    # determine n-d eigenvalues
    m = map(prod, Iterators.product((data[i].eigenval for i in 1:d)...))
    idx = sortperm(vec(m), rev=true)
    pidx = Iterators.product((1:length(data[i].eigenval) for i in 1:d)...)

    alldata = (collect(pidx)[idx], data)

    GaussianRandomField{typeof(kl),typeof(cov),typeof(pts),typeof(mean),typeof(alldata)}(mean,cov,pts,alldata)
end

# zero-mean GRF
GaussianRandomField(cov::SeparableCovarianceFunction{d}, kl, pts::Vararg{AbstractVector,d}; kwargs...) where d =
GaussianRandomField(zeros(eltype(cov), length.(pts)), cov, kl, pts...; kwargs...)

# constant mean GRF
GaussianRandomField(mean::Real, cov::SeparableCovarianceFunction{d}, kl, pts::Vararg{AbstractVector,d}; kwargs...) where d =
GaussianRandomField(fill(convert(eltype(cov), mean), length.(pts)), cov, kl, pts...; kwargs...)

# sample function
function sample(grf::GaussianRandomField{KarhunenLoeve{n},<:SeparableCovarianceFunction}; xi::Vector{<:Real} = randn(n)) where n
    length(xi) == n || throw(DimensionMismatch("length of random points vector must be equal to $(n)"))
    order, data = grf.data
    x = zeros(eltype(xi), length(grf.mean))
    d = length(grf.cov.cov)
    for i in 1:n
        orderi = order[i]
        ev = prod(data[j].eigenval[orderi[j]] for j = 1:d)
        ef = kron((data[j].eigenfunc[:,orderi[j]] for j = 1:d)...)
        x .+= (xi[i] * ev) .* ef
    end
    grf.mean + prod(grf.cov.cov[i].σ for i = 1:d) * reshape(x, size(grf.mean))
end

function sample(grf::GaussianRandomField{KarhunenLoeve{n},<:SeparableCovarianceFunction{1}}; xi::Vector{<:Real} = randn(n)) where n
    length(xi) == n || throw(DimensionMismatch("length of random points vector must be equal to $(n)"))
    order, data = grf.data
    x = data[1].eigenfunc * (data[1].eigenval .* xi)
    grf.mean + grf.cov.cov[1].σ*x
end

function Base.show(io::IO, s::SeparableCovarianceFunction)
    print(io, ndims(s), "d separable covariance function [ ", s.cov[1])
    for i in 2:length(s.cov)
        print(io, ", ", s.cov[i])
    end
    print(io, " ]")
end
