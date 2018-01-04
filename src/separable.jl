# separable.jl : implementation of separable kernels

## SeparableCovarianceFunction ##
"""
`struct SeparableCovarianceFunction{d,V}`

Implements a separable covariance function of types `V` for a `d`-dimensional Gaussian random field.
Usefull for defining anisotropic random fields.
"""
struct SeparableCovarianceFunction{d,V}
    cov::V
end

"""
`SeparableCovarianceFunction(d, cov)`

Create a separable covariance function in `d` dimensions for the covariance structures `cov`. Usefull for defining anisotropic covariance functions.

Examples:
```
julia> m1 = Matern(0.1,1.0,2)
Matérn covariance structure with correlation length λ = 0.1, smoothness ν = 1.0 and 2.0-norm

julia> m2 = Matern(0.01,1.0,2)
Matérn covariance structure with correlation length λ = 0.01, smoothness ν = 1.0 and 2.0-norm

julia> c = SeparableCovarianceFunction(m1,m2)
2-dimensional separable covariance function with
- Matérn covariance structure with correlation length λ = 0.1, smoothness ν = 1.0 and 2.0-norm  
- Matérn covariance structure with correlation length λ = 0.01, smoothness ν = 1.0 and 2.0-norm

```
"""
SeparableCovarianceFunction(cov::Vector{T}) where {T<:CovarianceStructure} = SeparableCovarianceFunction{length(cov),Vector{T}}(cov)
SeparableCovarianceFunction(cov...) = SeparableCovarianceFunction([cov...])

function show(io::IO, s::SeparableCovarianceFunction{d}) where {d}
    str = "$(d)-dimensional separable covariance function with \n"
    for elem in s.cov
        str *= "  - $(elem)\n"
    end
    print(io, str) 
end

function GaussianRandomField(mean::Array{T} where {T<:AbstractFloat},cov::SeparableCovarianceFunction{d,T} where {T},k::KarhunenLoeve{n},pts::V...;kwargs...) where {d,n,V<:AbstractVector}
    all(size(mean).==length.(pts)) || throw(DimensionMismatch("size of the mean does not correspond to the dimension of the points"))
    length(pts) == d || throw(DimensionMismatch("number of point ranges must be equal to the dimension of the covariance function"))

    # generate the 1d grf's
    data = SpectralData[]
    for i in 1:length(cov.cov)
        cov_ = CovarianceFunction(1,cov.cov[i])
        push!(data,_GaussianRandomField(mean,cov_,k,pts[i];kwargs...).data)
    end
    eigenval = [data[i].eigenval for i in 1:d]

    # determine n-d eigenvalues
    p = Base.product([data[i].eigenval for i in 1:d]...)
    m = map(prod,p)
    idx = sortperm(m[:],rev=true)
    pidx = Base.product(range.(1,length.(eigenval))...)

    GaussianRandomField{typeof(cov),KarhunenLoeve{n}}(mean,cov,pts,(collect(pidx)[idx],data))
end

GaussianRandomField(cov::SeparableCovarianceFunction{d,Vector{N}} where {N<:CovarianceStructure{T}},k::KarhunenLoeve{n} where {n},pts::V...;kwargs...) where {d,T,V<:AbstractVector} = GaussianRandomField(zeros(T,length.(pts)...),cov,k,pts...;kwargs...)

GaussianRandomField(mean::R where {R<:Real},cov::SeparableCovarianceFunction{d,Vector{N}} where {N<:CovarianceStructure{T}},k::KarhunenLoeve{n} where {n},pts::V...;kwargs...) where {d,T,V<:AbstractVector} = GaussianRandomField(mean*ones(T,length.(pts)...),cov,k,pts...;kwargs...)

σ(grf::GaussianRandomField{C,KarhunenLoeve{n}} where {C<:SeparableCovarianceFunction{d},n}) where {d} = prod([grf.cov.cov[i].σ for i = 1:d])

function eigenvalues(grf::GaussianRandomField{C,KarhunenLoeve{n}} where {C<:SeparableCovarianceFunction{d},n}) where {d}
    (order,data) = grf.data
    ev = zeros(randdim(grf))
    for n in 1:length(ev)
        ev[n] = prod([data[i].eigenval[order[n][i]] for i = 1:d])
    end
    return ev
end

function eigenfunctions(grf::GaussianRandomField{C,KarhunenLoeve{n}} where {C<:SeparableCovarianceFunction{d},n}) where {d}
    (order,data) = grf.data
    if d == 1
        return data[1].eigenfunc
    else
        ef = zeros(prod(length.(grf.pts)),randdim(grf))
        for n in 1:size(ef,2)
            ef[:,n] = kron([data[i].eigenfunc[:,order[n][i]] for i = 1:d]...)
        end
        return ef
    end
end
