# TODO jldoctest
## covariance_functions.jl : GRF covariance functions

## CovarianceStructure ##
abstract type CovarianceStructure{T} end

## CovarianceFunction ##
"""
`struct CovarianceFunction{d,T}`

Implements a covariance function of type `T` for a `d`-dimensional Gaussian random field.
"""
struct CovarianceFunction{d,T}
    cov::T
end

"""
`CovarianceFunction(d, cov)`

Create a covariance function in `d` dimensions for the covariance structure `cov`.

Examples:
```
julia> m = Matern(0.1,1.0)
Matérn covariance structure with correlation length λ = 0.1, smoothness ν = 1.0, marginal standard deviation σ = 1.0 and 2.0-norm

julia> c = CovarianceFunction(2,m)
2-dimensional covariance function with Matérn covariance structure with correlation length λ = 0.1, smoothness ν = 1.0, marginal standard deviation σ = 1.0 and 2.0-norm

```
"""
function CovarianceFunction(d::N where {N<:Integer},cov::T) where {T<:CovarianceStructure}
    d > 0 || throw(ArgumentError("dimension must be positive, got $(d)"))
    CovarianceFunction{d,T}(cov)
end

apply(cov::CovarianceFunction,x,y) = apply(cov.cov,x,y)

function apply(cov::CovarianceStructure{T}, x::Tuple, y::Tuple) where {T<:Real}
    p = cov.p
    D = zeros(T,prod(length.(x)),prod(length.(y)))
    for (j,idy) in enumerate(Base.product(y...))
    	for (i,idx) in enumerate(Base.product(x...))
            @inbounds D[i,j] = sum((idx.-idy).^p).^(1/p)
        end
    end
    return apply.(cov,abs.(D))
end

function apply(cov::CovarianceStructure{T}, x::Matrix{T}, y::Matrix{T}) where {T<:Real}
	p = cov.p
	D = zeros(T,size(x,2),size(y,2))
	for j in 1:size(y,2), i in 1:size(x,2)
		@inbounds D[i,j] = sum((x[:,i].-y[:,j]).^p).^(1/p)
	end

	apply.(cov,abs.(D))
end

show(io::IO, c::CovarianceFunction{d}) where {d} = print(io, "$(d)-dimensional covariance function with $(c.cov)")
