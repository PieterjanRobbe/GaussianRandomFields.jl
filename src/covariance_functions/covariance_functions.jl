## covariance_functions.jl : utilities for Gaussian random field covariance functions

## CovarianceStructure ##
abstract type CovarianceStructure{T} end
abstract type IsotropicCovarianceStructure{T} <: CovarianceStructure{T} end
abstract type AnisotropicCovarianceStructure{T} <: CovarianceStructure{T} end

## CovarianceFunction ##
struct CovarianceFunction{d,T}
    cov::T
end

"""
	CovarianceFunction(d, cov)

Create a covariance function in `d` dimensions with covariance structure `cov`.

# Examples
```jldoctest
julia> m = Matern(0.1,1.0)
Matérn (λ=0.1, ν=1.0, σ=1.0, p=2.0)

julia> c = CovarianceFunction(2,m)
2d Matérn covariance function (λ=0.1, ν=1.0, σ=1.0, p=2.0)

```
"""
function CovarianceFunction(d::N where {N<:Integer},cov::T) where {T<:CovarianceStructure}
    d > 0 || throw(ArgumentError("dimension must be positive, got $(d)"))
    CovarianceFunction{d,T}(cov)
end

# return standard deviation of the Gaussian random field
std(cov::CovarianceFunction) = cov.cov.σ

# return number of dimension
ndims(::CovarianceFunction{d}) where {d} = d

# evaluate the covariance function
apply(cov::CovarianceFunction,x,y) = apply(cov.cov,x,y)

# apply for isotropic random fields
apply(cov::IsotropicCovarianceStructure,dx::Vector{T} where {T<:Real}) = apply(cov,sum(abs.(dx).^cov.p).^(1/cov.p))

# evaluate when pts is given as a kron product of 1d points
function apply(cov::CovarianceStructure{T}, x::Tuple, y::Tuple) where {T<:Real}
    C = zeros(T,prod(length.(x)),prod(length.(y)))
    if  size(C,1) == size(C,2)
        return apply_symmetric(cov,x,y,C)
    else
        return apply_non_symmetric(cov,x,y,C)
    end
end

function apply_symmetric(cov::CovarianceStructure{T}, x::Tuple, y::Tuple, C::Matrix{T}) where {T<:Real}
    for (j,idy) in enumerate(Base.product(y...))
        for (i,idx) in enumerate(Base.product(x...))
            if i <= j
	        @inbounds C[i,j] = apply(cov,collect(idx.-idy))
            end
        end
    end
    Symmetric(C,:U)
end

function apply_non_symmetric(cov::CovarianceStructure{T}, x::Tuple, y::Tuple, C::Matrix{T}) where {T<:Real}
    for (j,idy) in enumerate(Base.product(y...))
        for (i,idx) in enumerate(Base.product(x...))
	    @inbounds C[i,j] = apply(cov,collect(idx.-idy))
        end
    end
    C
end

# evaluate when pts is given as a Finite Element mesh
function apply(cov::CovarianceStructure{T}, tx::Tuple{T1,T2}, ty::Tuple{T1,T2}) where {T<:Real,T1<:AbstractMatrix,T2<:AbstractMatrix}
    x = first(tx) # select FE nodes
    y = first(ty)
    C = zeros(T,size(x,2),size(y,2))
    for j in 1:size(y,2), i in 1:size(x,2)
        if i <= j
            @inbounds C[i,j] = apply(x[:,i].-y[:,j])
        end
    end
    return Symmetric(C,:U)
end

# evaluate for KL eigenfunctions
function apply(cov::CovarianceStructure{T}, tx::Tuple{T1,T2}, y::Tuple) where {T<:Real,T1<:AbstractMatrix,T2<:AbstractMatrix}
    x = first(tx) # select FE nodes
    C = zeros(T,size(x,2),prod(length.(y)))
    for (j,idy) in enumerate(Base.product(y...))
        for i in 1:size(x,2)
            @inbounds C[i,j] = apply(x[:,i].-idy)
        end
    end
    CC
end

function show(io::IO, c::CovarianceFunction{d}) where {d}
    str = split(string(c.cov)," ")
    print(io, "$(d)d $(str[1]) covariance function $(join(str[2:end]," "))")
end
