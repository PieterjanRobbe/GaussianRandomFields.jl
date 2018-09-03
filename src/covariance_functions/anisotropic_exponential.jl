## anisotropic_exponential.jl : implementation of anisotropic exponential covariance function

## AnisotropicExponential ##
struct AnisotropicExponential{T,M<:Matrix{<:Real}} <: AnisotropicCovarianceStructure{T}
    A::M
    σ::T

    function AnisotropicExponential{T,M}(A::M, σ::T) where {T,M}
        isposdef(A) || throw(DomainError(A, "anisotropy matrix A must be positive definite"))
        σ > 0 || throw(DomainError(σ, "marginal standard deviation σ of exponential covariance cannot be negative or zero"))

        new{T,M}(A, σ)
    end
end

"""
    AnisotropicExponential(A, σ=1)

Create an anisotropic exponential covariance structure with anisotropy matrix A and (optional) marginal standard deviation `σ`.

# Examples
```jldoctest
julia> A = [1 0.5; 0.5 1]
2×2 Array{Float64,2}:
 1.0  0.5
 1.0  0.5

julia> a1 = AnisotropicExponential(A)
anisotropic exponential (A=[1.0 0.5; 0.5 1.0], σ=1.0)

```
"""
function AnisotropicExponential(A::Matrix{<:Real}; σ::Real=1.0)
	T = promote_type(eltype(A),typeof(σ))
	AnisotropicExponential{T,typeof(A)}(A, convert(T, σ))
end

# evaluate exponential covariance
apply(a::AnisotropicExponential, x::Vector{<:Real}) = exp(-dot(x, a.A * x))

# short name
shortname(::AnisotropicExponential) = "anisotropic exponential"
