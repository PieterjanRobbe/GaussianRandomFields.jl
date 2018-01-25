## anisotropic_exponential.jl : implementation of anisotropic exponential covariance function

## AnisotropicExponential ##
struct AnisotropicExponential{T,M} <: AnisotropicCovarianceStructure{T}
    A::M
    σ::T
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
function AnisotropicExponential(A::Matrix{T} where {T<:Real}; σ=1.0::T where {T<:Real}) 
	isposdef(A) || throw(ArgumentError("anisotropy matrix A must be positive definite"))
    σ > 0 || throw(ArgumentError("marginal standard deviation σ of exponential covariance cannot be negative or zero"))
	T = promote_type(eltype(A),typeof(σ))
	AnisotropicExponential{T,typeof(A)}(A,convert(T,σ)) 
end

# evaluate exponential covariance
function apply(a::AnisotropicExponential,x::Vector{T}) where {T<:Real}
    exp(-x'*a.A*x)
end

show(io::IO, a::AnisotropicExponential) = print(io, "anisotropic exponential (A=$(a.A), σ=$(a.σ))")
