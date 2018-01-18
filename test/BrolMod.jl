module BrolMod

export Triangular

using Reexport
@reexport using GaussianRandomFields

struct Triangular{T} <: CovarianceStructure{T}
    λ::T
    σ::T
    p::T
end

apply(t::Triangular{T},x::T) where {T<:Real} = x < t.λ ? 1-x/t.λ : zero(T)

show(io::IO, t::Triangular) = print(io, "Triangular (λ=$(t.λ), σ=$(t.σ), p=$(t.p))")


end
