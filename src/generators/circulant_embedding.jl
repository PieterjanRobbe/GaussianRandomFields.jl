# TODO
# - replace min EV search in spec, kl by minimum
# - use plan_fft in sample
# - even/odd number of points
# - make new GaussianRnadomField()
#   that only accepts StepRangeLen, 1d, 2d, 3d
# - implement 1d; padding
# - probably implement apply as well?
#   or call it using first diff...
# - implement sample
# - implement 2d
# - add See also notes to KL, Spec, Chol

## circulant_embedding.jl : Gaussian random field generator using fft; only for uniformly spaced GRFs

"""
    CirculantEmbedding <: GaussianRandomFieldGenerator

A [`GaussiandRandomFieldGenerator`](@ref) that uses FFT to compute samples of the Gaussian random field.

# Examples
```jldoctest
julia> 

```
See also: [`Cholesky`](@ref), [`Spectral`](@ref), [`KarhunenLoeve`](@ref)
"""
struct CirculantEmbedding <: GaussianRandomFieldGenerator end

const CirculantGRF = GaussianRandomField{C,CirculantEmbedding} where {C}

function _GaussianRandomField(mean,cov::CovarianceFunction{1},method::CirculantEmbedding,pts...)
    c = apply.(cov.cov,norm.(pts[1]-minimum(pts[1])))
    c̃ = circulantify(c)
    Λ = real(ifft(c̃)) # TODO sort???
    Λ⁺ = zeros(size(Λ))
    Λ⁻ = zeros(size(Λ))
    for i in 1:length(Λ)
        if Λ[i] < 0
            Λ⁻[i] = -Λ[i]
        else
            Λ⁺[i] = Λ[i]
        end
    end
    Λ_max = maximum(Λ⁻)
    Λ_max > 0. && warn("negative eigenvalue $(real(-Λ_max)) detected, Gaussian random field will be approximated (ignoring all negative eigenvalues)")
    data = sqrt.(length(mean)*Λ⁺)
@show Λ⁺
    GaussianRandomField{typeof(cov),CirculantEmbedding,typeof(pts)}(mean,cov,pts,data)
end

circulantify(c) = vcat(c,flipdim(c,1)[2:end-1])

# returns the required dimension of the random points
randdim(grf::CirculantGRF) = (length(grf.data),2) 

# sample function for both Spectral() and KarhunenLoeve(n) type
function _sample(grf::CirculantGRF, xi)
    xi = xi*[1; 1im]
    n = length(grf.mean)
    grf.mean + std(grf.cov)*reshape(real( fft( grf.data.*xi )[1:n] )./sqrt(n),size(grf.mean))
end

show(io::IO,::CirculantEmbedding) = print(io,"circulant embedding")
