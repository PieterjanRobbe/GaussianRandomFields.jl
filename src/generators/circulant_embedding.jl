# TODO
# - probably implement apply as well?
#   or call it using first diff...
# - implement 2d
# - update GaussianRandomField docs
# - update tut
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
struct CirculantEmbedding <: EquidistantGaussianRandomFieldGenerator end

const CirculantGRF = GaussianRandomField{C,CirculantEmbedding} where {C}

function _GaussianRandomField(mean,cov::CovarianceFunction{1},method::CirculantEmbedding,pts;padding=1)

	# add ghost points by padding
	padded_pts = pad(pts,padding)

	# compute eigenvalues of circulant matrix
    c = apply.(cov.cov,norm.(padded_pts-minimum(padded_pts)))
	Λ = irfft(c,2*length(c)-1)
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
	Λ_max > 0. && ( warn("negative eigenvalue $(real(-Λ_max)) detected, Gaussian random field will be approximated (ignoring all negative eigenvalues)"); warn("increase padding if possible") )

	# optimize
    Σ = sqrt.(length(mean)*Λ⁺)
	xi = randn(2*length(c)-1,2)*[1; 1im]
	P = plan_fft(Σ.*xi,1,flags=FFTW.MEASURE)

	GaussianRandomField{typeof(cov),CirculantEmbedding,typeof(tuple(pts))}(mean,cov,tuple(pts),(Σ,P))
end

circulantify(c) = vcat(c,flipdim(c,1)[2:end-1])

function pad(x,n)
	x0 = x[1]
	xn = x[end]
	dx = x[2]-x[1]
	x0:dx:n*xn
end

# returns the required dimension of the random points
randdim(grf::CirculantGRF) = (length(grf.data[1]),2) 

# sample function
function _sample(grf::CirculantGRF, xi)
    xi = xi*[1; 1im]
    n = length(grf.mean)
	grf.mean + std(grf.cov)*reshape(real( ( grf.data[2]*( grf.data[1].*xi ) )[1:n] )./sqrt(n),size(grf.mean))
end

show(io::IO,::CirculantEmbedding) = print(io,"circulant embedding")
