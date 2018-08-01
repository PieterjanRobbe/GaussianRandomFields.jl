## circulant_embedding.jl : Gaussian random field generator using fft; only for uniformly spaced GRFs

"""
    CirculantEmbedding <: GaussianRandomFieldGenerator

A [`GaussiandRandomFieldGenerator`](@ref) that uses FFT to compute samples of the Gaussian random field. Circulant embedding can only be applied if the points are specified on a structured grid.

# Examples
```jldoctest
julia> m = Matern(0.1,1.0)
Matérn (λ=0.1, ν=1.0, σ=1.0, p=2.0)

julia> c = CovarianceFunction(2,m)
2d Matérn covariance function (λ=0.1, ν=1.0, σ=1.0, p=2.0)

julia> pts1 = 0:0.02:1; pts2 = 0:0.02:1 
0.0:0.02:1.0

julia> grf = GaussianRandomField(c,CirculantEmbedding(),pts1,pts2)
Gaussian random field with 2d Matérn covariance function (λ=0.1, ν=1.0, σ=1.0, p=2.0) on a 51x51 structured grid, using a circulant embedding

julia> contourf(grf)
[...]

julia> plot_eigenvalues(grf)
[...]

```
With appropriate ordering, the covariance matrix of a Gaussian random field is a (nested block) Toeplitz matrix. This matrix can be embedded into a larger (nested block) circulant matrix, whose eigenvalues can be rapidly computed using FFT. A difficulty here is that although the covariance matrix is positive semi-definite, its circulant extension in general is not. As a remedy, one can add so-called *ghost points* outside the domain of interest using the optional flag `padding`.
```jldoctest
julia> m = Matern(1,1)
Matérn (λ=1.0, ν=1.0, σ=1.0, p=2.0)

julia> c = CovarianceFunction(1,m)
1d Matérn covariance function (λ=1.0, ν=1.0, σ=1.0, p=2.0)

julia> pts = range(0,stop = 1,length = 256)
0.0:0.00392156862745098:1.0

julia> grf = GaussianRandomField(c,CirculantEmbedding(),pts)
WARNING: negative eigenvalue -0.001465931113950698 detected, Gaussian random field will be approximated (ignoring all negative eigenvalues)
WARNING: increase padding if possible
Gaussian random field with 1d Matérn covariance function (λ=1.0, ν=1.0, σ=1.0, p=2.0) on a 256-point structured grid, using a circulant embedding

julia> grf = GaussianRandomField(c,CirculantEmbedding(),pts,padding=5)
Gaussian random field with 1d Matérn covariance function (λ=1.0, ν=1.0, σ=1.0, p=2.0) on a 256-point structured grid, using a circulant embedding

```
For faster sampling (but slower initialization), use the optional argument `measure` (default=true).

See also: [`Cholesky`](@ref), [`Spectral`](@ref), [`KarhunenLoeve`](@ref)
"""
struct CirculantEmbedding <: GaussianRandomFieldGenerator end

const CirculantGRF = GaussianRandomField{C,CirculantEmbedding} where {C}

function _GaussianRandomField(mean,cov::CovarianceFunction{d},method::CirculantEmbedding,pts...;padding=1,measure=true) where {d}

    # add ghost points by padding
    padded_pts = pad.(pts,padding)
    n = length.(padded_pts)
	n2 = n[2:d]
    padded_pts = normalize.(padded_pts)
	padded_pts = (padded_pts[1],mirror.(padded_pts[2:d])...) # do not mirror dimension 1

    # compute eigenvalues of circulant matrix
	c = zeros(n[1],(2.0*n2.-1)...)
    for (i,idx) in enumerate(Base.product(padded_pts...))
        c[i] = apply(cov.cov,collect(idx))
    end
	c̃ = zeros(n[1],(2.0 * n2)...)
	c̃[:,(range.(2,2.0 * n2.-1)...)...] = c
	c̃ = fftshift(c̃,2:d)
    Λ = irfft(c̃,2*size(c̃,1)-1)
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
    Λ_max > 0. && ( warn("negative eigenvalue $(-Λ_max) detected, Gaussian random field will be approximated (ignoring all negative eigenvalues)"); warn("increase padding if possible") )

    # optimize
    Σ = sqrt.(Λ⁺)
    P = measure ? plan_fft(Σ) : plan_fft(Σ,flags=FFTW.MEASURE)

    GaussianRandomField{typeof(cov),CirculantEmbedding,typeof(pts)}(mean,cov,pts,(Σ,P))
end

normalize(pts::R where {R<:AbstractRange}) = pts-pts[1]
mirror(pts::R where {R<:AbstractRange}) = -pts[end]:(pts[2]-pts[1]):pts[end]

function pad(x,n)
    x0 = x[1]
    xn = x[end]
    dx = x[2]-x[1]
    x0:dx:n*xn
end

# returns the required dimension of the random points
randdim(grf::CirculantGRF) = length(grf.data[1]) 

# sample function
function _sample(grf::CirculantGRF, xi)
    v = grf.data[1]
    y = v.*reshape(xi,size(v))
    P = grf.data[2]
    w = P*y # fft
    w = real(w) + imag(w)
    n = length.(grf.pts)
    z = w[range.(1,n)...] # select appropriate elements

    grf.mean + std(grf.cov)*z
end

show(io::IO,::CirculantEmbedding) = print(io,"circulant embedding")
