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

function _GaussianRandomField(mean, cov::CovarianceFunction{d}, method::CirculantEmbedding, pts::Vararg{AbstractRange,d}; padding::Int=1, measure::Bool=true) where {d}
    # add ghost points by padding but do not mirror dimension 1
    padded_pts = ntuple(d) do i
        i == 1 ? shift_extend(pts[i]; n=padding) : mirror_shift_extend(pts[i]; n=padding)
    end
    n = length.(padded_pts)

    # compute eigenvalues of circulant matrix
    c = zeros(n)
    @inbounds for (i, idx) in enumerate(Iterators.product(padded_pts...))
        c[i] = apply(cov.cov,collect(idx))
    end

    n2 = ntuple(i -> n[i + 1] + 1, d - 1)
    c̃ = zeros(n[1], n2...)
    c̃[:, broadcast(:, 2, n2)...] = c
    c̃ = fftshift(c̃, 2:d)
    Λ = irfft(c̃, 2*size(c̃,1)-1)

    # store sqrt of eigenvalues for more efficient sampling
    n₋, λ₋ = 0, 0.0
    @inbounds for i in eachindex(Λ)
        λ = Λ[i]
        if λ < 0
            # reset negative eigenvalues to zero
            Λ[i] = 0

            # update statistics about negative eigenvalues
            n₋ += 1
            λ < λ₋ && (λ₋ = λ)
        else
            Λ[i] = sqrt(λ)
        end
    end
    λ₋ < 0 && @warn begin
        "$n₋ negative eigenvalues ≥ $λ₋ detected, Gaussian random field will be " *
        "approximated (ignoring all negative eigenvalues); increase padding if possible"
    end

    # optimize
    P = measure ? plan_fft(Λ) : plan_fft(Λ, flags=FFTW.MEASURE)
    data = (Λ, P)

    GaussianRandomField{CirculantEmbedding,typeof(cov),typeof(pts),typeof(mean),typeof(data)}(mean, cov, pts, data)
end

# Extend range and shift it such that first element is zero
shift_extend(r::StepRange; n::Int=1) =
    StepRange(zero(r.start), r.step, n * (r.stop - r.start))
shift_extend(r::UnitRange; n::Int=1) =
    UnitRange(zero(r.start), n * (r.stop - r.start))
shift_extend(r::LinRange; n::Int=1) =
    LinRange(zero(r.start), n * (r.stop - r.start), n * (r.len - 1) + 1)
shift_extend(r::AbstractRange; n::Int=1) =
    range(zero(first(r)); length=n * (length(r) - 1) + 1, stop=n * (last(r) - first(r)))

# Shift r such that first element is zero and mirror it
function mirror_shift_extend(r::StepRange; n::Int=1)
    a = n * (r.stop - r.start)
    StepRange(-a, r.step, a)
end
function mirror_shift_extend(r::UnitRange; n::Int=1)
    a = n * (r.stop - r.start)
    UnitRange(-a, a)
end
function mirror_shift_extend(r::LinRange; n::Int=1)
    a = n * (r.stop - r.start)
    LinRange(-a, a, 2 * n * (r.len - 1) + 1)
end
function mirror_shift_extend(r::AbstractRange; n::Int=1)
    a = n * (last(r) - first(r))
    range(-a; length=2 * n * (length(r) - 1) + 1, stop=a)
end

# returns the required dimension of the random points
randdim(grf::GaussianRandomField{CirculantEmbedding}) = length(grf.data[1])

# sample function
function _sample(grf::GaussianRandomField{CirculantEmbedding}, xi)
    v, P = grf.data
    y = v.*reshape(xi,size(v))
    w = P*y # fft
    w = real(w) + imag(w)
    n = length.(grf.pts)
    z = w[broadcast(:,1,n)...] # select appropriate elements

    grf.mean + std(grf.cov)*z
end

Base.show(io::IO, ::CirculantEmbedding) = print(io, "circulant embedding")
