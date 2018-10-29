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

# whether a covariance function is even in every dimension
# this simplifies the circulant embedding algorithm
Base.iseven(::CovarianceStructure) = false
Base.iseven(::IsotropicCovarianceStructure) = true

function _GaussianRandomField(mean, cov::CovarianceFunction{d}, method::CirculantEmbedding,
                              pts::Vararg{AbstractRange,d};
                              minpadding::Union{Int,Dims{d}} = 0,
                              measure::Bool = true,
                              primes::Bool = false) where {d}
    # normalize points
    normedpts = map(x -> x .- first(x), pts)

    # compute size of minimum circulant embedding
    ρ = cov.cov
    factors = primes ? [2, 3, 5, 7] : [2]
    dims = circulant_minsize.(Ref(ρ), normedpts, minpadding, Ref{Vector{Int}}(factors))

    # compute eigenvalues of the circulant embedding
    Λ = circulant_eigvals(ρ, normedpts, dims)

    # store sqrt of eigenvalues for more efficient sampling
    n₋, λ₋ = 0, 0.0
    M = length(Λ)
    @inbounds for i in eachindex(Λ)
        λ = Λ[i]
        if λ < 0
            # reset negative eigenvalues to zero
            Λ[i] = 0

            # update statistics about negative eigenvalues
            n₋ += 1
            λ < λ₋ && (λ₋ = λ)
        else
            Λ[i] = sqrt(λ / M)
        end
    end
    λ₋ < 0 && @warn begin
        "$n₋ negative eigenvalues ≥ $λ₋ detected, Gaussian random field will be " *
        "approximated (ignoring all negative eigenvalues); increase padding if possible"
    end

    # optimize
    P = measure ? plan_fft(Λ, flags=FFTW.MEASURE) : plan_fft(Λ)
    data = (Λ, P)

    GaussianRandomField{CirculantEmbedding,typeof(cov),typeof(pts),typeof(mean),typeof(data)}(mean, cov, pts, data)
end

"""
    circulant_minsize(cov::CovarianceStructure, pts::AbstractRange,
                      minpadding::Int, factors::Vector{Int})

Return size of minimum circulant embedding of covariance matrix of points `pts` with
covariance function `cov` and minimum padding `minpadding` that can be written as product of
integers in `factors`.

Typically `factors` is chosen to be `[2]` or `[2, 3, 5, 7]` to speed up FFT computations.
"""
function circulant_minsize(cov::CovarianceStructure, pts::AbstractRange,
                           minpadding::Int, factors::Vector{Int})
    if iseven(cov)
        2 * nextprod(factors, length(pts) + minpadding - 1)
    else
        2 * nextprod(factors, length(pts) + minpadding)
    end
end

"""
    circulant_eigvals(cov::CovarianceStructure, pts::NTuple{N,AbstractRange},
                      dims::Dims{N}) where {N}

Compute eigenvalues of circulant embedding with dimensions `dims` of covariance matrix of
points `pts` with covariance function `cov`.
"""
@generated function circulant_eigvals(cov::CovarianceStructure{T},
                                      pts::NTuple{N,AbstractRange},
                                      dims::Dims{N}) where {T,N}
    quote
        if iseven(cov)
            # compute first row of circulant embedding
            # since the covariance function is even in every dimension,
            # instead of 2n only n+1 values have to be computed in every dimension
            C = Array{$T}(undef, div.(dims, 2) .+ 1)
            x = Array{$T}(undef, $N)

            @nloops $N i C d -> begin
                @inbounds x[d] = extrapolate(pts[d], i_d)
            end begin
                @inbounds (@nref $N C i) = apply(cov, x)
            end

            # compute eigenvalues of circulant embedding
            # since the covariance function is even in every dimension,
            # we can perform a real even FFT and extend the resulting array of eigenvalues
            FFTW.r2r!(C, FFTW.REDFT00)
            Λ = Array{eltype(C)}(undef, dims)
            Imax = CartesianIndex(size(Λ) .+ 2)
            @inbounds for i in CartesianIndices(Λ)
                Λ[i] = C[min(i, Imax - i)]
            end
            Λ
        else
            # compute first row of circulant embedding
            C = zeros($T, dims)
            x = Array{$T}(undef, $N)

            dims2 = dims .+ 2
            mids = div.(dims2, 2)
            @nloops $N i C d -> begin
                @inbounds begin
                    m = mids[d]
                    i_d == m && continue

                    if i_d < m
                        x[d] = extrapolate(pts[d], i_d)
                    else
                        x[d] = -extrapolate(pts[d], dims2[d] - i_d)
                    end
                end
            end begin
                @inbounds (@nref $N C i) = apply(cov, x)
            end

            # compute eigenvalues of circulant embedding
            real.(fft(C))
        end
    end
end

# extrapolate ranges
extrapolate(r::Union{StepRangeLen,LinRange}, i::Integer) = Base.unsafe_getindex(r, i)
extrapolate(r::UnitRange{T}, i::Integer) where T = convert(T, r.start + (i - 1))
extrapolate(r::Base.OneTo{T}, i::Integer) where T = convert(T, i)
extrapolate(r::AbstractRange{T}, i::Integer) where T =
    convert(T, first(r) + (i - 1)*step(r))

# returns the required dimension of the random points
randdim(grf::GaussianRandomField{CirculantEmbedding}) = length(grf.data[1])

# sample function
function _sample(grf::GaussianRandomField{CirculantEmbedding}, xi)
    v, P = grf.data

    # compute multiplication with square root of circulant embedding via FFT
    y = v .* reshape(xi, size(v))
    w = P * y

    # extract realization of random field
    μ, σ = grf.mean, std(grf.cov)
    z = Array{eltype(grf.cov)}(undef, length.(grf.pts))
    @inbounds for i in CartesianIndices(z)
        wi = w[i]
        z[i] = μ[i] + σ * (real(wi) + imag(wi))
    end
    z
end

Base.show(io::IO, ::CirculantEmbedding) = print(io, "circulant embedding")
