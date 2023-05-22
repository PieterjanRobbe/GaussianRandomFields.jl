## karhunen_loeve.jl : approximate Gaussian random field generator using a Karhunen-Lo\`eve decomposition

"""
    KarhunenLoeve(n)

Returns a [`GaussianRandomFieldGenerator`](@ref) using a Karhunen-Loève (KL) expansion with `n` terms. 

# Optional Arguments for `GaussianRandomField`
- `quad::QuadratureRule`: quadrature rule used for the integral equation (see [`QuadratureRule`](@ref)), default is `EOLE()`. 
- `nq::Integer`: number of quadrature points in each dimension, where we require `nq^d > n`. Default is `nq = ceil(n^(1/d))`.
- `eigensolver::EigenSolver`: which method to use for the eigenvalue decomposition (see [`AbstractEigenSolver`](@ref)). The default is `EigenSolver()`.

# Examples
```jldoctest label1
julia> cov = CovarianceFunction(2, Matern(.3, 1))
2d Matérn covariance function (λ=0.3, ν=1.0, σ=1.0, p=2.0)

julia> pts = range(0, stop=1, length=51)
0.0:0.02:1.0

julia> grf = GaussianRandomField(cov, KarhunenLoeve(300), pts, pts)
Gaussian random field with 2d Matérn covariance function (λ=0.3, ν=1.0, σ=1.0, p=2.0) on a 51×51 structured grid, using a KL expansion with 300 terms

julia> plot_eigenvalues(grf) # plot the eigenvalue decay
[...]

julia> plot_eigenfunction(grf, 4) # plots the fourth eigenfunction
[...]

```
If more terms `n` are used in the expansion, the approximation becomes better.
```jldoctest label1; setup=:(import Random; Random.seed!(12345))
julia> for n in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
           grf = GaussianRandomField(cov, KarhunenLoeve(n), pts, pts)
           println(rel_error(grf))
       end
0.6983828486813854
0.454941868632304
0.23231277904920067
0.10079295313241687
0.026470201665282467
0.009784266729696567
0.003565488318168386
0.0010719081249264129
0.00019809766995382283
4.085273649512278e-5

```
!!! note
    Techniqually, the KL expansion is computed using the Nystrom method. For nonstructured grids, we use a bounding box approach. Try using the `Spectral` method if this is not what you want.

!!! warning
    To avoid an *end effect* in the eigenvalue decay, choose `nq^d ≥ 5n`.

```jldoctest label1
julia> grf = GaussianRandomField(cov, KarhunenLoeve(300), pts, pts, quad=GaussLegendre())
Gaussian random field with 2d Matérn covariance function (λ=0.3, ν=1.0, σ=1.0, p=2.0) on a 51×51 structured grid, using a KL expansion with 300 terms

julia> grf = GaussianRandomField(cov, KarhunenLoeve(300), pts, pts, nq=40)
Gaussian random field with 2d Matérn covariance function (λ=0.3, ν=1.0, σ=1.0, p=2.0) on a 51×51 structured grid, using a KL expansion with 300 terms

```
See also: [`Cholesky`](@ref), [`Spectral`](@ref), [`CirculantEmbedding`](@ref)
"""
struct KarhunenLoeve{n} <: GaussianRandomFieldGenerator
    function KarhunenLoeve{n}() where n
        n > 0 || throw(DomainError(n, "in KarhunenLoeve{n}, number of terms n must be positive"))
        new{n}()
    end
end

KarhunenLoeve(n::Integer) = KarhunenLoeve{n}()

function _GaussianRandomField(mean, cov::CovarianceFunction{d}, method::KarhunenLoeve{n},
                              pts...;
                              nq::Integer=ceil(typeof(n), n^(1/d)),
                              quad::QuadratureRule=EOLE(),
                              eigensolver::AbstractEigenSolver=EigsSolver()) where {d,n}
    # check if number of terms and number of quadrature points are compatible
    nq = nq > 0 ? ntuple(i -> nq, d) : length.(pts)
    prod_nq = prod(nq)
    prod_nq < n && throw(ArgumentError("too many terms requested, increase nq or lower n"))
    # adjustment for ARPACK error when looking for ALL eigenvalues
    eigensolver isa EigsSolver && prod_nq == n && (nq = nq .+ 1)

    # determine bounding box when irregular mesh
    is_irr = pts[1] isa AbstractMatrix
    a = is_irr ? Tuple(minimum(pts[1],dims=2)) : minimum.(pts)
    b = is_irr ? Tuple(maximum(pts[1],dims=2)) : maximum.(pts)

    # compute quadrature nodes and weights
    struc = get_nodes_and_weights.(nq, a, b, Ref(quad))
    nodes = first.(struc)
    weights = last.(struc)

    # eigenvalue problem
    C = apply(cov, nodes...)
    W = d == 1 ? Diagonal(weights...) : Diagonal(kron(weights...))
    W .= sqrt.(W)
    B = Symmetric(W * C * W) # should be symmetric and positive semi-definite
    isposdef(B) || @warn "equivalent eigenvalue problem is not SPD, results may be wrong or inaccurate"

    # solve
    eigenval, eigenfunc = compute(B, n, eigensolver)

    # compute eigenfunctions in nodes
    K = apply(cov.cov, pts, nodes)
    Λ = Diagonal(1 ./ eigenval)
    eigenfunc = K * W * eigenfunc * Λ

    m = findfirst(x -> x < 0, eigenval)
    if m != nothing
        m -= 1
        @warn "$(length(eigenval) - m) negative eigenvalues ≥ $(eigenval[end]) detected, 
            Gaussian random field will be approximated (ignoring all negative eigenvalues)"
    else
        m = n
    end

    resize!(eigenval, m)
    eigenfunc = eigenfunc[:, 1:m]

    # store eigenvalues and eigenfunctions
    eigenval .= sqrt.(eigenval) # note: store sqrt of eigenval for more efficient sampling
    data = SpectralData(eigenval, eigenfunc)

    GaussianRandomField{KarhunenLoeve{length(eigenval)},typeof(cov),typeof(pts),typeof(mean),typeof(data)}(mean,cov,pts,data)
end

# returns the required dimension of the random points
randdim(grf::GaussianRandomField{KarhunenLoeve{n}}) where n = n

# relative error in the KL approximation
"""
    rel_error(grf)

Returns the relative error in the Karhunen-Loève approximation of the random field, computed as

``1 - \\displaystyle\\frac{\\sum \\theta_j^2}{\\sigma^2 \\int_D \\mathrm{d}x}``.

Only useful for fields defined on a rectangular domain.

# Examples
```jldoctest; setup=:(import Random; Random.seed!(12345))
julia> cov = CovarianceFunction(2, Matern(.3, 1))
2d Matérn covariance function (λ=0.3, ν=1.0, σ=1.0, p=2.0)

julia> pts = range(0, stop=1, length=51)
0.0:0.02:1.0

julia> grf = GaussianRandomField(cov, KarhunenLoeve(300), pts, pts)
Gaussian random field with 2d Matérn covariance function (λ=0.3, ν=1.0, σ=1.0, p=2.0) on a 51×51 structured grid, using a KL expansion with 300 terms

julia> rel_error(grf)
0.00046070730242930846

```
"""
function rel_error(grf::GaussianRandomField{<:KarhunenLoeve})
    Leb = prod(grf.pts) do point
        a, b = extrema(point)
        b - a
    end
    1 - sum(abs2, grf.data.eigenval) / (std(grf.cov)^2 * Leb)
end

# sample function for both Spectral() and KarhunenLoeve(n) type
function _sample(grf::GaussianRandomField{<:Union{Spectral,KarhunenLoeve}}, xi)
    grf.mean + reshape((grf.data.eigenfunc * Diagonal(grf.data.eigenval)) * xi, size(grf.mean))
end

Base.show(io::IO, ::KarhunenLoeve{n}) where n = print(io, "KL expansion with $(n) terms")
