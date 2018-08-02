## karhunen_loeve.jl : approximate Gaussian random field generator using a Karhunen-Lo\`eve decomposition
 
"""
    KarhunenLoeve{n} <: GaussianRandomFieldGenerator

A [`GaussiandRandomFieldGenerator`](@ref) using a Karhunen-Lo\u00e8ve (KL) expansion with `n` terms. 

# Examples
```jldoctest
julia> m = Matern(0.1,1.0)
Matérn (λ=0.1, ν=1.0, σ=1.0, p=2.0)

julia> c = CovarianceFunction(2,m)
2d Matérn covariance function (λ=0.1, ν=1.0, σ=1.0, p=2.0)

julia> pts1 = 0:0.02:1; pts2 = 0:0.02:1 
0.0:0.02:1.0

julia> grf = GaussianRandomField(c,KarhunenLoeve(300),pts1,pts2)
Gaussian random field with 2d Matérn covariance function (λ=0.1, ν=1.0, σ=1.0, p=2.0) on a 51x51 structured grid, using a KL expansion with 300 terms

julia> plot_eigenvalues(grf) # plot the eigenvalue decay
[...]

julia> plot_eigenfunction(grf,4) # plots the fourth eigenfunction
[...]

```
The more terms are retained in the expansion, the better the approximation will be.
```jldoctest
julia> nterms = [1 2 5 10 20 50 100 200 500 1000]
1×10 Array{Int64,2}:
 1  2  5  10  20  50  100  200  500  1000

julia> for n in nterms
       grf = GaussianRandomField(c,KarhunenLoeve(n),pts1,pts2)
       @show rel_error(grf)
       end
rel_error(grf) = 0.7499982529722711
rel_error(grf) = 0.49999825591379987
rel_error(grf) = 0.4425751861338164
rel_error(grf) = 0.35789475278408045
rel_error(grf) = 0.16805079842673853
rel_error(grf) = 0.11187098338277579
rel_error(grf) = 0.05130466343704787
rel_error(grf) = 0.017343327476498027
rel_error(grf) = 0.0034278579378175245
rel_error(grf) = 0.0007216777927243623

```
The KL expansion is computed using the Nystrom method. Optional argument are the quadrature rule `quad` and the number of quadrature points in each direction `nq`. Possible values for `quad` are `GaussLegendre()`, `Midpoint()`, `Simpson()`, `Trapezoidal` and `EOLE()` (default). The total number of quadrature points must be larger than or equal to the requested number of terms. Default is `ceil(n^(1/d))`. This value is best left untouched.

```jldoctest
julia> grf = GaussianRandomField(c,KarhunenLoeve(300),pts1,pts2,quad=GaussLegendre())
Gaussian random field with 2d Matérn covariance function (λ=0.1, ν=1.0, σ=1.0, p=2.0) on a 51x51 structured grid, using a KL expansion with 300 terms

julia> grf = GaussianRandomField(c,KarhunenLoeve(300),pts1,pts2,quad=Simpson(),nq=20)
Gaussian random field with 2d Matérn covariance function (λ=0.1, ν=1.0, σ=1.0, p=2.0) on a 51x51 structured grid, using a KL expansion with 300 terms

```
See also: [`Cholesky`](@ref), [`Spectral`](@ref), [`CirculantEmbedding`](@ref)
"""
struct KarhunenLoeve{n} <: GaussianRandomFieldGenerator end 

KarhunenLoeve(n::Integer) = n > 0 ? KarhunenLoeve{n}() : throw(ArgumentError("in KarhunenLoeve(n), number of terms n must be positive"))

const KarhunenLoeveGRF{n} = GaussianRandomField{C,KarhunenLoeve{n}} where {n,C}
const KL{n} = KarhunenLoeve{n}

function _GaussianRandomField(mean,cov::CovarianceFunction{d},method::KL{n},pts...;
                              nq::T=ceil(typeof(n),n^(1/d)), quad::QuadratureRule=EOLE()) where {d,n,T<:Integer}
    # check if number of terms and number of quadrature points are compatible
    nq = nq > 0 ? nq .* tuple(ones(T,d)...) : length.(pts) 
    nq = prod(nq) == n ? nq.+1 : nq # adjustment for ARPACK error when looking for ALL eigenvalues
    prod(nq) < n && throw(ArgumentError("too many terms requested, increase nq or lower n"))

    # determine bounding box when irregular mesh
    is_irr = typeof(pts[1]) <: AbstractMatrix
    a = is_irr ? Tuple(minimum(pts[1],dims=2)) : minimum.(pts)
    b = is_irr ? Tuple(maximum(pts[1],dims=2)) : maximum.(pts)

    # compute quadrature nodes and weights
    struc = get_nodes_and_weights.(nq,a,b,Ref(quad))
    nodes = first.(struc)
    weights = last.(struc) 

    # eigenvalue problem
    C = apply(cov,nodes,nodes)
    W = d == 1 ? diagm(0 => weights...) : diagm(0 => kron(weights...))
    Wsqrt = sqrt.(W)
    B = Symmetric(Wsqrt*C*Wsqrt) # should be symmetric and positive semi-definite
    isposdef(B) || warn("equivalent eigenvalue problem is not SPD, results may be wrong or inaccurate")
    
    # solve
    (eigenval,eigenfunc) = eigs(B,nev=n,ritzvec=true,which=:LM)
 	N = find_last_positive(eigenval)
	N == length(eigenval) || warn("negative eigenvalue $(eigenval[N+1]) detected, Gaussian random field will be approximated (ignoring all negative eigenvalues)")

    # compute eigenfunctions in nodes
    K = apply(cov,pts,nodes)
    Λ = diagm(0 => 1 ./ eigenval)
    eigenfunc = K*Wsqrt*eigenfunc*Λ

    # store eigenvalues and eigenfunctions
    data = SpectralData(sqrt.(eigenval[1:N]),eigenfunc[:,1:N]) # note: store sqrt of eigenval for more efficient sampling

    GaussianRandomField{typeof(cov),KL{N},typeof(pts)}(mean,cov,pts,data)
end

# returns the required dimension of the random points
randdim(grf::KarhunenLoeveGRF{n}) where {n} = n 

# relative error in the KL approximation
function rel_error(grf::KarhunenLoeveGRF)
    Leb = prod(maximum.(grf.pts).-minimum.(grf.pts))
	return (std(grf.cov)^2*Leb - sum(grf.data.eigenval.^2))/(std(grf.cov)^2*Leb)
end

# sample function for both Spectral() and KarhunenLoeve(n) type
function _sample(grf::Union{SpectralGRF,KarhunenLoeveGRF}, xi)
    grf.mean + std(grf.cov)*reshape(( grf.data.eigenfunc*diagm(0 => grf.data.eigenval) )*xi,size(grf.mean))
end

show(io::IO,::KarhunenLoeve{n}) where {n} = print(io,"KL expansion with $(n) terms")
