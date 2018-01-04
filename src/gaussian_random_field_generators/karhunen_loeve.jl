# TODO KL test updaten en meer gevallen testen
# TODO fool proof test
## karhunen_loeve.jl : approximate Gaussian random field generator using a Karhunen-Lo\`eve decomposition
 
"""
`struct KarhunenLoeve(n)`

Implements a Gaussiand random field generator using a Karhunen-Lo\u00C8ve expansion with n terms. 

Examples:
```
```
"""
struct KarhunenLoeve{n} <: GaussianRandomFieldGenerator end 

KarhunenLoeve(n::Integer) = n > 0 ? KarhunenLoeve{n}() : throw(ArgumentError("in KarhunenLoeve(n), number of terms n must be positive!"))

abstract type QuadratureRule end
struct GaussLegendre <: QuadratureRule end
struct EOLE <: QuadratureRule end

function _GaussianRandomField(mean,cov::CovarianceFunction{d},method::KarhunenLoeve{n},pts...;
                              nq::T=ceil(Int,n^(1/d)), quad::QuadratureRule=EOLE()) where {d,n,T<:Integer}
    # check if number of terms and number of quadrature points are compatible
    nq = nq > 0 ? nq.*tuple(ones(T,d)...) : length.(pts) # adjustment for ARPACK error when looking for ALL eigenvalues
    nq = prod(nq) == n ? nq.+1 : nq
    prod(nq) < n && throw(ArgumentError("too many terms requested, increase nq or lower n"))

    # compute quadrature nodes and weights
    struc = get_nodes_and_weights.(nq,minimum.(pts),maximum.(pts),quad)
    nodes = first.(struc)
    weights = last.(struc) 

    # eigenvalue problem
    C = apply(cov,nodes,nodes)
    W = d == 1 ? diagm(weights...) : diagm(kron(weights...))
    Wsqrt = sqrt.(W)
    B = Symmetric(Wsqrt*C*Wsqrt) # should be symmetric and positive semi-definite
    isposdef(B) || warn("equivalent eigenvalue problem is not SPD, results may be wrong or inaccurate")
    
    # solve
    (eigenval,eigenfunc) = eigs(B,nev=n,ritzvec=true,which=:LM)
    N = n + 1
    if eigenval[end] < 0.
        found = false
        N = 0
        while !found
            N += 1
            if eigenval[N] < 0
                found = true
            end
        end
        warn("negative eigenvalue $(eigenval[N]) detected, ignoring all negative eigenvalues")
    end
    K = apply(cov,pts,nodes)
    Λ = diagm(1./eigenval)
    eigenfunc = K*Wsqrt*eigenfunc*Λ

    # return data
    data = SpectralData(sqrt.(eigenval[1:N-1]),eigenfunc[:,1:N-1])
    GaussianRandomField{typeof(cov),KarhunenLoeve{N-1}}(mean,cov,pts,data)
end

# returns the required dimension of the random points
randdim(grf::GaussianRandomField{C,KarhunenLoeve{n}} where {C}) where {n} = n 

# relative error in the KL approximation
function rel_error(grf::GaussianRandomField{C,KarhunenLoeve{n}} where {C,n})
    @show Leb = prod(maximum.(grf.pts).-minimum.(grf.pts))
    return (Leb - sum(grf.data.eigenval.^2))/Leb
end

function _sample(grf::GaussianRandomField{C,KarhunenLoeve{n}} where {C,n}, xi)
    grf.mean + σ(grf)*reshape(( eigenfunctions(grf)*diagm(eigenvalues(grf)) )*xi,length.(grf.pts))
end

eigenvalues(grf::GaussianRandomField{C,KarhunenLoeve{n}} where {C<:CovarianceFunction,n}) = grf.data.eigenval
eigenfunctions(grf::GaussianRandomField{C,KarhunenLoeve{n}} where {C<:CovarianceFunction,n}) = grf.data.eigenfunc
σ(grf::GaussianRandomField{C,KarhunenLoeve{n}} where {C<:CovarianceFunction,n}) = grf.cov.cov.σ


# get first n 1d Gauss-Legendre points and weights on [a,b]
function get_nodes_and_weights(n,a,b,q::GaussLegendre)
    nodes, weights = gausslegendre(n)
    weights = (b-a)/2*weights
    nodes = (b-a)/2*nodes + (a+b)/2
    return nodes, weights
end

# get first n 1d uniform quadrature points and wieghts on [a,b]
function get_nodes_and_weights(n,a,b,q::EOLE)
    nodes = linspace(a,b,n)
    weights = (b-a)/n*ones(nodes)
    return nodes, weights
end
