## quadrature.jl : nystrom integration methods

abstract type QuadratureRule end

struct GaussLegendre <: QuadratureRule end
struct EOLE <: QuadratureRule end
struct Simpson <: QuadratureRule end
struct Midpoint <: QuadratureRule end
struct Trapezoidal <: QuadratureRule end

# get nodes and weights of Gauss-Legendre quadrature on [a,b]
function get_nodes_and_weights(n::Int, a, b, q::GaussLegendre)
    nodes, weights = gausslegendre(n)
    weights = (b-a)/2*weights
    nodes = (b-a)/2*nodes .+ (a+b)/2
    return nodes, weights
end

# get nodes and weights of structured grid on [a,b]
function get_nodes_and_weights(n::Int, a, b, q::EOLE)
    nodes = range(a; stop = b,length = n)
    weights = fill((b-a) / n, n)
    return nodes, weights
end

# get nodes and weights of Simpson's rule on [a,b]
function get_nodes_and_weights(n::Int, a, b, q::Simpson)
    iseven(n) || begin
        @warn "to use Simpson's rule, n must be even (received $(n)). I will continue with n = $(n+1)"
        n += 1
    end
    Δx = (b-a)/n
    nodes = a:Δx:b
    weights = repeat(2:2:4, outer=Int(n/2))
    weights[1] = 1
    push!(weights,1)
    weights *= Δx/3
    return nodes, weights
end

# get nodes and weights of Midpoint rule on [a,b]
function get_nodes_and_weights(n::Int, a, b, q::Midpoint)
    Δx = (b-a)/n
    nodes = (a+Δx/2):Δx:(b-Δx/2)
    weights = fill(Δx, size(nodes))
    return nodes, weights
end

# get nodes and weights of Trapezoidal rule on [a,b]
function get_nodes_and_weights(n::Int, a, b, q::Trapezoidal)
    Δx = (b-a)/n
    nodes = a:Δx:b
    weights = fill(Δx, length(nodes))
    weights[1] /= 2
    weights[end] /= 2
    return nodes, weights
end
