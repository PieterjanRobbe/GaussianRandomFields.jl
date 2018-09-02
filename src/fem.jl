## fem.jl ##

function GaussianRandomField(mean::Vector{<:Real}, cov::CovarianceFunction{d}, method::GaussianRandomFieldGenerator, p::Matrix{<:Real}, t::Matrix{Int}; mode="nodes", kwargs...) where d
    size(p,2) == d || throw(DimensionMismatch("second dimension of points must be equal to $(d)"))
    size(t,2) == d+1 || throw(DimensionMismatch("second dimension of nodes must be equal to $(d+1)"))
	method isa CirculantEmbedding && throw(ArgumentError("cannot use circulant embedding with a finite element mesh"))

    if mode == "center"
        length(mean) == size(t,1) || throw(DimensionMismatch("size of the mean does not correspond to the dimension of the points"))
        pts = compute_centers(p,t)'
        tri = Matrix{Int}(undef,0,0)
    elseif mode == "nodes"
        length(mean) == size(p,1) || throw(DimensionMismatch("size of the mean does not correspond to the dimension of the points"))
        pts = p'
        tri = t'
    else
        throw(ArgumentError("unknown mode $(mode)"))
    end
    _GaussianRandomField(mean, cov, method, pts, tri; kwargs...)
end

GaussianRandomField(cov::CovarianceFunction, method::GaussianRandomFieldGenerator, p::Matrix{<:Real}, t::Matrix{Int}; kwargs...) = GaussianRandomField(0, cov, method, p, t; kwargs...)

function GaussianRandomField(mean::Real, cov::CovarianceFunction, method::GaussianRandomFieldGenerator, p::Matrix{<:Real}, t::Matrix{Int}; mode="nodes", kwargs...)
    T = promote_type(typeof(mean), eltype(p))
    if mode == "center"
        M = fill(convert(T, mean), size(t, 1))
    elseif mode == "nodes"
        M = fill(convert(T, mean), size(p, 1))
    else
        throw(ArgumentError("unknown mode $(mode)"))
    end
    GaussianRandomField(M, cov, method, p, t; mode=mode, kwargs...)
end

function compute_centers(p,t)
    d = size(p, 2)
    vec_t = vec(t)
    size_t = size(t)

    pts = Array{Float64}(undef, size(t, 1), d)
    @inbounds for i in 1:d
        x = reshape(p[vec_t, i], size_t)
        mean!(view(pts, :, i), x)
    end
    pts
end

shape(grf::GaussianRandomField{G,C,<:NTuple{2,AbstractMatrix}} where {G,C}) =
    size(grf.pts[1], 2)

showpoints(io::IO, points::NTuple{2,AbstractMatrix}) =
    print(io, "mesh with ", size(points[1], 2), " points and ", size(points[2], 2),
          " elements")
