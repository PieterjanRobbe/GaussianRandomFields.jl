## fem.jl ##
abstract type GaussianRandomFieldGenerator end

function GaussianRandomField(mean::Vector{T},cov::CovarianceFunction{d},method::GaussianRandomFieldGenerator,p::Matrix{T},t::Matrix{N};mode="nodes",kwargs...) where {d,T<:Real,N<:Int}
    size(p,2) == d || throw(DimensionMismatch("second dimension of points must be equal to $(d)"))
    size(t,2) == d+1 || throw(DimensionMismatch("second dimension of nodes must be equal to $(d+1)"))
	typeof(method) <: CirculantEmbedding && throw(ArgumentError("cannot use circulant embedding with a finite element mesh"))
    if mode == "center"
        length(mean) == size(t,1) || throw(DimensionMismatch("size of the mean does not correspond to the dimension of the points"))
        pts = compute_centers(p,t)'
        tri = Matrix{N}(undef,0,0)
    elseif mode == "nodes"
        length(mean) == size(p,1) || throw(DimensionMismatch("size of the mean does not correspond to the dimension of the points"))
        pts = p'
        tri = t'
    else
        throw(ArgumentError("unknown mode $(mode)"))
    end
    _GaussianRandomField(mean,cov,method,pts,tri;kwargs...)
end

GaussianRandomField(cov::CovarianceFunction,method::GaussianRandomFieldGenerator,p::Matrix{T},t::Matrix{N} where {N<:Int};kwargs...) where {T<:Real} = GaussianRandomField(0,cov,method,p,t;kwargs...)

function GaussianRandomField(mean::N where {N<:Real},cov::CovarianceFunction,method::GaussianRandomFieldGenerator,p::Matrix{T},t::Matrix{N} where {N<:Int};mode="nodes",kwargs...) where {T<:Real}
    if mode == "center"
        M = mean*ones(T,size(t,1))
    elseif mode == "nodes"
        M = mean*ones(T,size(p,1))
    else
        throw(ArgumentError("unknown mode $(mode)"))
    end
    GaussianRandomField(M,cov,method,p,t;mode=mode,kwargs...)
end

function compute_centers(p,t)
    n = size(t,1)
    d = size(p,2)
    pts = zeros(n,d) 	 
    for i in 1:d
        x = p[t[:],i]
        x = reshape(x,size(t))
        pts[:,i] = mean(x,dims=2)
    end
    pts
end

shape(grf::GaussianRandomField{C,M,Tuple{T1,T2}}) where {C,M,T1<:AbstractMatrix,T2<:AbstractMatrix} = size(grf.pts[1],2)

show(io::IO,grf::GaussianRandomField{C,M,Tuple{T1,T2}}) where {C,M,T1<:AbstractMatrix,T2<:AbstractMatrix} = print(io, "Gaussian random field with $(grf.cov) on a mesh with $(size(grf.pts[1],2)) points and $(size(grf.pts[2],2)) elements, using a $(M())")
