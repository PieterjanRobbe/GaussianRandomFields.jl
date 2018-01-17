## plot.jl : functions for easy visualization of Gaussian random fields

## type aliases ##
const OneDimCov = CovarianceFunction{1}
const TwoDimCov = CovarianceFunction{2}
const ThreeDimCov = CovarianceFunction{3}
const TwoDimSepCov = SeparableCovarianceFunction{2}
const ThreeDimSepCov = SeparableCovarianceFunction{3}

const OneDimGRF = GaussianRandomField{C} where {C<:OneDimCov}
const TwoDimGRF = GaussianRandomField{C} where {C<:Union{TwoDimCov,TwoDimSepCov}}
const ThreeDimGRF = GaussianRandomField{C} where {C<:Union{ThreeDimCov,ThreeDimSepCov}}
const FiniteElemGRF = GaussianRandomField{C,M,Tuple{T1,T2}} where {C<:TwoDimCov,M,T1<:AbstractMatrix,T2<:AbstractMatrix}

const OneDimSpectralGRF = GaussianRandomField{C,KarhunenLoeve{n}} where {C<:OneDimCov,n}
const TwoDimSpectralGRF = GaussianRandomField{C,KarhunenLoeve{n}} where {C<:TwoDimCov,n}
const ThreeDimSpectralGRF = GaussianRandomField{C,KarhunenLoeve{n}} where {C<:ThreeDimCov,n}
const TwoDimSpectralSepGRF = GaussianRandomField{C,KarhunenLoeve{n}} where {C<:TwoDimSepCov,n}
const ThreeDimSpectralSepGRF = GaussianRandomField{C,KarhunenLoeve{n}} where {C<:ThreeDimSepCov,n}
const FiniteElemSpectralGRF = GaussianRandomField{C,KarhunenLoeve{n},Tuple{T1,T2}} where {C<:TwoDimCov,n,T1<:AbstractMatrix,T2<:AbstractMatrix}

## 1D ##
function plot(grf::OneDimGRF;n=1,kwargs...)
    for i in 1:n
        plot(grf.pts[1],sample(grf),kwargs...)
    end
end

## 2D ##
plot(grf::TwoDimGRF;kwargs...) = surf(grf,kwargs...)
plot(grf::FiniteElemGRF;kwargs...) = tricontourf(grf,kwargs...)

function surf(grf::TwoDimGRF;kwargs...)
    x,y = grf.pts
    xgrid = repmat(x,1,length(y))
    ygrid = repmat(y',length(x),1)
    plot_surface(ygrid,xgrid,reshape(sample(grf),(length(x),length(y))),rstride=2,edgecolors="k",cstride=2,cmap=ColorMap("viridis"),kwargs...)
end

function contourf(grf::TwoDimGRF;kwargs...)
    x,y = grf.pts
    xgrid = repmat(x,1,length(y))
    ygrid = repmat(y',length(x),1)
    contourf(xgrid,ygrid,reshape(sample(grf),(length(x),length(y))),kwargs...)
    colorbar()
end

function contour(grf::TwoDimGRF;kwargs...)
    x,y = grf.pts
    xgrid = repmat(x,1,length(y))
    ygrid = repmat(y',length(x),1)
    cp = contour(xgrid,ygrid,reshape(sample(grf),(length(x),length(y))),colors="black",kwargs...)
    clabel(cp, inline=1, fontsize=10)
end

function tricontourf(grf::FiniteElemGRF;kwargs...)
    (p,t) = grf.pts
    isempty(t) && throw(ArgumentError("cannot plot mesh when random field is computed in element centers"))
    x = p[1,:]
    y = p[2,:]
    tricontourf(x,y,sample(grf),triangles=t'-1,cmap=get_cmap("viridis"),kwargs...)
    triplot(x,y,triangles=t'-1,color="k",linewidth=0.5,kwargs...)
end

function plot_trisurf(grf::FiniteElemGRF;kwargs...)
    (p,t) = grf.pts
    isempty(t) && throw(ArgumentError("cannot plot mesh when random field is computed in element centers"))
    x = p[1,:]
    y = p[2,:]
    plot_trisurf(x,y,sample(grf),triangles=t'-1,cmap=get_cmap("viridis"),kwargs...)
end

## 3D ##
function plot(grf::ThreeDimGRF;kwargs...)
    kwargs = Dict(kwargs)
    x,y,z = grf.pts
    nx,ny,nz = length.(grf.pts)
    nx2 = haskey(kwargs,:ix) ? is_valid_idx(kwargs[:ix],nx) : round(Int,nx/2)
    ny2 = haskey(kwargs,:iy) ? is_valid_idx(kwargs[:iy],ny) : round(Int,ny/2)
    nz2 = haskey(kwargs,:iz) ? is_valid_idx(kwargs[:iz],nz) : round(Int,nz/2)
    A = reshape(sample(grf),(nx,ny,nz))
    slice(x,y,z,nx2,ny2,nz2,A,:x)
    slice(x,y,z,nx2,ny2,nz2,A,:y)
    slice(x,y,z,nx2,ny2,nz2,A,:z)
end

is_valid_idx(idx,n) = ( typeof(idx)<:Integer && idx > 0 && idx < n+1 ) ? idx : throw(ArgumentError("invalid index $(idx)"))

function slice(x,y,z,dx,dy,dz,A,mode,kwargs...)
    for quadrant in (:I,:II,:III,:IV)
        quadrant_slice((x,y,z),(dx,dy,dz),A,quadrant,mode,kwargs...)
    end
end

function quadrant_slice(xs,dxs,A,quadrant,mode,kwargs...)
    if mode == :x
        order = (1,2,3)
    elseif mode == :y
        order = (2,1,3)
    elseif mode == :z
        order = (3,2,1)
    end

    if quadrant == :I
        idcs1 = 1:dxs[order[2]]
        idcs2 = 1:dxs[order[3]]
    elseif quadrant == :II
        idcs1 = dxs[order[2]]:length(xs[order[2]])
        idcs2 = 1:dxs[order[3]]
    elseif quadrant == :III
        idcs1 = dxs[order[2]]:length(xs[order[2]])
        idcs2 = dxs[order[3]]:length(xs[order[3]])
    elseif quadrant == :IV
        idcs1 = 1:dxs[order[2]]
        idcs2 = dxs[order[3]]:length(xs[order[3]])
    end
    mslices = (dxs[order[1]],idcs1,idcs2)
    cut = A[[mslices[i] for i in order]...]
	cut = mode == :z ? cut' : cut
    xgrid = xs[order[1]][dxs[order[1]]]*ones(length(idcs1),length(idcs2))
    ygrid = [xs[order[2]][i] for i = idcs1, j = idcs2]
    zgrid = [xs[order[3]][j] for i = idcs1, j = idcs2]
    grids = (xgrid,ygrid,zgrid)
    plot_surface([grids[i] for i in order]...,facecolors=get_cmap("viridis").o((cut-minimum(A))/(maximum(A)-minimum(A))),shade=false,kwargs...)
end

## eigenvalues and eigenfunctions ##
function plot_eigenvalues(grf::GaussianRandomField{C,KarhunenLoeve{n}} where {C<:CovarianceFunction,n})
    ev = grf.data.eigenval.^2
    loglog(1:length(ev),ev)
    xlabel("n")
    ylabel("magnitude")
end

function plot_eigenvalues(grf::GaussianRandomField{S,KarhunenLoeve{n}} where {S<:SeparableCovarianceFunction}) where {n}
	(order,data) = grf.data
	ev = zeros(n)
	for i in eachindex(ev)
		ev[i] = prod([data[j].eigenval[order[i][j]] for j = 1:length(grf.cov.cov)]).^2
	end
    loglog(1:length(ev),ev)
    xlabel("n")
    ylabel("magnitude")
end

function plot_eigenfunction(grf::GaussianRandomField{C,KarhunenLoeve{n}} where {C,n}, n::Integer)
    ( n > 0 && n < randdim(grf)+1 ) || throw(ArgumentError("eigenfunction index n must be between 1 and $(randdim(grf))"))
    _plot_eigenfunction(grf,n)
end

function _plot_eigenfunction(grf::OneDimSpectralGRF, n::Integer)
    plot(grf.pts[1],grf.data.eigenfunc[:,n])
end

function _plot_eigenfunction(grf::TwoDimSpectralGRF, n::Integer)
    x,y = grf.pts
    xgrid = repmat(x,1,length(y))
    ygrid = repmat(y',length(x),1)
    contourf(xgrid,ygrid,reshape(grf.data.eigenfunc[:,n],(length(x),length(y))))
    colorbar()
end

function _plot_eigenfunction(grf::TwoDimSpectralSepGRF, n::Integer)
    x,y = grf.pts
    xgrid = repmat(x,1,length(y))
    ygrid = repmat(y',length(x),1)
	(order,data) = grf.data
	ef = kron([data[j].eigenfunc[:,order[n][j]] for j = 1:length(grf.cov.cov)]...)
    contourf(xgrid,ygrid,reshape(ef,(length(x),length(y))))
    colorbar()
end

function _plot_eigenfunction(grf::ThreeDimSpectralGRF, n::Integer)
    x,y,z = grf.pts
    nx = length(x); nx2 =round(Int,nx/2)
    ny = length(y); ny2 =round(Int,ny/2)
    nz = length(z); nz2 =round(Int,nz/2)
    A = reshape(grf.data.eigenfunc[:,n],(nx,ny,nz))
    slice(x,y,z,nx2,ny2,nz2,A,:x)
    slice(x,y,z,nx2,ny2,nz2,A,:y)
    slice(x,y,z,nx2,ny2,nz2,A,:z)
end

function _plot_eigenfunction(grf::ThreeDimSpectralSepGRF, n::Integer)
    x,y,z = grf.pts
    nx = length(x); nx2 =round(Int,nx/2)
    ny = length(y); ny2 =round(Int,ny/2)
    nz = length(z); nz2 =round(Int,nz/2)
	(order,data) = grf.data
	ef = kron([data[j].eigenfunc[:,order[n][j]] for j = 1:length(grf.cov.cov)]...)
    A = reshape(ef,(nx,ny,nz))
    slice(x,y,z,nx2,ny2,nz2,A,:x)
    slice(x,y,z,nx2,ny2,nz2,A,:y)
    slice(x,y,z,nx2,ny2,nz2,A,:z)
end

function _plot_eigenfunction(grf::FiniteElemSpectralGRF, n::Integer)
    (p,t) = grf.pts
    isempty(t) && throw(ArgumentError("cannot plot mesh when random field is computed in element centers"))
    x = p[1,:]
    y = p[2,:]
    tricontourf(x,y,grf.data.eigenfunc[:,n],triangles=t'-1,cmap=get_cmap("viridis"))
    triplot(x,y,triangles=t'-1,color="k",linewidth=0.5)
end
