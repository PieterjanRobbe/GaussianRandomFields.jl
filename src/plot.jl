# TODO plot separable
# TODO write tests
# TODO jldoctest

## plot.jl : functions for easy visualization of Gaussian random fields
const OneDimCov = Union{CovarianceFunction{1},SeparableCovarianceFunction{1}}
const TwoDimCov = Union{CovarianceFunction{2},SeparableCovarianceFunction{2}}
const ThreeDimCov = Union{CovarianceFunction{3},SeparableCovarianceFunction{3}}

const OneDimGRF = GaussianRandomField{C} where {C<:OneDimCov}
const TwoDimGRF = GaussianRandomField{C} where {C<:TwoDimCov}
const ThreeDimGRF = GaussianRandomField{C} where {C<:ThreeDimCov}

const OneDimSpectralGRF = GaussianRandomField{C,KarhunenLoeve{n}} where {C<:OneDimCov,n}
const TwoDimSpectralGRF = GaussianRandomField{C,KarhunenLoeve{n}} where {C<:TwoDimCov,n}
const ThreeDimSpectralGRF = GaussianRandomField{C,KarhunenLoeve{n}} where {C<:ThreeDimCov,n}

## 1D ##
function plot(grf::OneDimGRF;n=1,kwargs...)
    for i in 1:n
        plot(grf.pts[1],sample(grf),kwargs...)
    end
end

## 2D ##
plot(grf::TwoDimGRF;kwargs...) = surf(grf,kwargs...)

function surf(grf::TwoDimGRF;kwargs...)
    x,y = grf.pts
    xgrid = [x[i] for i = 1:length(x), j = 1:length(y)]
    ygrid = [y[j] for i = 1:length(x), j = 1:length(y)]
    plot_surface(ygrid,xgrid,reshape(sample(grf),(length(x),length(y))),rstride=2,edgecolors="k",cstride=2,cmap=ColorMap("viridis"))
end

function contourf(grf::TwoDimGRF;kwargs...)
    x,y = grf.pts
    xgrid = repmat(x,1,length(y))
    ygrid = repmat(y',length(x),1)
    contourf(ygrid,xgrid,reshape(sample(grf),(length(x),length(y)))')
    colorbar()
end

function contour(grf::TwoDimGRF;kwargs...)
    x,y = grf.pts
    xgrid = repmat(x,1,length(y))
    ygrid = repmat(y',length(x),1)
    cp = contour(ygrid,xgrid,reshape(sample(grf),(length(x),length(y)))',colors="black")
    clabel(cp, inline=1, fontsize=10)
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
        order = (2,3,1)
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
    xgrid = xs[order[1]][dxs[order[1]]]*ones(length(idcs1),length(idcs2))
    ygrid = [xs[order[2]][i] for i = idcs1, j = idcs2]
    zgrid = [xs[order[3]][j] for i = idcs1, j = idcs2]
    grids = (xgrid,ygrid,zgrid)
    plot_surface([grids[i] for i in order]...,facecolors=get_cmap("viridis").o((cut-minimum(A))/(maximum(A)-minimum(A))),shade=false,kwargs...)
end

## eigenvalues and eigenfunctions ##
function plot_eigenvalues(grf::GaussianRandomField{C,KarhunenLoeve{n}} where {C,n})
    ev = eigenvalues(grf).^2
    loglog(1:length(ev),ev)
    xlabel("n")
    ylabel("magnitude")
end

function plot_eigenfunction(grf::GaussianRandomField{C,KarhunenLoeve{n}} where {C,n}, n::Integer)
    ( n > 0 && n < randdim(grf)+1 ) || throw(ArgumentError("eigenfunction index n must be between 1 and $(randdim(grf))"))
    _plot_eigenfunction(grf,n)
end

function _plot_eigenfunction(grf::OneDimSpectralGRF, n::Integer)
    plot(grf.pts[1],eigenfunctions(grf)[:,n])
end

function _plot_eigenfunction(grf::TwoDimSpectralGRF, n::Integer)
    x,y = grf.pts
    xgrid = repmat(x,1,length(y))
    ygrid = repmat(y',length(x),1)
    contourf(xgrid,ygrid,reshape(eigenfunctions(grf)[:,n],(length(x),length(y))))
    colorbar()
end

function _plot_eigenfunction(grf::ThreeDimSpectralGRF, n::Integer)
    x,y,z = grf.pts
    nx = length(x); nx2 =round(Int,nx/2)
    ny = length(y); ny2 =round(Int,ny/2)
    nz = length(z); nz2 =round(Int,nz/2)
    A = reshape(eigenfunctions(grf)[:,n],(nx,ny,nz))
    slice(x,y,z,nx2,ny2,nz2,A,:x)
    slice(x,y,z,nx2,ny2,nz2,A,:y)
    slice(x,y,z,nx2,ny2,nz2,A,:z)
end
