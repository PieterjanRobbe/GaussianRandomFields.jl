## plot.jl : functions for easy visualization of Gaussian random fields

## type aliases ##
const OneDimCov = CovarianceFunction{1}
const TwoDimCov = CovarianceFunction{2}
const ThreeDimCov = CovarianceFunction{3}
const OneDimSepCov = SeparableCovarianceFunction{1}
const TwoDimSepCov = SeparableCovarianceFunction{2}
const ThreeDimSepCov = SeparableCovarianceFunction{3}

const OneDimGRF = GaussianRandomField{G,<:Union{OneDimCov,OneDimSepCov}} where G
const TwoDimGRF = GaussianRandomField{G,<:Union{TwoDimCov,TwoDimSepCov}} where G
const ThreeDimGRF = GaussianRandomField{G,<:Union{ThreeDimCov,ThreeDimSepCov}} where G
const FiniteElemGRF = GaussianRandomField{G,<:TwoDimCov,<:NTuple{2,AbstractMatrix}} where G

const OneDimSpectralGRF = GaussianRandomField{<:KarhunenLoeve,<:OneDimCov}
const TwoDimSpectralGRF = GaussianRandomField{<:KarhunenLoeve,<:TwoDimCov}
const ThreeDimSpectralGRF = GaussianRandomField{<:KarhunenLoeve,<:ThreeDimCov}
const OneDimSpectralSepGRF = GaussianRandomField{<:KarhunenLoeve,<:OneDimSepCov}
const TwoDimSpectralSepGRF = GaussianRandomField{<:KarhunenLoeve,<:TwoDimSepCov}
const ThreeDimSpectralSepGRF = GaussianRandomField{<:KarhunenLoeve,<:ThreeDimSepCov}
const FiniteElemSpectralGRF = GaussianRandomField{<:KarhunenLoeve,<:TwoDimCov,<:NTuple{2,AbstractMatrix}}

## 1D ##
function PyPlot.plot(grf::OneDimGRF; n::Int=1, kwargs...)
    for _ in 1:n
        plot(grf.pts[1], sample(grf); kwargs...)
    end
end

## 2D ##
PyPlot.plot(grf::TwoDimGRF; kwargs...) = surf(grf; kwargs...)
PyPlot.plot(grf::FiniteElemGRF; kwargs...) = tricontourf(grf; kwargs...)

function PyPlot.surf(grf::TwoDimGRF; kwargs...)
    x, y = grf.pts
    xgrid = repeat(x, 1, length(y))
    ygrid = repeat(y', length(x), 1)
    plot_surface(ygrid, xgrid, reshape(sample(grf), length(x), length(y));
                 rstride=2, edgecolors="k", cstride=2, cmap=ColorMap("viridis"), kwargs...)
end

function PyPlot.contourf(grf::TwoDimGRF; kwargs...)
    x, y = grf.pts
    xgrid = repeat(x, 1, length(y))
    ygrid = repeat(y', length(x), 1)
    contourf(xgrid, ygrid, reshape(sample(grf), length(x), length(y)); kwargs...)
    colorbar()
end

function PyPlot.contour(grf::TwoDimGRF; kwargs...)
    x, y = grf.pts
    xgrid = repeat(x, 1, length(y))
    ygrid = repeat(y', length(x), 1)
    cp = contour(xgrid, ygrid, reshape(sample(grf), length(x), length(y));
                 colors="black", kwargs...)
    clabel(cp, inline=1, fontsize=10)
end

function PyPlot.tricontourf(grf::FiniteElemGRF; kwargs...)
    p, t = grf.pts
    isempty(t) && throw(ArgumentError("cannot plot mesh when random field is computed in element centers"))
    x = p[1, :]
    y = p[2, :]
    triangles = t' .- 1
    tricontourf(x, y, sample(grf); triangles=triangles, cmap=get_cmap("viridis"), kwargs...)
    triplot(x, y; triangles=triangles, color="k", linewidth=0.5, kwargs...)
end

function PyPlot.plot_trisurf(grf::FiniteElemGRF; kwargs...)
    p, t = grf.pts
    isempty(t) && throw(ArgumentError("cannot plot mesh when random field is computed in element centers"))
    x = p[1, :]
    y = p[2, :]
    plot_trisurf(x, y, sample(grf); triangles=t' .- 1, cmap=get_cmap("viridis"), kwargs...)
end

## 3D ##
function PyPlot.plot(grf::ThreeDimGRF;
                     ix::Int=round(Int, length(grf.pts[1]) / 2),
                     iy::Int=round(Int, length(grf.pts[2]) / 2),
                     iz::Int=round(Int, length(grf.pts[3]) / 2), kwargs...)
    nx, ny, nz = length.(grf.pts)
    0 < ix ≤ nx || throw(DomainError(ix, "invalid index"))
    0 < iy ≤ ny || throw(DomainError(iy, "invalid index"))
    0 < iz ≤ nz || throw(DomainError(iz, "invalid index"))
    dxs = (ix, iy, iz)

    A = reshape(sample(grf), nx, ny, nz)
    for mode in (:x, :y, :z)
        slice(grf.pts, dxs, A, mode; kwargs...)
    end
end

function slice(xs, dxs, A, mode; kwargs...)
    if mode == :x
        order = (1,2,3)
    elseif mode == :y
        order = (2,1,3)
    elseif mode == :z
        order = (3,2,1)
    end

    minA, maxA = extrema(A)
    diffA = maxA - minA

    for quadrant in (:I, :II, :III, :IV)
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

        mslices = (dxs[order[1]], idcs1, idcs2)
        cut = A[[mslices[i] for i in order]...]
        cut = mode == :z ? cut' : cut

        xgrid = fill(xs[order[1]][dxs[order[1]]], length(idcs1), length(idcs2))
        ygrid = [xs[order[2]][i] for i = idcs1, j = idcs2]
        zgrid = [xs[order[3]][j] for i = idcs1, j = idcs2]
        grids = (xgrid,ygrid,zgrid)

        plot_surface((grids[i] for i in order)...;
                     facecolors=get_cmap("viridis").o((cut .- minA) ./ diffA), shade=false,
                     kwargs...)
    end
end

## eigenvalues and eigenfunctions ##
function plot_eigenvalues(grf::GaussianRandomField{<:KarhunenLoeve,<:CovarianceFunction})
    ev = grf.data.eigenval.^2
    loglog(1:length(ev), ev)
    xlabel("n")
    ylabel("magnitude")
end

function plot_eigenvalues(grf::GaussianRandomField{CirculantEmbedding,<:CovarianceFunction})
    ev = sort(vec(grf.data[1]), rev=true).^2
    loglog(1:length(ev), ev)
    xlabel("n")
    ylabel("magnitude")
end

function plot_eigenvalues(grf::GaussianRandomField{<:KarhunenLoeve,<:SeparableCovarianceFunction})
	order, data = grf.data
	ev = Array{Float64}(undef, length(order))
	@inbounds for (i, o) in enumerate(order)
		ev[i] = prod(data[j].eigenval[o[j]] for j in 1:length(data)) ^ 2
	end
    loglog(1:length(ev), ev)
    xlabel("n")
    ylabel("magnitude")
end

function plot_eigenfunction(grf::GaussianRandomField{<:KarhunenLoeve}, n::Int; kwargs...)
    0 < n ≤ randdim(grf) || throw(DomainError(n, "eigenfunction index n must be between 1 and $(randdim(grf))"))
    _plot_eigenfunction(grf, n; kwargs...)
end

function _plot_eigenfunction(grf::OneDimSpectralGRF, n::Int; kwargs...)
    plot(grf.pts[1], view(grf.data.eigenfunc, :, n))
end

function _plot_eigenfunction(grf::TwoDimSpectralGRF, n::Int; kwargs...)
    x, y = grf.pts
    xgrid = repeat(x, 1, length(y))
    ygrid = repeat(y', length(x), 1)
    contourf(xgrid, ygrid, reshape(view(grf.data.eigenfunc, :, n), length(x), length(y));
             kwargs...)
    colorbar()
end

function _plot_eigenfunction(grf::OneDimSpectralSepGRF, n::Int; kwargs...)
	order, data = grf.data
	plot(grf.pts[1], view(data[1].eigenfunc, :,n))
end

function _plot_eigenfunction(grf::TwoDimSpectralSepGRF, n::Int; kwargs...)
    x,y = grf.pts
    xgrid = repeat(x, 1, length(y))
    ygrid = repeat(y', length(x), 1)
	order, data = grf.data
    ordern = order[n]
	ef = kron((view(data[j].eigenfunc, :, ordern[j]) for j = 1:length(grf.cov.cov))...)
    contourf(xgrid, ygrid, reshape(ef, length(x), length(y)); kwargs...)
    colorbar()
end

function _plot_eigenfunction(grf::ThreeDimSpectralGRF, n::Int; kwargs...)
    nxs = length.(grf.pts)
    dxs = round.(Int, nxs ./ 2)
    A = reshape(view(grf.data.eigenfunc, :, n), length.(grf.pts))
    for mode in (:x, :y, :z)
        slice(grf.pts, dxs, A, mode)
    end
end

function _plot_eigenfunction(grf::ThreeDimSpectralSepGRF, n::Int; kwargs...)
    nxs = length.(grf.pts)
    dxs = round.(Int, nxs ./ 2)
	order, data = grf.data
    ordern = order[n]
	ef = kron((view(data[j].eigenfunc, :, ordern[j]) for j = 1:length(grf.cov.cov))...)
    A = reshape(ef, nxs)
    for mode in (:x, :y, :z)
        slice(grf.pts, dxs, A, mode)
    end
end

function _plot_eigenfunction(grf::FiniteElemSpectralGRF, n::Int; kwargs...)
    p, t = grf.pts
    isempty(t) && throw(ArgumentError("cannot plot mesh when random field is computed in element centers"))
    x = p[1, :]
    y = p[2, :]
    triangles = t' .- 1
    tricontourf(x, y, view(grf.data.eigenfunc, :, n); triangles=triangles,
                cmap=get_cmap("viridis"), kwargs...)
    triplot(x, y; triangles=triangles, color="k", linewidth=0.5, kwargs...)
end

## plot_covariance_matrix ##
function plot_covariance_matrix(c::CovarianceFunction{d},
                                pts::Vararg{AbstractRange,d}) where d
    C = apply(c, pts, pts)
    imshow(C)
    colorbar()
end
