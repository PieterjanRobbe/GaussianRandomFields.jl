## plot.jl : functions for easy visualization of Gaussian random fields

# 1D
plot(grf::GaussianRandomField{G, C}; kwargs...) where {G, C <: AbstractCovarianceFunction{1}} = plot(grf.pts[1], sample(grf); kwargs...)

plot!(grf::GaussianRandomField{G, C}; kwargs...) where {G, C <: AbstractCovarianceFunction{1}} = plot!(grf.pts[1], sample(grf); kwargs...)

# 2D
for f in [:surface, :contour, :contourf, :heatmap]
    expr = quote
        function $f(grf::GaussianRandomField{G, C}; kwargs...) where {G, C <: AbstractCovarianceFunction{2}}
            x, y = grf.pts
            $f(x, y, sample(grf); kwargs...)
        end
    end
    eval(expr)
end

# Plot eigenvalues
plot_eigenvalues_sub(eigenvalues) = plot(1:length(eigenvalues), eigenvalues, xaxis=:log, yaxis=:log, xlabel="eigenvalue", ylabel="magnitude")

plot_eigenvalues(grf::GaussianRandomField{<:KarhunenLoeve}) = plot_eigenvalues_sub(grf.data.eigenval.^2)

plot_eigenvalues(grf::GaussianRandomField{<:CirculantEmbedding}) = begin
    data = grf.data[1]
    eigenvalues = sort(vec(data), rev=true).^2 .* length(data)
    plot_eigenvalues_sub(eigenvalues)
end

plot_eigenvalues(grf::GaussianRandomField{<:KarhunenLoeve,<:SeparableCovarianceFunction}) = begin
    order, data = grf.data
    eigenvalues = Array{Float64}(undef, length(order))
    @inbounds for (i, o) in enumerate(order)
        eigenvalues[i] = prod(data[j].eigenval[o[j]] for j in 1:length(data))^2
    end
    plot_eigenvalues_sub(eigenvalues)
end

# Plot eigenfunctions
plot_eigenfunction_1d(x, eigenfunction; kwargs...) = plot(x, eigenfunction; kwargs...)

plot_eigenfunction_2d(x, y, eigenfunction; kwargs...) = contourf(x, y, eigenfunction; kwargs...)

plot_eigenfunction(grf::GaussianRandomField{<:KarhunenLoeve, C}, n; kwargs...) where C <: AbstractCovarianceFunction{1} = begin
    x = grf.pts[1]
    plot_eigenfunction_1d(x, view(grf.data.eigenfunc, :, n); kwargs...)
end

plot_eigenfunction(grf::GaussianRandomField{<:KarhunenLoeve, C}, n) where C <: SeparableCovarianceFunction{1} = begin
    x = grf.pts[1]
    order, data = grf.data
    ordern = order[n]
    eigenfunction = view(data[1].eigenfunc, :, ordern[1])
    plot_eigenfunction_1d(x, eigenfunction)
end

plot_eigenfunction(grf::GaussianRandomField{<:KarhunenLoeve, C}, n; kwargs...) where C <: AbstractCovarianceFunction{2} = begin
    x, y = grf.pts
    plot_eigenfunction_2d(x, y, view(grf.data.eigenfunc, :, n); kwargs...)
end

plot_eigenfunction(grf::GaussianRandomField{<:KarhunenLoeve, C}, n) where C <: SeparableCovarianceFunction{2} = begin
    x, y = grf.pts
    order, data = grf.data
    ordern = order[n]
    eigenfunction = kron((view(data[j].eigenfunc, :, ordern[j]) for j = 1:length(grf.cov.cov))...)
    plot_eigenfunction_2d(x, y, eigenfunction)
end

# Plot covariance matrix
function plot_covariance_matrix(c::CovarianceFunction{d},
                                pts::Vararg{AbstractRange, d}) where d
    heatmap(apply(c, pts, pts))
end
