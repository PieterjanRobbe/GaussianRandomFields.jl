## plot.jl : functions for easy visualization of Gaussian random fields

# 1D
"""
plot(grf[, kwargs...])

Plot a sample of the one-dimensional Gaussian random field `grf` and pass optional arguments `kwargs` to the `Plots.plot` command.

# Examples
```jldoctest
julia> cov = CovarianceFunction(1, Exponential(1))
1d exponential covariance function (λ=1.0, σ=1.0, p=2.0)

julia> pts = range(0, stop=1, length=501)
0.0:0.002:1.0

julia> grf = GaussianRandomField(cov, CirculantEmbedding(), pts)
Gaussian random field with 1d exponential covariance function (λ=1.0, σ=1.0, p=2.0) on a 501-point structured grid, using a circulant embedding

julia> plot(grf)
[...]

```
See also: [`plot!`](@ref)
"""
plot(grf::GaussianRandomField{G, C}; kwargs...) where {G, C <: AbstractCovarianceFunction{1}} = plot(grf.pts[1], sample(grf); kwargs...)

"""
plot!(grf[, kwargs...])

Add a sample of the one-dimensional Gaussian random field `grf` to the existing plot. 

See also: [`plot`](@ref)
"""
plot!(grf::GaussianRandomField{G, C}; kwargs...) where {G, C <: AbstractCovarianceFunction{1}} = plot!(grf.pts[1], sample(grf); kwargs...)

# 2D
"""
    surface(grf[, kwargs])

Plot a surface of the two-dimensional Gaussian random field `grf` and pass optional arguments `kwargs` to the `Plots.surface` command.

# Examples
```jldoctest
julia> cov = CovarianceFunction(2, Matern(.3, 1))
2d Matérn covariance function (λ=0.3, ν=1.0, σ=1.0, p=2.0)

julia> pts = range(0, stop=1, length=101)
0.0:0.01:1.0

julia> grf = GaussianRandomField(cov, CirculantEmbedding(), pts, pts, minpadding=157)
Gaussian random field with 2d Matérn covariance function (λ=0.3, ν=1.0, σ=1.0, p=2.0) on a 101×101 structured grid, using a circulant embedding

julia> surface(grf)
[...]

```
See also: [`contour`](@ref), [`contourf`](@ref), [`heatmap`](@ref)
"""
function surface(grf::GaussianRandomField{G, C}; kwargs...) where {G, C <: AbstractCovarianceFunction{2}}
    x, y = grf.pts
    surface(x, y, sample(grf); kwargs...)
end

"""
    contour(grf[, kwargs])

Plot a contour map of the two-dimensional Gaussian random field `grf` and pass optional arguments `kwargs` to the `Plots.contour` command.

# Examples
```jldoctest
julia> cov = CovarianceFunction(2, Matern(.3, 1))
2d Matérn covariance function (λ=0.3, ν=1.0, σ=1.0, p=2.0)

julia> pts = range(0, stop=1, length=101)
0.0:0.01:1.0

julia> grf = GaussianRandomField(cov, CirculantEmbedding(), pts, pts, minpadding=157)
Gaussian random field with 2d Matérn covariance function (λ=0.3, ν=1.0, σ=1.0, p=2.0) on a 101×101 structured grid, using a circulant embedding

julia> contour(grf)
[...]

```
See also: [`surface`](@ref), [`contourf`](@ref), [`heatmap`](@ref)
"""
function contour(grf::GaussianRandomField{G, C}; kwargs...) where {G, C <: AbstractCovarianceFunction{2}}
    x, y = grf.pts
    contour(x, y, sample(grf); kwargs...)
end

"""
    contourf(grf[, kwargs])

Plot a filled contour map of the two-dimensional Gaussian random field `grf` and pass optional arguments `kwargs` to the `Plots.contourf` command.

# Examples
```jldoctest
julia> cov = CovarianceFunction(2, Matern(.3, 1))
2d Matérn covariance function (λ=0.3, ν=1.0, σ=1.0, p=2.0)

julia> pts = range(0, stop=1, length=101)
0.0:0.01:1.0

julia> grf = GaussianRandomField(cov, CirculantEmbedding(), pts, pts, minpadding=157)
Gaussian random field with 2d Matérn covariance function (λ=0.3, ν=1.0, σ=1.0, p=2.0) on a 101×101 structured grid, using a circulant embedding

julia> contourf(grf)
[...]

```
See also: [`surface`](@ref), [`contour`](@ref), [`heatmap`](@ref)
"""
contourf(grf::GaussianRandomField; kwargs...) = contour(grf, fill=true; linewidth=0, kwargs...)

"""
    heatmap(grf[, kwargs])

Plot a heatmap of the two-dimensional Gaussian random field `grf` and pass optional arguments `kwargs` to the `Plots.heatmap` command.

# Examples
```jldoctest
julia> cov = CovarianceFunction(2, Matern(.3, 1))
2d Matérn covariance function (λ=0.3, ν=1.0, σ=1.0, p=2.0)

julia> pts = range(0, stop=1, length=101)
0.0:0.01:1.0

julia> grf = GaussianRandomField(cov, CirculantEmbedding(), pts, pts, minpadding=157)
Gaussian random field with 2d Matérn covariance function (λ=0.3, ν=1.0, σ=1.0, p=2.0) on a 101×101 structured grid, using a circulant embedding

julia> heatmap(grf)
[...]

```
See also: [`surface`](@ref), [`contour`](@ref), [`contourf`](@ref)
"""
function heatmap(grf::GaussianRandomField{G, C}; kwargs...) where {G, C <: AbstractCovarianceFunction{2}}
    x, y = grf.pts
    heatmap(x, y, sample(grf); kwargs...)
end

# Plot eigenvalues
plot_eigenvalues_sub(eigenvalues; kwargs...) = plot(1:length(eigenvalues), eigenvalues; xaxis=:log, yaxis=:log, xlabel="eigenvalue", ylabel="magnitude", kwargs...)

"""
    plot_eigenvalues(grf)

Log-log plot of the eigenvalues of the Gaussian random field. Only available for Gaussian random field generators of type `Spectral`, `KarhunenLoeve` and `CirculantEmbedding`.

# Examples
```jldoctest
julia> cov = CovarianceFunction(2, Gaussian(.1))
2d Gaussian covariance function (λ=0.1, σ=1.0, p=2.0)

julia> pts = range(0, stop=1, length=51)
0.0:0.02:1.0

julia> grf = GaussianRandomField(cov, KarhunenLoeve(100), pts, pts)
Gaussian random field with 2d Gaussian covariance function (λ=0.1, σ=1.0, p=2.0) on a 51×51 structured grid, using a KL expansion with 100 terms

julia> plot_eigenvalues(grf)
[...]

```
See also: [`plot_eigenfunction`](@ref)
"""
plot_eigenvalues(grf::GaussianRandomField{<:Union{KarhunenLoeve, Spectral}}; kwargs...) = plot_eigenvalues_sub(grf.data.eigenval.^2; kwargs...)

plot_eigenvalues(grf::GaussianRandomField{<:CirculantEmbedding}; kwargs...) = begin
    data = grf.data[1]
    eigenvalues = sort(vec(data), rev=true).^2 .* length(data)
    plot_eigenvalues_sub(eigenvalues; kwargs...)
end

plot_eigenvalues(grf::GaussianRandomField{<:KarhunenLoeve,<:SeparableCovarianceFunction}; kwargs...) = begin
    order, data = grf.data
    eigenvalues = Array{Float64}(undef, length(order))
    @inbounds for (i, o) in enumerate(order)
        eigenvalues[i] = prod(data[j].eigenval[o[j]] for j in 1:length(data))^2
    end
    plot_eigenvalues_sub(eigenvalues; kwargs...)
end

# Plot eigenfunctions
plot_eigenfunction_1d(x, eigenfunction; kwargs...) = plot(x, eigenfunction; kwargs...)

plot_eigenfunction_2d(x, y, eigenfunction; kwargs...) = contourf(x, y, eigenfunction; linewidth=0, kwargs...)

"""
    plot_eigenfunction(grf, n)

Contour plot of the `n`th eigenfunction of the Gaussian random field. Only available for Gaussian random field generators of type `Spectral` and `KarhunenLoeve`.

# Examples
```jldoctest
julia> cov = CovarianceFunction(2, Gaussian(.1))
2d Gaussian covariance function (λ=0.1, σ=1.0, p=2.0)

julia> pts = range(0, stop=1, length=51)
0.0:0.02:1.0

julia> grf = GaussianRandomField(cov, KarhunenLoeve(100), pts, pts)
Gaussian random field with 2d Gaussian covariance function (λ=0.1, σ=1.0, p=2.0) on a 51×51 structured grid, using a KL expansion with 100 terms

julia> plot_eigenfunction(grf, 6) # 6th eigenfunction
[...]

```
See also: [`plot_eigenfunction`](@ref)
"""
plot_eigenfunction(grf::GaussianRandomField{<:Union{KarhunenLoeve, Spectral}, C}, n; kwargs...) where C <: AbstractCovarianceFunction{1} = begin
    x = grf.pts[1]
    plot_eigenfunction_1d(x, view(grf.data.eigenfunc, :, n); kwargs...)
end

plot_eigenfunction(grf::GaussianRandomField{<:Union{KarhunenLoeve, Spectral}, C}, n; kwargs...) where C <: SeparableCovarianceFunction{1} = begin
    x = grf.pts[1]
    order, data = grf.data
    ordern = order[n]
    eigenfunction = view(data[1].eigenfunc, :, ordern[1])
    plot_eigenfunction_1d(x, eigenfunction; kwargs...)
end

plot_eigenfunction(grf::GaussianRandomField{<:Union{KarhunenLoeve, Spectral}, C}, n; kwargs...) where C <: AbstractCovarianceFunction{2} = begin
    x, y = grf.pts
    plot_eigenfunction_2d(x, y, view(grf.data.eigenfunc, :, n); kwargs...)
end

plot_eigenfunction(grf::GaussianRandomField{<:Union{KarhunenLoeve, Spectral}, C}, n; kwargs...) where C <: SeparableCovarianceFunction{2} = begin
    x, y = grf.pts
    order, data = grf.data
    ordern = order[n]
    eigenfunction = kron((view(data[j].eigenfunc, :, ordern[j]) for j = 1:length(grf.cov.cov))...)
    plot_eigenfunction_2d(x, y, eigenfunction; kwargs...)
end

# Plot covariance matrix
"""
    plot_covariance_matrix(grf[, pts...])

    Evaluate the covariance function of the Gaussian random field `grf` in the (optional) points `pts` and plot the result.

# Examples
```jldoctest
julia> cov = CovarianceFunction(2, Matern(.3, 1))
2d Matérn covariance function (λ=0.3, ν=1.0, σ=1.0, p=2.0)

julia> pts = range(0, stop=1, length=11)
0.0:0.1:1.0

julia> grf = GaussianRandomField(cov, CirculantEmbedding(), pts, pts, minpadding=7)
Gaussian random field with 2d Matérn covariance function (λ=0.3, ν=1.0, σ=1.0, p=2.0) on a 11×11 structured grid, using a circulant embedding

julia> plot_covariance_matrix(grf)
[...]

julia> pts = range(0, stop=1, length=21)
0.0:0.05:1.0

julia> plot_covariance_matrix(grf, pts, pts)
[...]

```
"""
plot_covariance_matrix(grf::GaussianRandomField) = plot_covariance_matrix(grf::GaussianRandomField, grf.pts...)

function plot_covariance_matrix(grf::GaussianRandomField,
                                pts::Vararg{AbstractRange, d}) where d
    heatmap(apply(grf.cov, pts...))
end
