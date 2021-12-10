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
plot

@recipe f(grf::GaussianRandomField{G, C}) where {G, C <: AbstractCovarianceFunction{1}} = (grf.pts[1], sample(grf))

"""
plot!(grf[, kwargs...])

Add a sample of the one-dimensional Gaussian random field `grf` to the existing plot. 

See also: [`plot`](@ref)
"""
plot!

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
surface

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
contour

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
contourf

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
heatmap


@recipe f(grf::GaussianRandomField{G, C}) where {G, C <: AbstractCovarianceFunction{2}} = begin
    linewidth -> get(plotattributes, :seriestype, :auto) == :contourf ? 0 : :auto
    # Plots.[surface|contour|contourf|heatmap] all take a triplet of arguments (x, y, z)
    # where x is the sequence of coordinates of the spatial grid along the dimension
    # to be plotted on the horizontal axis, y is the sequence of coordinates of the
    # spatial grid along the dimension to be plotted on the vertical axis, and z is a
    # two-dimensional array of shape (length(y), length(x)) with the first dimension
    # corresponding to vertical plot axis and the second dimension the horizontal plot
    # axis. As the sample(grf) returns a 2D array with first dimension corresponding
    # to grf.pts[1] and second dimension corresponding to grf.pts[2] we need to
    # transpose the sampled array to match with the Plots API
    grf.pts..., transpose(sample(grf))
end

# Plot eigenvalues
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
plot_eigenvalues

@userplot Plot_Eigenvalues

recipetype(::Val{:plot_eigenvalues}, args...) = Plot_Eigenvalues(args)

@recipe f(plt::Plot_Eigenvalues) = begin
    grf = plt.args[1]
    
    if grf isa GaussianRandomField{<:Union{KarhunenLoeve, Spectral},<:CovarianceFunction}
        val = grf.data.eigenval.^2
    elseif grf isa GaussianRandomField{<:CirculantEmbedding}
        val = sort(vec(grf.data[1]), rev=true).^2 * length(grf.data[1])
    elseif grf isa GaussianRandomField{<:KarhunenLoeve,<:SeparableCovarianceFunction}
        order, data = grf.data
        val = Array{Float64}(undef, length(order))
        for (i, o) in enumerate(order)
            val[i] = prod(data[j].eigenval[o[j]] for j in 1:length(data))^2
        end
    end

    @series begin
        seriestype --> :line
        xaxis --> :log
        yaxis --> :log
        xguide --> "eigenvalue number"
        yguide --> "magnitude"
        1:length(val), val
    end
end

# Plot eigenfunctions
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
plot_eigenfunction

@userplot Plot_Eigenfunction

recipetype(::Val{:plot_eigenfunction}, args...) = Plot_Eigenfunction(args)

@recipe f(plt::Plot_Eigenfunction) = begin
    grf = plt.args[1]
    n = plt.args[2]

    if grf isa GaussianRandomField{<:Union{KarhunenLoeve, Spectral}, <:AbstractCovarianceFunction{1}}
        x = grf.pts[1]
        if grf isa GaussianRandomField{G, <:CovarianceFunction{1}} where G
            f = view(grf.data.eigenfunc, :, n)
        elseif grf isa GaussianRandomField{G, <:SeparableCovarianceFunction{1}} where G
            order, data = grf.data
            ordern = order[n]
            f = view(data[1].eigenfunc, :, ordern[1])
        end
        @series begin
            seriestype --> :line
            x, f
        end
    end

    if grf isa GaussianRandomField{<:Union{KarhunenLoeve, Spectral}, <:AbstractCovarianceFunction{2}}
        x, y = grf.pts
        if grf isa GaussianRandomField{G, <:CovarianceFunction{2}} where G
            f = view(grf.data.eigenfunc, :, n)
        elseif grf isa GaussianRandomField{G, <:SeparableCovarianceFunction{2}} where G
            order, data = grf.data
            ordern = order[n]
            f = kron((view(data[j].eigenfunc, :, ordern[j]) for j = 1:length(grf.cov.cov))...)
        end
        @series begin
            seriestype --> :contourf
            linewidth --> 0
            x, y, f
        end
    end
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
plot_covariance_matrix

@userplot Plot_Covariance_Matrix

recipetype(::Val{:plot_covariance_matrix}, args...) = Plot_Covariance_Matrix(args)

@recipe function f(plt::Plot_Covariance_Matrix)
    @series begin
        seriestype --> :heatmap
        grf = plt.args[1]
        pts = length(plt.args) > 1 ? plt.args[2] : grf.pts
        Z = apply(grf.cov, pts...)
        x = axes(Z, 1)
        x, x, Z
    end
end
