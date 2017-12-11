## plot.jl : functions for easy visualization of Gaussian random fields

function plot(x,grf::GaussianRandomField{CovarianceFunction{1,T},N} where {T,N};n=1,kwargs...)
    for i in 1:n
        plot(x,sample(grf),kwargs...)
    end
end

function plot(x,y,grf::GaussianRandomField{CovarianceFunction{2,T},N} where {T,N};kwargs...)
    xgrid = repmat(x,1,length(y))
    ygrid = repmat(y',length(x),1)
    plot_surface(xgrid,ygrid,reshape(sample(grf),(length(x),length(y))),rstride=2,edgecolors="k",cstride=2,cmap=ColorMap("viridis"))
end

function contourf(x,y,grf::GaussianRandomField{CovarianceFunction{2,T},N} where {T,N};kwargs...)
    xgrid = repmat(x,1,length(y))
    ygrid = repmat(y',length(x),1)
    contourf(xgrid,ygrid,reshape(sample(grf),(length(x),length(y))))
    colorbar()
end

function contour(x,y,grf::GaussianRandomField{CovarianceFunction{2,T},N} where {T,N};kwargs...)
    xgrid = repmat(x,1,length(y))
    ygrid = repmat(y',length(x),1)
    cp = contour(xgrid,ygrid,reshape(sample(grf),(length(x),length(y))),colors="black")
    clabel(cp, inline=1, fontsize=10)
end
