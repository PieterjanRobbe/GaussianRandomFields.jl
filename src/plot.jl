## plot.jl : functions for easy visualization of Gaussian random fields

function plot(grf::GaussianRandomField{CovarianceFunction{1,T},N} where {T,N};n=1,kwargs...)
    p = plot(vec(grf.pts),sample(grf),kwargs...)
    for i in 2:n
        plot!(vec(grf.pts),sample(grf),kwargs...)
    end
    gui()
end
