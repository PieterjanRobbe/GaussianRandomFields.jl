# TODO fix when KL implementation is done
## analytic.jl :special cases where eigenfunctions are known analytically

# (separable) exponential covariance function with p = 0.5
function GaussianRandomField(cov::CovarianceFunction{1,Exponential},m::Method where {M<:GaussianRandomFieldGenerator},pts::AbstractArray{T,N} where {T,N})
    if cov.p == 1
        return ...
    else
        GaussianRandomField(cov,m,collect(pts))
    end
end
