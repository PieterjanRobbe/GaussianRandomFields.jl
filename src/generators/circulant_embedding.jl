# TODO
# - update GaussianRandomField docs
# - update tut
# - add See also notes to KL, Spec, Chol
# TODO  
#   merk op: in 1d geen verschil want p-norm van getal is zelfde => iso = aniso
#   in 2d wel een verschil: ifft ipv irfft ==> refactor fft afhankelijk van iso/aniso
# TODO grid not equal to [0,1]???
# TODO add plot_covariance function when the points are ranges: T, BTTB, BTBTTB... ???
# TODO and plot_eigenvalues
#####
#
# JUST implemented 2d case; test generating + sampling for ISO/ANISO
# test ander grid dan 0:1
# refactor fft voor meer efficientie
# join 1d
# docs; extra tests
#
#####
#
#
#
#
#
#
#
#
#
##### TODO  29/01
#
# plotting of eigenvalues (check with paper?); plot covariance function (only for ranges?; make surf for 1d, imagesc for multiple d; with extent?)
# jpin 1d with --->
# for (isotropic?) kernel, only need first column of C, (no mirroring), and do real fft (irfft).
# 
# testing; docs; tut ::: DONE
#####

## circulant_embedding.jl : Gaussian random field generator using fft; only for uniformly spaced GRFs

"""
    CirculantEmbedding <: GaussianRandomFieldGenerator

A [`GaussiandRandomFieldGenerator`](@ref) that uses FFT to compute samples of the Gaussian random field.

# Examples
```jldoctest
julia> 

```
See also: [`Cholesky`](@ref), [`Spectral`](@ref), [`KarhunenLoeve`](@ref)
"""
struct CirculantEmbedding <: GaussianRandomFieldGenerator end

const CirculantGRF = GaussianRandomField{C,CirculantEmbedding} where {C}

function _GaussianRandomField(mean,cov::CovarianceFunction{1},method::CirculantEmbedding,pts;padding=1)

    # add ghost points by padding
    padded_pts = pad(pts,padding)

    # compute eigenvalues of circulant matrix
    c = apply.(cov.cov,abs.(normalize(pts)))
    Λ = irfft(c,2*length(c)-1)
    Λ⁺ = zeros(size(Λ))
    Λ⁻ = zeros(size(Λ))
    for i in 1:length(Λ)
        if Λ[i] < 0
            Λ⁻[i] = -Λ[i]
        else
            Λ⁺[i] = Λ[i]
        end
    end
    Λ_max = maximum(Λ⁻)
    Λ_max > 0. && ( warn("negative eigenvalue $(real(-Λ_max)) detected, Gaussian random field will be approximated (ignoring all negative eigenvalues)"); warn("increase padding if possible") )

    # optimize
    Σ = sqrt.(Λ⁺)
    P = plan_fft(Σ,1,flags=FFTW.MEASURE)

    GaussianRandomField{typeof(cov),CirculantEmbedding,typeof(tuple(pts))}(mean,cov,tuple(pts),(Σ,P))
end

normalize(pts::R where {R<:Range}) = pts-pts[1]
mirror(pts::R where {R<:Range}) = -pts[end]:(pts[2]-pts[1]):pts[end]

function _GaussianRandomField(mean,cov::CovarianceFunction,method::CirculantEmbedding,pts...;padding=1)

    # add ghost points by padding
    padded_pts = pad.(pts,padding)
    n = length.(padded_pts)
    padded_pts = normalize.(padded_pts)
    padded_pts = mirror.(padded_pts)

    # compute eigenvalues of circulant matrix
    c = zeros(2.*n.-1)
    for (i,idx) in enumerate(Base.product(padded_pts...))
        c[i] = apply(cov.cov,collect(idx))
    end
    c̃ = zeros(2.*n)
    c̃[range.(2,2.*n.-1)...] = c
    c̃ = fftshift(c̃)
    Λ = real(ifft(c̃))
    Λ⁺ = zeros(size(Λ))
    Λ⁻ = zeros(size(Λ))
    for i in 1:length(Λ)
        if Λ[i] < 0
            Λ⁻[i] = -Λ[i]
        else
            Λ⁺[i] = Λ[i]
        end
    end
    Λ_max = maximum(Λ⁻)
    Λ_max > 0. && ( warn("negative eigenvalue $(real(-Λ_max)) detected, Gaussian random field will be approximated (ignoring all negative eigenvalues)"); warn("increase padding if possible") )

    # optimize
    Σ = sqrt.(Λ⁺)
    P = plan_fft(Σ,flags=FFTW.MEASURE)

    GaussianRandomField{typeof(cov),CirculantEmbedding,typeof(pts)}(mean,cov,pts,(Σ,P))
end

circulantify(c) = vcat(c,flipdim(c,1)[2:end-1])

function pad(x,n)
    x0 = x[1]
    xn = x[end]
    dx = x[2]-x[1]
    x0:dx:n*xn
end

# returns the required dimension of the random points
randdim(grf::CirculantGRF) = length(grf.data[1]) 
#randdim(grf::CirculantGRF) = (length(grf.data[1]),2) 

# sample function
function _sample(grf::CirculantGRF, xi)
    
    v = grf.data[1]
    @show size(v)
    @show size(xi)
    y = v.*reshape(xi,size(v))
    P = grf.data[2]
    w = P*y#./length(v)
    #@show length(w)
    w = real(w) + imag(w)
    #===== for 1d
    z = w[1:length(grf.mean)]
    #@show size(grf.mean)
      ==========#
    @show size(w)
    @show size(grf.mean)
    @show n = length.(grf.pts)
    #######z = w[:,1:2:n[2]]
    #z = reshape(w[:,1:n[2]],(n[1],2*n[2]))
    #z = z[:,1:2:end]
    z = w[range.(1,n)...]
    ## =========== ##

    grf.mean + std(grf.cov)*z



    #xi = xi*[1; 1im]
    #@show n = prod(length.(grf.pts))
    # TODO CE in 2d works, but here's an error: select only every 2nd pts (see page 271)
    #u = real( ( fft( grf.data[1].*reshape(xi,size(grf.data[1])) ) ) )./sqrt(length(grf.data[1]))
    #@show size(u)

    #u = reshape(u[2*prod(n)],(n[1],2*n[2]))
    #u = u[:,1:2:2*n[2]]
    #grf.mean + std(grf.cov)*u
    #grf.mean + std(grf.cov)*reshape(real( ( grf.data[2]*( grf.data[1].*reshape(xi,size(grf.data[1])) ) )[1:n] )./sqrt(ndims(grf.cov)*length(grf.data[1])),size(grf.mean))

    ####
    #grf.mean + std(grf.cov)*reshape(real( ( grf.data[2]*( grf.data[1].*reshape(xi,size(grf.data[1])) ) )[1:n] ),size(grf.mean))
end

show(io::IO,::CirculantEmbedding) = print(io,"circulant embedding")
