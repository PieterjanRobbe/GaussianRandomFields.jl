## test_spectral.jl : test spectral method for GRF generation

@testset "spectral decomposition   " begin

## 1d Mat\'ern ##
cov = CovarianceFunction(1,Matern(0.3,1))
pts = 0:0.01:1
grf = GaussianRandomField(cov,Spectral(),pts)
@test isa(grf,GaussianRandomField)
@test isa(grf.cov,CovarianceFunction)
@test isa(grf.cov.cov,Matern)
@test ndims(grf.cov) == 1
@test isa(grf,GaussianRandomField{C,Spectral} where {C})
@test length(grf.pts[1]) == length(pts)
@test length(sample(grf)) == length(pts)

## 2d Exponential ##
cov = CovarianceFunction(2,Exponential(1))
pts1 = 0:0.05:1
pts2 = 0:0.05:1
grf = GaussianRandomField(cov,Spectral(),pts1,pts2)
@test isa(grf,GaussianRandomField)
@test isa(grf.cov,CovarianceFunction)
@test isa(grf.cov.cov,Exponential)
@test ndims(grf.cov) == 2
@test isa(grf,GaussianRandomField{C,Spectral} where {C})
@test length(grf.pts) == 2
@test length(grf.pts[1]) == length(pts1)
@test length(grf.pts[2]) == length(pts2)
@test size(sample(grf),1) == length(pts1)
@test size(sample(grf),2) == length(pts2)

## 3d Gaussian ##
cov = CovarianceFunction(3,Matern(1,2.5))
pts1 = 0:0.1:1
pts2 = 0:0.1:1
pts3 = 0:0.1:1
grf = GaussianRandomField(cov,Spectral(),pts1,pts2,pts3)
@test isa(grf,GaussianRandomField)
@test isa(grf.cov,CovarianceFunction)
@test isa(grf.cov.cov,Matern)
@test ndims(grf.cov) == 3
@test isa(grf,GaussianRandomField{C,Spectral} where {C})
@test length(grf.pts) == 3
@test length(grf.pts[1]) == length(pts1)
@test length(grf.pts[2]) == length(pts2)
@test length(grf.pts[3]) == length(pts3)
@test size(sample(grf),1) == length(pts1)
@test size(sample(grf),2) == length(pts2)
@test size(sample(grf),3) == length(pts3)

## non-SPD covariance matrix ##
cov = CovarianceFunction(2,SquaredExponential(0.5))
pts1 = 0:0.05:1
pts2 = 0:0.05:1
@suppress grf = GaussianRandomField(cov,Spectral(),pts1,pts2) 
@test isa(grf,GaussianRandomField)
@test isa(grf.cov,CovarianceFunction)
@test isa(grf.cov.cov,SquaredExponential)
@test ndims(grf.cov) == 2
@test isa(grf,GaussianRandomField{C,Spectral} where {C})
@test length(grf.pts) == 2
@test length(grf.pts[1]) == length(pts1)
@test length(grf.pts[2]) == length(pts2)
@test size(sample(grf),1) == length(pts1)
@test size(sample(grf),2) == length(pts2)

## test non-equidistant structured grid  ##
cov = CovarianceFunction(2,Matern(1.,2.5,σ=1.,p=2))
pts1 = sin.(-pi/2:0.2:pi/2)
pts2 = sin.(-pi/2:0.1:pi/2)
grf = GaussianRandomField(cov,Spectral(),pts1,pts2)
@test isa(grf,GaussianRandomField)
@test isa(grf.cov,CovarianceFunction)
@test isa(grf.cov.cov,Matern)
@test ndims(grf.cov) == 2
@test isa(grf,GaussianRandomField{C,Spectral} where {C})
@test length(grf.pts) == 2
@test length(grf.pts[1]) == length(pts1)
@test length(grf.pts[2]) == length(pts2)
@test size(sample(grf),1) == length(pts1)
@test size(sample(grf),2) == length(pts2)

## test anisotropic
cov = CovarianceFunction(2,AnisotropicExponential([1000 0; 0 1000]))
pts1 = range(0,stop = 10,length = 64)
pts2 = range(0,stop = 10,length = 64)
grf = GaussianRandomField(cov,Spectral(),pts1,pts2)
@test isa(grf,GaussianRandomField)
@test isa(grf.cov,CovarianceFunction)
@test isa(grf.cov.cov,AnisotropicExponential)
@test ndims(grf.cov) == 2
@test isa(grf,GaussianRandomField{C,Spectral} where {C})
@test length(grf.pts) == 2
@test length(grf.pts[1]) == length(pts1)
@test length(grf.pts[2]) == length(pts2)
@test size(sample(grf),1) == length(pts1)
@test size(sample(grf),2) == length(pts2)

end
