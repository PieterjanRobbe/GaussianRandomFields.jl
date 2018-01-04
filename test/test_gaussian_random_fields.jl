# TODO make some tests for KL or Spectral
## test_gaussian_random_fields.jl : test implementation of GRF constructors

verbose && print("testing Gaussian random fields...")

# vectors instead of linspaces
cov = CovarianceFunction(2,Exponential(0.1))
pts1 = collect(0:0.05:1)
pts2 = collect(0:0.01:1)
grf = GaussianRandomField(cov,Cholesky(),pts1,pts2)
@test typeof(grf) == GaussianRandomField{CovarianceFunction{2,Exponential{Float64}},Cholesky}
@test size(sample(grf),1) .== length(pts1)
@test size(sample(grf),2) .== length(pts2)

# test non-equidistant grid
cov = CovarianceFunction(2,Matern(0.3,1.))
pts1 = 0.5 * (cos.((0:0.05:1)*pi) + 1)
pts2 = 0.5 * (cos.((0:0.05:1)*pi) + 1)
grf = GaussianRandomField(cov,Cholesky(),pts1,pts2)
@test typeof(grf) == GaussianRandomField{CovarianceFunction{2,Matern{Float64}},Cholesky}
@test size(sample(grf),1) .== length(pts1)
@test size(sample(grf),2) .== length(pts2)

# test wrong dimension of points
cov = CovarianceFunction(2,Exponential(0.3))
pts = 0:0.1:1
@test_throws DimensionMismatch GaussianRandomField(cov,Cholesky(),pts)

# test sample with random number vector (and wrong length)
cov = CovarianceFunction(1,Matern(1.0,2.,σ=1.))
pts = 0:0.01:1
grf = GaussianRandomField(cov,Cholesky(),pts)
@test typeof(grf) == GaussianRandomField{CovarianceFunction{1,Matern{Float64}},Cholesky}
@test length(sample(grf,xi=rand(length(pts)))) == length(pts)
@test_throws DimensionMismatch sample(grf,xi=rand(length(pts)+1))

# test supplying a mean value for the field
cov = CovarianceFunction(2,Exponential(1))
pts1 = 0:0.05:1
pts2 = 0:0.05:1
grf = GaussianRandomField(1,cov,Cholesky(),pts1,pts2)
@test typeof(grf) == GaussianRandomField{CovarianceFunction{2,Exponential{Float64}},Cholesky}
@test size(sample(grf),1) .== length(pts1)
@test size(sample(grf),2) .== length(pts2)
grf = GaussianRandomField(2.0*ones(length(pts1),length(pts2)),cov,Cholesky(),pts1,pts2)
@test typeof(grf) == GaussianRandomField{CovarianceFunction{2,Exponential{Float64}},Cholesky}
@test size(sample(grf),1) .== length(pts1)
@test size(sample(grf),2) .== length(pts2)
@test_throws DimensionMismatch GaussianRandomField(2.0*ones(5,10),cov,Cholesky(),pts1,pts2)

verbose && println("done")
