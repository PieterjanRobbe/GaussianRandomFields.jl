## test_spectral.jl : test spectral method for GRF generation

verbose && print("testing spectral method...")

# matern
cov = CovarianceFunction(1,Matern(0.3,1))
pts = 0:0.01:1
grf = GaussianRandomField(cov,Spectral(),pts)
@test typeof(grf) == GaussianRandomField{CovarianceFunction{1,Matern{Float64}},Spectral}
@test length(sample(grf)) == length(pts)

# exponential
cov = CovarianceFunction(2,Exponential(1))
pts1 = 0:0.05:1
pts2 = 0:0.05:1
grf = GaussianRandomField(cov,Spectral(),pts1,pts2)
@test typeof(grf) == GaussianRandomField{CovarianceFunction{2,Exponential{Float64}},Spectral}
@test size(sample(grf),1) .== length(pts1)
@test size(sample(grf),2) .== length(pts2)

# squared exponential
cov = CovarianceFunction(2,SquaredExponential(0.1))
grf = GaussianRandomField(cov,Spectral(),pts1,pts2)
@test typeof(grf) == GaussianRandomField{CovarianceFunction{2,SquaredExponential{Float64}},Spectral}
@test size(sample(grf),1) .== length(pts1)
@test size(sample(grf),2) .== length(pts2)

# TODO is there an easy fix for this, such as the padding in CirculantEmbedding?? Try with KL!!
# non-SPD covariance matrix (works with Spectral if negative eigenvalues are ignored)
cov = CovarianceFunction(2,SquaredExponential(0.5))
@suppress grf = GaussianRandomField(cov,Spectral(),pts1,pts2) 
@test typeof(grf) == GaussianRandomField{CovarianceFunction{2,SquaredExponential{Float64}},Spectral}
@test size(sample(grf),1) .== length(pts1)
@test size(sample(grf),2) .== length(pts2)

# test domain with negative sides 
cov = CovarianceFunction(2,Matern(1.,2.5,σ=1.,p=2))
pts1 = collect(-1:0.1:1)
pts2 = collect(-1:0.1:1)
grf = GaussianRandomField(cov,Spectral(),pts1,pts2)
@test typeof(grf) == GaussianRandomField{CovarianceFunction{2,Matern{Float64}},Spectral}
@test size(sample(grf),1) .== length(pts1)
@test size(sample(grf),2) .== length(pts2)

verbose && println("done")
