## test_cholesky.jl : test Cholesky factorization for GRF generation

verbose && print("testing Cholesky factorization...")

# matern
cov = CovarianceFunction(1,Matern(0.3,1))
pts = 0:0.01:1
grf = GaussianRandomField(cov,Cholesky(),pts)
@test typeof(grf) == GaussianRandomField{CovarianceFunction{1,Matern{Float64}},Cholesky}
@test length(sample(grf)) == length(pts)

# exponential
cov = CovarianceFunction(2,Exponential(1))
pts1 = 0:0.05:1
pts2 = 0:0.05:1
grf = GaussianRandomField(cov,Cholesky(),pts1,pts2)
@test typeof(grf) == GaussianRandomField{CovarianceFunction{2,Exponential{Float64}},Cholesky}
@test size(sample(grf),1) .== length(pts1)
@test size(sample(grf),2) .== length(pts2)

# squared exponential
cov = CovarianceFunction(2,SquaredExponential(0.1))
grf = GaussianRandomField(cov,Cholesky(),pts1,pts2)
@test typeof(grf) == GaussianRandomField{CovarianceFunction{2,SquaredExponential{Float64}},Cholesky}
@test size(sample(grf),1) .== length(pts1)
@test size(sample(grf),2) .== length(pts2)

# non-SPD covariance matrix
cov = CovarianceFunction(2,SquaredExponential(0.5))
@test_throws ArgumentError GaussianRandomField(cov,Cholesky(),pts1,pts2) 

# test domain with negative sides 
cov = CovarianceFunction(2,Matern(1.,2.5,σ=1.,p=2))
pts1 = collect(-1:0.1:1)
pts2 = collect(-1:0.1:1)
grf = GaussianRandomField(cov,Cholesky(),pts1,pts2)
@test typeof(grf) == GaussianRandomField{CovarianceFunction{2,Matern{Float64}},Cholesky}
@test size(sample(grf),1) .== length(pts1)
@test size(sample(grf),2) .== length(pts2)

verbose && println("done")
