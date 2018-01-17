# TODO add more tests!!
## test_karhunen_loeve.jl : test Karhunen-Lo\`eve expansion for GRF generation

verbose && print("testing Karhunen-Lo\u00e8ve expansion...")

# matern
cov = CovarianceFunction(1,Matern(0.3,1))
pts = 0:0.01:1
grf = GaussianRandomField(cov,KarhunenLoeve(1000),pts)
@test typeof(grf) <: GaussianRandomField{CovarianceFunction{1,Matern{T}},KarhunenLoeve{1000}} where {T}
@test length(sample(grf)) == length(pts)

####### TODO test pts / 1 pt 1 pt , 100 pt 1 pt , 0 pt ... ==> bij covariance_functions

# exponential
cov = CovarianceFunction(2,Exponential(1))
pts1 = 0:0.05:1
pts2 = 0:0.05:1
grf = GaussianRandomField(cov,KarhunenLoeve(100),pts1,pts2)
@test typeof(grf) <: GaussianRandomField{CovarianceFunction{2,Exponential{T}},KarhunenLoeve{100}} where {T}
@test size(sample(grf),1) .== length(pts1)
@test size(sample(grf),2) .== length(pts2)

# squared exponential
cov = CovarianceFunction(2,SquaredExponential(0.1))
grf = GaussianRandomField(cov,KarhunenLoeve(500),pts1,pts2)
@test typeof(grf) <: GaussianRandomField{CovarianceFunction{2,SquaredExponential{T}},KarhunenLoeve{500}} where {T}
@test size(sample(grf),1) .== length(pts1)
@test size(sample(grf),2) .== length(pts2)

# non-SPD covariance matrix (works with KarhunenLoeve if negative eigenvalues are ignored)
cov = CovarianceFunction(2,SquaredExponential(0.5))
@suppress grf = GaussianRandomField(cov,KarhunenLoeve(200),pts1,pts2) 
@test typeof(grf) <: GaussianRandomField{CovarianceFunction{2,SquaredExponential{T}},KarhunenLoeve{150}} where {T}
@test size(sample(grf),1) .== length(pts1)
@test size(sample(grf),2) .== length(pts2)

# test domain with negative sides 
cov = CovarianceFunction(2,Matern(1.,2.5,σ=1.,p=2))
pts1 = collect(-1:0.1:1)
pts2 = collect(-1:0.1:1)
grf = GaussianRandomField(cov,KarhunenLoeve(458),pts1,pts2)
@test typeof(grf) <: GaussianRandomField{CovarianceFunction{2,Matern{T}},KarhunenLoeve{458}} where {T}
@test size(sample(grf),1) .== length(pts1)
@test size(sample(grf),2) .== length(pts2)

verbose && println("done")
