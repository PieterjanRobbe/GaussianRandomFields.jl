## test_separable.jl : test separable kernels for GRF generation

verbose && print("testing separable kernels...")

# exponential 2d
e1 = Exponential(0.1)
e2 = Exponential(0.01)
cov = SeparableCovarianceFunction(e1,e2)
pts1 = 0:0.05:1
pts2 = 0:0.05:1
grf = GaussianRandomField(cov,KarhunenLoeve(500),pts1,pts2)
@test typeof(grf) == GaussianRandomField{SeparableCovarianceFunction{2,Vector{Exponential{Float64}}},KarhunenLoeve{500}}
@test size(sample(grf),1) == length(pts1)
@test size(sample(grf),2) == length(pts2)

# exponential 1d
e1 = Exponential(0.75)
cov = SeparableCovarianceFunction(e1)
pts = 0:0.01:1
grf = GaussianRandomField(cov,KarhunenLoeve(300),pts,quad=GaussLegendre())
@test typeof(grf) == GaussianRandomField{SeparableCovarianceFunction{1,Vector{Exponential{Float64}}},KarhunenLoeve{300}}
@test length(sample(grf)) == length(pts)

# matern/exponential 2d
m1 = Matern(0.2,1.0)
e1 = Exponential(0.01)
cov = SeparableCovarianceFunction([m1,e1])
pts1 = 0:0.05:1
pts2 = 0:0.05:1
grf = GaussianRandomField(cov,KarhunenLoeve(250),pts1,pts2,quad=EOLE())
@test typeof(grf) == GaussianRandomField{SeparableCovarianceFunction{2,Vector{GaussianRandomFields.CovarianceStructure{Float64}}},KarhunenLoeve{250}}
@test size(sample(grf),1) == length(pts1)
@test size(sample(grf),2) == length(pts2)

# squared exponential/exponential 2d
s1 = SquaredExponential(0.1)
e1 = Exponential(0.01)
cov = SeparableCovarianceFunction(m1,e1)
pts1 = 0:0.01:1
pts2 = 0:0.01:1
grf = GaussianRandomField(cov,KarhunenLoeve(25),pts1,pts2)
@test typeof(grf) == GaussianRandomField{SeparableCovarianceFunction{2,Vector{GaussianRandomFields.CovarianceStructure{Float64}}},KarhunenLoeve{25}}
@test size(sample(grf),1) == length(pts1)
@test size(sample(grf),2) == length(pts2)

# matern/exponential 3d
m1 = Matern(0.2,1.0)
e1 = Exponential(0.01)
e2 = Exponential(0.1)
cov = SeparableCovarianceFunction(m1,e1,e2)
pts1 = 0:0.1:1
pts2 = 0:0.1:1
pts3 = 0:0.1:1
grf = GaussianRandomField(cov,KarhunenLoeve(100),pts1,pts2,pts3)
@test typeof(grf) == GaussianRandomField{SeparableCovarianceFunction{3,Vector{GaussianRandomFields.CovarianceStructure{Float64}}},KarhunenLoeve{100}}
@test size(sample(grf),1) == length(pts1)
@test size(sample(grf),2) == length(pts2)
@test size(sample(grf),3) == length(pts3)

verbose && println("done")
