## test_separable.jl : test separable kernels for GRF generation

@testset "separable GRFs           " begin

## 1d Exponential ##
cov = SeparableCovarianceFunction(Exponential(0.1))
pts = linspace(0,1,1001)
grf = GaussianRandomField(cov,KarhunenLoeve(1000),pts)
@test isa(grf,GaussianRandomField)
@test isa(grf.cov,SeparableCovarianceFunction)
@test isa(grf.cov.cov,Vector)
@test isa(grf.cov.cov[1],Exponential)
@test ndims(grf.cov) == 1
@test isa(grf,GaussianRandomField{C,KarhunenLoeve{1000}} where {C})
@test length(grf.pts[1]) == length(pts)
@test length(sample(grf)) == length(pts)

## 2d Exponential ##
e1 = Exponential(0.1)
e2 = Exponential(0.01)
cov = SeparableCovarianceFunction(e1,e2)
pts1 = 0:0.05:1
pts2 = 0:0.05:1
grf = GaussianRandomField(cov,KarhunenLoeve(500),pts1,pts2)
@test isa(grf,GaussianRandomField)
@test isa(grf.cov,SeparableCovarianceFunction)
@test isa(grf.cov.cov,Vector)
@test isa(grf.cov.cov[1],Exponential)
@test isa(grf.cov.cov[2],Exponential)
@test ndims(grf.cov) == 2
@test isa(grf,GaussianRandomField{C,KarhunenLoeve{500}} where {C})
@test length(grf.pts) == 2
@test length(grf.pts[1]) == length(pts1)
@test length(grf.pts[2]) == length(pts2)
@test size(sample(grf),1) == length(pts1)
@test size(sample(grf),2) == length(pts2)

## 2d Exponential ##
e1 = Exponential(0.1)
e2 = Exponential(0.01)
cov = SeparableCovarianceFunction(e1,e2)
pts1 = 0:0.05:1
pts2 = 0:0.05:1
grf = GaussianRandomField(cov,KarhunenLoeve(500),pts1,pts2)
@test isa(grf,GaussianRandomField)
@test isa(grf.cov,SeparableCovarianceFunction)
@test isa(grf.cov.cov,Vector)
@test isa(grf.cov.cov[1],Exponential)
@test isa(grf.cov.cov[2],Exponential)
@test ndims(grf.cov) == 2
@test isa(grf,GaussianRandomField{C,KarhunenLoeve{500}} where {C})
@test length(grf.pts) == 2
@test length(grf.pts[1]) == length(pts1)
@test length(grf.pts[2]) == length(pts2)
@test size(sample(grf),1) == length(pts1)
@test size(sample(grf),2) == length(pts2)

# matern/exponential 2d
m1 = Matern(0.2,1.0)
e1 = Exponential(0.01)
cov = SeparableCovarianceFunction([m1,e1])
pts1 = 0:0.05:1
pts2 = 0:0.05:1
grf = GaussianRandomField(cov,KarhunenLoeve(250),pts1,pts2,quad=EOLE())
@test isa(grf,GaussianRandomField)
@test isa(grf.cov,SeparableCovarianceFunction)
@test isa(grf.cov.cov,Vector)
@test isa(grf.cov.cov[1],Matern)
@test isa(grf.cov.cov[2],Exponential)
@test ndims(grf.cov) == 2
@test isa(grf,GaussianRandomField{C,KarhunenLoeve{250}} where {C})
@test length(grf.pts) == 2
@test length(grf.pts[1]) == length(pts1)
@test length(grf.pts[2]) == length(pts2)
@test size(sample(grf),1) == length(pts1)
@test size(sample(grf),2) == length(pts2)

## 3d separable Matern ##
m1 = Matern(0.2,1.0)
m2 = Matern(0.4,3.0)
m3 = Matern(0.6,4.0)
cov = SeparableCovarianceFunction(m1,m1,m2)
pts1 = 0:0.1:1
pts2 = 0:0.1:1
pts3 = 0:0.1:1
grf = GaussianRandomField(cov,KarhunenLoeve(100),pts1,pts2,pts3)
@test isa(grf,GaussianRandomField)
@test isa(grf.cov,SeparableCovarianceFunction)
@test isa(grf.cov.cov,Vector)
@test isa(grf.cov.cov[1],Matern)
@test isa(grf.cov.cov[2],Matern)
@test isa(grf.cov.cov[3],Matern)
@test ndims(grf.cov) == 3
@test isa(grf,GaussianRandomField{C,KarhunenLoeve{100}} where {C})
@test length(grf.pts) == 3
@test length(grf.pts[1]) == length(pts1)
@test length(grf.pts[2]) == length(pts2)
@test length(grf.pts[3]) == length(pts3)
@test size(sample(grf),1) == length(pts1)
@test size(sample(grf),2) == length(pts2)
@test size(sample(grf),3) == length(pts3)

end
