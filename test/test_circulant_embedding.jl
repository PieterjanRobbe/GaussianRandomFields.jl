## test_circulant_embedding.jl : test circulant embedding for GRF generation

@testset "circulant embedding        " begin

## 1d Exponential ##
cov = CovarianceFunction(1,Exponential(0.1))
pts = 0:1//1001:1
grf = GaussianRandomField(cov,CirculantEmbedding(),pts)
@test isa(grf,GaussianRandomField)
@test isa(grf.cov,CovarianceFunction)
@test isa(grf.cov.cov,Exponential)
@test ndims(grf.cov) == 1
@test isa(grf,GaussianRandomField{C,CirculantEmbedding} where {C})
@test length(grf.pts[1]) == length(pts)
@test length(sample(grf)) == length(pts)

## 1d Matern with padding ##
m = CovarianceFunction(1,Matern(1.0,1.0))
grf = GaussianRandomField(m,CirculantEmbedding(),pts,padding=8)
@test isa(grf,GaussianRandomField)
@test isa(grf.cov,CovarianceFunction)
@test isa(grf.cov.cov,Matern)
@test ndims(grf.cov) == 1
@test isa(grf,GaussianRandomField{C,CirculantEmbedding} where {C})
@test length(grf.pts[1]) == length(pts)
@test length(sample(grf)) == length(pts)

## 2d Anisotropic ##
A = [1000 0; 0 1000]
a = AnisotropicExponential(A)
c = CovarianceFunction(2,a)
pts1 = linspace(0,1,128)
pts2 = linspace(0,1,256)
@suppress grf = GaussianRandomField(c,CirculantEmbedding(),pts1,pts2,padding=4)
@test isa(grf,GaussianRandomField)
@test isa(grf.cov,CovarianceFunction)
@test isa(grf.cov.cov,AnisotropicExponential)
@test ndims(grf.cov) == 2
@test isa(grf,GaussianRandomField{C,CirculantEmbedding} where {C})
@test length(grf.pts) == 2
@test length(grf.pts[1]) == length(pts1)
@test length(grf.pts[2]) == length(pts2)
@test size(sample(grf),1) == length(pts1)
@test size(sample(grf),2) == length(pts2)

## 3d AnisotropicExponential ##
A = [100 0 0; 0 100 0; 0 0 100]
a = AnisotropicExponential(A)
c = CovarianceFunction(3,a)
pts = 0:0.025:1
@suppress grf = GaussianRandomField(c,CirculantEmbedding(),pts,pts,pts,padding=1,measure=false)
@test isa(grf,GaussianRandomField)
@test isa(grf.cov,CovarianceFunction)
@test isa(grf.cov.cov,AnisotropicExponential)
@test ndims(grf.cov) == 3
@test isa(grf,GaussianRandomField{C,CirculantEmbedding} where {C})
@test length(grf.pts) == 3
@test length(grf.pts[1]) == length(pts)
@test length(grf.pts[2]) == length(pts)
@test length(grf.pts[3]) == length(pts)
@test size(sample(grf),1) == length(pts)
@test size(sample(grf),2) == length(pts)
@test size(sample(grf),3) == length(pts)

## Larger domain ##
A = [1 0.8; 0.8 1]
m = AnisotropicExponential(A)
c = CovarianceFunction(2,m)
pts1 = linspace(-5,5,128)
pts2 = linspace(10,0,128)
@suppress grf = GaussianRandomField(c,CirculantEmbedding(),pts1,pts2,padding=2)
@test isa(grf,GaussianRandomField)
@test isa(grf.cov,CovarianceFunction)
@test isa(grf.cov.cov,AnisotropicExponential)
@test ndims(grf.cov) == 2
@test isa(grf,GaussianRandomField{C,CirculantEmbedding} where {C})
@test length(grf.pts) == 2
@test length(grf.pts[1]) == length(pts1)
@test length(grf.pts[2]) == length(pts2)
@test size(sample(grf),1) == length(pts1)
@test size(sample(grf),2) == length(pts2)

## Dedicated plotting commands ##
m = Matern(0.5,2)
c = CovarianceFunction(3,m)
pts1 = linspace(0,1,16)
pts2 = linspace(0,1,8)
pts3 = linspace(0,1,4)
plot_covariance_matrix(c,pts1,pts2,pts3)
m = Matern(0.1,1)
c = CovarianceFunction(2,m)
pts = linspace(0,1,512)
g = GaussianRandomField(c,CirculantEmbedding(),pts,pts,padding=4)
contourf(g)
plot_eigenvalues(g)

end
