## test_circulant_embedding.jl : test circulant embedding for GRF generation

@testset "circulant embedding      " begin

    ## 1d Exponential ##
    cov = CovarianceFunction(1,Exponential(0.1))
    pts = 0:1//1001:1
    grf = GaussianRandomField(cov,CirculantEmbedding(),pts)
    @test isa(grf,GaussianRandomField)
    @test isa(grf.cov,CovarianceFunction)
    @test isa(grf.cov.cov,Exponential)
    @test ndims(grf.cov) == 1
    @test isa(grf,GaussianRandomField{CirculantEmbedding})
    @test length(grf.pts[1]) == length(pts)
    @test length(sample(grf)) == length(pts)

    ## 1d Matern with padding ##
    m = CovarianceFunction(1,Matern(1.0,1.0))
    grf = GaussianRandomField(m,CirculantEmbedding(),pts,minpadding=8000)
    @test isa(grf,GaussianRandomField)
    @test isa(grf.cov,CovarianceFunction)
    @test isa(grf.cov.cov,Matern)
    @test ndims(grf.cov) == 1
    @test isa(grf,GaussianRandomField{CirculantEmbedding})
    @test length(grf.pts[1]) == length(pts)
    @test length(sample(grf)) == length(pts)

    ## 2d Anisotropic ##
    A = [1000 0; 0 1000]
    a = AnisotropicExponential(A)
    c = CovarianceFunction(2,a)
    pts1 = range(0,stop = 1,length = 128)
    pts2 = range(0,stop = 1,length = 256)
    @suppress grf = GaussianRandomField(c,CirculantEmbedding(),pts1,pts2,minpadding=(128,256))
    @test isa(grf,GaussianRandomField)
    @test isa(grf.cov,CovarianceFunction)
    @test isa(grf.cov.cov,AnisotropicExponential)
    @test ndims(grf.cov) == 2
    @test isa(grf,GaussianRandomField{CirculantEmbedding})
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
    @suppress grf = GaussianRandomField(c,CirculantEmbedding(),pts,pts,pts,measure=false)
    @test isa(grf,GaussianRandomField)
    @test isa(grf.cov,CovarianceFunction)
    @test isa(grf.cov.cov,AnisotropicExponential)
    @test ndims(grf.cov) == 3
    @test isa(grf,GaussianRandomField{CirculantEmbedding})
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
    pts1 = range(-5,stop = 5,length = 128)
    pts2 = range(10,stop = 0,length = 128)
    @suppress grf = GaussianRandomField(c,CirculantEmbedding(),pts1,pts2,minpadding=128,primes=false)
    @test isa(grf,GaussianRandomField)
    @test isa(grf.cov,CovarianceFunction)
    @test isa(grf.cov.cov,AnisotropicExponential)
    @test ndims(grf.cov) == 2
    @test isa(grf,GaussianRandomField{CirculantEmbedding})
    @test length(grf.pts) == 2
    @test length(grf.pts[1]) == length(pts1)
    @test length(grf.pts[2]) == length(pts2)
    @test size(sample(grf),1) == length(pts1)
    @test size(sample(grf),2) == length(pts2)

    ## Dedicated plotting commands ##
    m = Matern(0.5,2)
    c = CovarianceFunction(3,m)
    pts1 = range(0,stop = 1,length = 16)
    pts2 = range(0,stop = 1,length = 8)
    pts3 = range(0,stop = 1,length = 4)
    #plot_covariance_matrix(c,pts1,pts2,pts3); close()
    m = Matern(0.1,1)
    c = CovarianceFunction(2,m)
    pts = range(0,stop = 1,length = 512)
    g = GaussianRandomField(c,CirculantEmbedding(),pts,pts,minpadding=512,measure=false)
    #contourf(g); close()
    #plot_eigenvalues(g); close()

end
