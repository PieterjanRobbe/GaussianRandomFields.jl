## test_fem.jl : test GRF defined on FEM

@testset "unstructured grid        " begin

    ## Generate a set of random points ##
    pts = rand(512, 2)
    m = Matern(0.2,2.0)

    ## Cholesky ##
    grf = GaussianRandomField(CovarianceFunction(2,m),Cholesky(),pts)
    @test isa(grf,GaussianRandomField)
    @test isa(grf.cov,CovarianceFunction)
    @test isa(grf.cov.cov,Matern)
    @test ndims(grf.cov) == 2
    @test isa(grf,GaussianRandomField{Cholesky})
    @test length(grf.pts) == 2
    @test size(grf.pts[1],1) == 2
    @test size(grf.pts[1],2) == size(pts,1)

    ## Spectral ##
    grf = GaussianRandomField(CovarianceFunction(2,m),Spectral(),pts)
    @test isa(grf,GaussianRandomField{Spectral})

    ## CirculantEmbedding ##
    @test_throws ArgumentError GaussianRandomField(CovarianceFunction(2,m),CirculantEmbedding(),pts)

    ## KarhunenLoeve ##
    grf = GaussianRandomField(CovarianceFunction(2,m),KarhunenLoeve(128),pts)
    @test isa(grf,GaussianRandomField{KarhunenLoeve{128}})

    ## Anisotropic random field ##
    a = AnisotropicExponential([1000 0; 0 1000])
    cov = CovarianceFunction(2,a)
    grf = GaussianRandomField(cov,KarhunenLoeve(500),pts,quad=GaussLegendre())
    @test isa(grf.cov.cov,AnisotropicExponential)

    ## GaussianRandomField with mean value ##
    grf = GaussianRandomField(1,CovarianceFunction(2,m),Cholesky(),pts)
    @test isa(grf,GaussianRandomField{Cholesky})
    @test length(grf.mean) == size(pts,1)
    @test all(grf.mean.==1)

    grf = GaussianRandomField(ones(size(pts,1)),CovarianceFunction(2,m),Spectral(),pts)
    @test isa(grf,GaussianRandomField{Spectral})
    @test length(grf.mean) == size(pts,1)
    @test all(grf.mean.==1)

    ## Test spherical domain ##
    pts = 4*rand(1024, 3) .- 2
    d = pts[:,1].*pts[:,1] + pts[:,2].*pts[:,2] + pts[:,3].*pts[:,3]
    pts = pts[d .< 4, :]
    cov = CovarianceFunction(3, Matern(1, 3/2))
    grf = GaussianRandomField(cov, KarhunenLoeve(128), pts)
    z = sample(grf)
    @test isa(grf,GaussianRandomField{KarhunenLoeve{128}})
    grf = GaussianRandomField(cov, Spectral(), pts)
    @test isa(grf,GaussianRandomField{Spectral})
    grf = GaussianRandomField(cov, Cholesky(), pts)
    @test isa(grf,GaussianRandomField{Cholesky})

end
