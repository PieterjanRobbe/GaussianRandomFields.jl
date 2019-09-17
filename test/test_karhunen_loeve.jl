## test_karhunen_loeve.jl : test Karhunen-Lo\`eve expansion for GRF generation

@testset "Karhunen-Lo\u00e8ve expansion " begin

    ## 1d Mat\'ern ##
    cov = CovarianceFunction(1,Matern(0.3,1))
    pts = 0:0.01:1
    grf = GaussianRandomField(cov,KarhunenLoeve(500),pts)
    @test isa(grf,GaussianRandomField)
    @test isa(grf.cov,CovarianceFunction)
    @test isa(grf.cov.cov,Matern)
    @test ndims(grf.cov) == 1
    @test isa(grf,GaussianRandomField{KarhunenLoeve{500}})
    @test length(grf.pts[1]) == length(pts)
    @test length(sample(grf)) == length(pts)

    ## test anisotropic
    cov = CovarianceFunction(2,AnisotropicExponential(100*[1 0.2; 0.2 1]))
    pts1 = range(0,stop = 1,length = 128)
    pts2 = range(0,stop = 1,length = 128)
    grf = GaussianRandomField(cov,KarhunenLoeve(1000),pts1,pts2)
    @test isa(grf,GaussianRandomField)
    @test isa(grf.cov,CovarianceFunction)
    @test isa(grf.cov.cov,AnisotropicExponential)
    @test ndims(grf.cov) == 2
    @test isa(grf,GaussianRandomField{KarhunenLoeve{1000}})
    @test length(grf.pts) == 2
    @test length(grf.pts[1]) == length(pts1)
    @test length(grf.pts[2]) == length(pts2)
    @test size(sample(grf),1) == length(pts1)
    @test size(sample(grf),2) == length(pts2)

    ## variation of number of KL terms ##
    @test_throws DomainError KarhunenLoeve{0}()
    @test_throws DomainError KarhunenLoeve(0)

    cov = CovarianceFunction(2,Matern(0.3,1))
    pts = 0:0.01:1
    grf = GaussianRandomField(cov,KarhunenLoeve(17),pts,pts, eigensolver=EigenSolver())
    @test isa(grf,GaussianRandomField)
    @test isa(grf.cov,CovarianceFunction)
    @test isa(grf.cov.cov,Matern)
    @test ndims(grf.cov) == 2
    @test isa(grf,GaussianRandomField{KarhunenLoeve{17}})
    @test length(grf.pts[1]) == length(pts)
    @test length(grf.pts[2]) == length(pts)
    @test size(sample(grf),1) == length(pts)
    @test size(sample(grf),2) == length(pts)

    cov = CovarianceFunction(2,Matern(0.3,1))
    pts = 0:0.01:1
    grf = GaussianRandomField(cov,KarhunenLoeve(1),pts,pts, eigensolver=EigenSolver())
    @test isa(grf,GaussianRandomField)
    @test isa(grf.cov,CovarianceFunction)
    @test isa(grf.cov.cov,Matern)
    @test ndims(grf.cov) == 2
    @test isa(grf,GaussianRandomField{KarhunenLoeve{1}})
    @test length(grf.pts[1]) == length(pts)
    @test length(grf.pts[2]) == length(pts)
    @test size(sample(grf),1) == length(pts)
    @test size(sample(grf),2) == length(pts)

    mat = Matern(0.1,1.0)
    cov = CovarianceFunction(1,mat)
    pts1 = 0:0.01:1
    nterms = [1 2 5 10 20 50 100 200 500 1000]
    err = Float64[]
    for n in nterms
        grf = GaussianRandomField(cov,KarhunenLoeve(n),pts1)
        push!(err,rel_error(grf))
    end
    @test err[1]/err[end] > 1e7

    ## quadrature methods ##
    mat = Matern(0.1,1.0)
    cov = CovarianceFunction(1,mat)
    pts1 = 0:0.025:1
    grf = GaussianRandomField(cov,KarhunenLoeve(1000),pts1,quad=GaussLegendre())
    @test rel_error(grf) < 1e-6
    grf = GaussianRandomField(cov,KarhunenLoeve(1000),pts1,quad=EOLE())
    @test rel_error(grf) < 1e-6
    @suppress grf = GaussianRandomField(cov,KarhunenLoeve(1000),pts1,quad=Simpson())
    @test rel_error(grf) < 1e-6
    grf = GaussianRandomField(cov,KarhunenLoeve(1000),pts1,quad=Midpoint())
    @test rel_error(grf) < 1e-6
    grf = GaussianRandomField(cov,KarhunenLoeve(1000),pts1,quad=Trapezoidal())
    @test rel_error(grf) < 1e-6

    ## test 3D Gaussian ##
    cov = CovarianceFunction(3,Gaussian(1.0))
    pts1 = 0:0.05:1
    pts2 = 0:0.05:1
    pts3 = 0:0.05:1
    grf = GaussianRandomField(cov,KarhunenLoeve(103),pts1,pts2,pts3)
    @test isa(grf,GaussianRandomField)
    @test isa(grf.cov,CovarianceFunction)
    @test isa(grf.cov.cov,Gaussian)
    @test ndims(grf.cov) == 3
    @test isa(grf,GaussianRandomField{KarhunenLoeve{103}})
    @test length(grf.pts) == 3
    @test length(grf.pts[1]) == length(pts1)
    @test length(grf.pts[2]) == length(pts2)
    @test length(grf.pts[3]) == length(pts3)
    @test size(sample(grf),1) == length(pts1)
    @test size(sample(grf),2) == length(pts2)
    @test size(sample(grf),3) == length(pts3)

    ## variation of number of points ##
    cov = CovarianceFunction(1,Matern(1,1))
    pts = 0:0.1:0
    @test_throws ArgumentError grf = GaussianRandomField(cov,KarhunenLoeve(300),pts)
    pts = 0:0.1:0.1
    grf = GaussianRandomField(cov,KarhunenLoeve(300),pts)
    pts = 1:-0.1:0
    grf = GaussianRandomField(cov,KarhunenLoeve(300),pts)
    grf = GaussianRandomField(cov,KarhunenLoeve(300),pts,nq=1000)
    @test_throws ArgumentError GaussianRandomField(cov,KarhunenLoeve(300),pts,nq=100)

    ## non-SPD covariance matrix ##
    cov = CovarianceFunction(2,SquaredExponential(0.5))
    @suppress grf = GaussianRandomField(cov,KarhunenLoeve(200),pts1,pts2)
    @test isa(grf,GaussianRandomField)
    @test isa(grf.cov,CovarianceFunction)
    @test isa(grf.cov.cov,SquaredExponential)
    @test ndims(grf.cov) == 2
    @test isa(grf,GaussianRandomField{<:KarhunenLoeve})
    @test length(grf.pts[1]) == length(pts1)
    @test length(grf.pts[2]) == length(pts2)
    @test size(sample(grf),1) == length(pts1)
    @test size(sample(grf),2) == length(pts2)

    ## test domain with negative sides ##
    cov = CovarianceFunction(2,Matern(1.,2.5,σ=1.,p=2))
    pts1 = collect(-2:0.1:0)
    pts2 = collect(-1:0.1:1)
    grf = GaussianRandomField(cov,KarhunenLoeve(458),pts1,pts2)
    @test isa(grf,GaussianRandomField)
    @test isa(grf.cov,CovarianceFunction)
    @test isa(grf.cov.cov,Matern)
    @test ndims(grf.cov) == 2
    @test isa(grf,GaussianRandomField{KarhunenLoeve{458}})
    @test length(grf.pts[1]) == length(pts1)
    @test length(grf.pts[2]) == length(pts2)
    @test size(sample(grf),1) == length(pts1)
    @test size(sample(grf),2) == length(pts2)
end
