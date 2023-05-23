## test_gaussian_random_fields.jl : test implementation of GRF constructors

@testset "Gaussian random fields   " begin

    # vectors instead of linspaces
    cov = CovarianceFunction(2,Exponential(0.1))
    pts1 = collect(0:0.05:1)
    pts2 = collect(0:0.01:1)
    grf = GaussianRandomField(cov,KarhunenLoeve(500),pts1,pts2)
    @test isa(grf,GaussianRandomField)
    @test isa(grf.cov,CovarianceFunction)
    @test isa(grf.cov.cov,Exponential)
    @test ndims(grf.cov) == 2
    @test isa(grf,GaussianRandomField{KarhunenLoeve{500}})
    @test length(grf.pts) == 2
    @test length(grf.pts[1]) == length(pts1)
    @test length(grf.pts[2]) == length(pts2)
    @test size(sample(grf),1) == length(pts1)
    @test size(sample(grf),2) == length(pts2)

    # test non-equidistant grid
    cov = CovarianceFunction(2,Matern(0.3,1.))
    pts1 = 0.5 * (cos.((0:0.05:1)*pi) .+ 1)
    pts2 = 0.5 * (cos.((0:0.05:1)*pi) .+ 1)
    grf = GaussianRandomField(cov,Cholesky(),pts1,pts2)
    @test isa(grf,GaussianRandomField)
    @test isa(grf.cov,CovarianceFunction)
    @test isa(grf.cov.cov,Matern)
    @test ndims(grf.cov) == 2
    @test isa(grf,GaussianRandomField{Cholesky})
    @test length(grf.pts) == 2
    @test length(grf.pts[1]) == length(pts1)
    @test length(grf.pts[2]) == length(pts2)
    @test size(sample(grf),1) == length(pts1)
    @test size(sample(grf),2) == length(pts2)

    # test wrong dimension of points
    cov = CovarianceFunction(2,Exponential(0.3))
    pts = 0:0.1:1
    @test_throws MethodError GaussianRandomField(cov, Cholesky(), pts)

    # test sample with random number vector (and wrong length)
    cov = CovarianceFunction(1,Matern(1.0,2.,σ=1.))
    pts = 0:0.01:1
    grf = GaussianRandomField(cov,Cholesky(),pts)
    @test length(sample(grf,xi=rand(length(pts)))) == length(pts)
    @test_throws DimensionMismatch sample(grf,xi=rand(length(pts)+1))

    # test supplying a mean value for the field
    cov = CovarianceFunction(2,Exponential(1))
    pts1 = 0:0.05:1
    pts2 = 0:0.05:1
    grf = GaussianRandomField(1,cov,Cholesky(),pts1,pts2)
    @test isa(grf,GaussianRandomField)
    @test isa(grf.cov,CovarianceFunction)
    @test isa(grf.cov.cov,Exponential)
    @test ndims(grf.cov) == 2
    @test isa(grf,GaussianRandomField{Cholesky})
    @test length(grf.pts) == 2
    @test length(grf.pts[1]) == length(pts1)
    @test length(grf.pts[2]) == length(pts2)
    @test size(sample(grf),1) == length(pts1)
    @test size(sample(grf),2) == length(pts2)
    @test size(grf.mean,1) == length(pts1)
    @test size(grf.mean,2) == length(pts2)
    @test all(grf.mean.==1)

    grf = GaussianRandomField(2.0*ones(length(pts1),length(pts2)),cov,Cholesky(),pts1,pts2)
    @test isa(grf,GaussianRandomField)
    @test isa(grf.cov,CovarianceFunction)
    @test isa(grf.cov.cov,Exponential)
    @test ndims(grf.cov) == 2
    @test isa(grf,GaussianRandomField{Cholesky})
    @test length(grf.pts) == 2
    @test length(grf.pts[1]) == length(pts1)
    @test length(grf.pts[2]) == length(pts2)
    @test size(sample(grf),1) == length(pts1)
    @test size(sample(grf),2) == length(pts2)
    @test size(grf.mean,1) == length(pts1)
    @test size(grf.mean,2) == length(pts2)
    @test all(grf.mean.==2.)
    @test_throws DimensionMismatch GaussianRandomField(2.0*ones(5,10),cov,Cholesky(),pts1,pts2)

    # test custom RNG option
    @test sample(MersenneTwister(1234), grf)[3, 7] ≈ 1.35910824971202
    @test sample(MersenneTwister(1234), grf, 2)[1][3, 7] ≈ 1.35910824971202
    @test sample(MersenneTwister(1234), grf, 2)[2][3, 7] ≈ 2.2485431735206025
    @test_throws DomainError sample(MersenneTwister(1234), grf, 0)
end
