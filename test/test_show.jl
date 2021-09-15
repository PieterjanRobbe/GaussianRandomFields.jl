## test_show.jl : test show commands

@testset "display commands         " begin

    str = string(Matern(1,1))
    @test str isa String
    str = string(Exponential(1))
    @test str isa String
    str = string(Whittle(1))
    @test str isa String
    str = string(Linear(1))
    @test str isa String
    str = string(Spherical(1))
    @test str isa String
    str = string(SquaredExponential(1))
    @test str isa String
    str = string(AnisotropicExponential([1000 0; 0 1000]))
    @test str isa String
    str = string(CovarianceFunction(2,Matern(1.0,2)))
    @test str isa String
    str = string(Cholesky())
    @test str isa String
    str = string(Spectral())
    @test str isa String
    str = string(KarhunenLoeve(10))
    @test str isa String
    str = string(CirculantEmbedding())
    @test str isa String
    str = string(GaussianRandomField(CovarianceFunction(1,Exponential(1)),Cholesky(),0:0.1:1))
    @test str isa String
    str = string(SeparableCovarianceFunction(Exponential(1),Exponential(1)))
    @test str isa String
    str = string(GaussianRandomField(CovarianceFunction(2,Exponential(0.1)),Cholesky(),0:0.1:1,0:0.1:1))
    @test str isa String
    str = string(GaussianRandomField(CovarianceFunction(2,Exponential(0.1)),KarhunenLoeve(10),rand(25,2)))
    @test str isa String
    str = string(GaussianRandomField(CovarianceFunction(2,Exponential(0.1)),KarhunenLoeve(10),rand(3,2),collect([1,2,3]')))
    @test str isa String

end
