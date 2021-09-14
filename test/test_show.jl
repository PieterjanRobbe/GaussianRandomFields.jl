## test_show.jl : test show commands

@info "Testing display commands"

str = string(Matern(1,1))
@test str isa String
str = string(Exponential(1))
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

@info " Done testing display commands"
