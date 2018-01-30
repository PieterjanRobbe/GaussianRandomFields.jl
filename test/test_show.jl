## test_show.jl : test show commands

info("Testing display commands")

@show Matern(1,1)
@show Exponential(1)
@show SquaredExponential(1)
@show AnisotropicExponential([1000 0; 0 1000])
@show CovarianceFunction(2,Matern(1.0,2))
@show Cholesky()
@show Spectral()
@show KarhunenLoeve(10)
@show CirculantEmbedding()
@show GaussianRandomField(CovarianceFunction(1,Exponential(1)),Cholesky(),0:0.1:1)
@show SeparableCovarianceFunction(Exponential(1),Exponential(1))

info(" Done testing display commands")
