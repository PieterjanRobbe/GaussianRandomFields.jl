## test_covariance_functions.jl : tests for covariance_functions.jl

## Matérn ##
verbose && print("testing Mat\u00E9rn covariance structure...")

m = Matern(0.1,2.5,1)
@test typeof(m) <: Matern
m = Matern(1,2.5,1)
@test typeof(m) <: Matern
m = Matern(0.1,2,1)
@test typeof(m) <: Matern
m = Matern(1//10,2,1)
@test typeof(m) <: Matern
m = Matern(0.1,2,1.)
@test typeof(m) <: Matern

@test_throws ArgumentError Matern(-0.1,2.5,1)
@test_throws ArgumentError Matern(0.1,-2.5,1)
@test_throws ArgumentError Matern(0.1,2.5,-1)

verbose && println("done")

## Exponential ##
verbose && print("testing exponential covariance structure...")

e = Exponential(0.1,1)
@test typeof(e) <: Exponential
e = Exponential(1,1)
@test typeof(e) <: Exponential
e = Exponential(1,1.)
@test typeof(e) <: Exponential
e = Exponential(.1,1.)
@test typeof(e) <: Exponential

@test_throws ArgumentError Exponential(-0.1,1)
@test_throws ArgumentError Exponential(0.1,-2)

verbose && println("done")

## SquaredExponential ##
verbose && print("testing squared exponential covariance structure...")

s = SquaredExponential(0.1,1)
@test typeof(s) <: SquaredExponential
s = SquaredExponential(1,1)
@test typeof(s) <: SquaredExponential
s = SquaredExponential(1,1.)
@test typeof(s) <: SquaredExponential
s = SquaredExponential(.1,1.)
@test typeof(s) <: SquaredExponential

@test_throws ArgumentError SquaredExponential(-0.1,1)
@test_throws ArgumentError SquaredExponential(0.1,-2)

verbose && println("done")

## CovarianceFunction ##
verbose && print("testing covariance function...")

m = Matern(0.1,0.5,2)
c = CovarianceFunction(2,m)
@test typeof(c) <: CovarianceFunction

e = Exponential(0.1,2)
c = CovarianceFunction(1,e)
@test typeof(c) <: CovarianceFunction

s = SquaredExponential(0.1,2)
c = CovarianceFunction(2,s)
@test typeof(c) <: CovarianceFunction

@test_throws ArgumentError CovarianceFunction(0,m)
@test_throws ArgumentError CovarianceFunction(-1,e)

verbose && println("done")

## SeparableCovarianceFunction ##
verbose && print("testing separable covariance function...")

m1 = Matern(0.1,0.5,2)
m2 = Matern(0.01,2.,2)
m3 = Matern(0.1,1.0,2)
s = SeparableCovarianceFunction(m1,m2,m3)
@test typeof(s) <: SeparableCovarianceFunction
s = SeparableCovarianceFunction([m1,m2,m3])
@test typeof(s) <: SeparableCovarianceFunction
s = SeparableCovarianceFunction(m,e)
@test typeof(s) <: SeparableCovarianceFunction

verbose && println("done")
