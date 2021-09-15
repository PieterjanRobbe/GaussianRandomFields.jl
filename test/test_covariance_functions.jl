## test_covariance_functions.jl : tests for covariance_functions.jl

## Matérn ##
@testset "Mat\u00E9rn covariance        " begin

m = Matern(0.1,2.5)
@test isa(m,Matern)
m = Matern(1,2.5)
@test isa(m,Matern)
m = Matern(0.1,2)
@test isa(m,Matern)
m = Matern(1//10,2)
@test isa(m,Matern)
m = Matern(0.1,2,p=1.)
@test isa(m,Matern)
m = Matern(0.1,2,σ=1.)
@test isa(m,Matern)
m = Matern(0.1,2,σ=2.)
@test isa(m,Matern)
m = Matern(0.1,2,σ=1.,p=2)
@test isa(m,Matern)

@test_throws DomainError Matern(-0.1,2.5)
@test_throws DomainError Matern(0.1,-2.5)
@test_throws DomainError Matern(0.1,2.5,p=-1)
@test_throws DomainError Matern(0.1,2.5,σ=-1)
@test_throws DomainError Matern(0.1,2.5,p=0.5)
@test_throws DomainError Matern(0.1,2.5,p=Inf)
end

## Exponential ##
@testset "exponential covariance   " begin

e = Exponential(0.1)
@test isa(e,Exponential)
e = Exponential(1)
@test isa(e,Exponential)
e = Exponential(1,p=1.)
@test isa(e,Exponential)
e = Exponential(.1,p=1)
@test isa(e,Exponential)
e = Exponential(.1,σ=10)
@test isa(e,Exponential)
e = Exponential(.1,σ=2.,p=1)
@test isa(e,Exponential)

@test_throws DomainError Exponential(-0.1,p=1)
@test_throws DomainError Exponential(0.1,σ=-2)
@test_throws DomainError Exponential(0.1,p=0.9)
@test_throws DomainError Exponential(0.1,p=Inf)

end

## SquaredExponential ##
@testset "Gaussian covariance      " begin

s = SquaredExponential(0.1)
@test isa(s,Gaussian)
s = Gaussian(0.1)
@test isa(s,SquaredExponential)
s = SquaredExponential(1)
@test isa(s,SquaredExponential)
s = SquaredExponential(1,p=1.)
@test isa(s,SquaredExponential)
s = SquaredExponential(.1,σ=1.)
@test isa(s,SquaredExponential)
s = SquaredExponential(.1,σ=10.,p=2)
@test isa(s,SquaredExponential)
s = SquaredExponential(.1,p=10.,σ=2)
@test isa(s,SquaredExponential)

@test_throws DomainError SquaredExponential(-0.1)
@test_throws DomainError SquaredExponential(0.1,p=0.5)
@test_throws DomainError SquaredExponential(0.1,σ=-2)
@test_throws DomainError Gaussian(0.1,p=Inf)
end

## Linear ##
@testset "linear covariance        " begin

l = Linear(0.1)
@test isa(l,Linear)
l = Linear(1)
@test isa(l,Linear)
l = Linear(1,p=1.)
@test isa(l,Linear)
l = Linear(.1,p=1)
@test isa(l,Linear)
l = Linear(.1,σ=10)
@test isa(l,Linear)
l = Linear(.1,σ=2.,p=1)
@test isa(l,Linear)

@test_throws DomainError Linear(-0.1,p=1)
@test_throws DomainError Linear(0.1,σ=-2)
@test_throws DomainError Linear(0.1,p=0.9)
@test_throws DomainError Linear(0.1,p=Inf)

end

## Spherical ##
@testset "spherical covariance     " begin

l = Spherical(0.1)
@test isa(l,Spherical)
l = Spherical(1)
@test isa(l,Spherical)
l = Spherical(1,p=1.)
@test isa(l,Spherical)
l = Spherical(.1,p=1)
@test isa(l,Spherical)
l = Spherical(.1,σ=10)
@test isa(l,Spherical)
l = Spherical(.1,σ=2.,p=1)
@test isa(l,Spherical)

@test_throws DomainError Spherical(-0.1,p=1)
@test_throws DomainError Spherical(0.1,σ=-2)
@test_throws DomainError Spherical(0.1,p=0.9)
@test_throws DomainError Spherical(0.1,p=Inf)

end

## Whittle ##
@testset "Whittle covariance       " begin

l = Whittle(0.1)
@test isa(l,Whittle)
l = Whittle(1)
@test isa(l,Whittle)
l = Whittle(1,p=1.)
@test isa(l,Whittle)
l = Whittle(.1,p=1)
@test isa(l,Whittle)
l = Whittle(.1,σ=10)
@test isa(l,Whittle)
l = Whittle(.1,σ=2.,p=1)
@test isa(l,Whittle)

@test_throws DomainError Whittle(-0.1,p=1)
@test_throws DomainError Whittle(0.1,σ=-2)
@test_throws DomainError Whittle(0.1,p=0.9)
@test_throws DomainError Whittle(0.1,p=Inf)

end

## AnisotropicExponential ##
@testset "anisotropic exponential  " begin

A = [1 0; 0 1]
s = AnisotropicExponential(A)
@test isa(s,AnisotropicExponential)
A = [1 0.8 0.8; 0.8 1 0.8; 0.8 0.8 1]
s = AnisotropicExponential(A)
@test isa(s,AnisotropicExponential)
A = [1000 0; 0 1000]
s = AnisotropicExponential(A,σ=2)
@test isa(s,AnisotropicExponential)
s = AnisotropicExponential(A,σ=2.)
@test isa(s,AnisotropicExponential)

A = [10 10; -10 10]
@test_throws DomainError AnisotropicExponential(A)
A = [10 0; 0 10]
@test_throws DomainError AnisotropicExponential(A,σ=-1)
end

## CovarianceFunction ##
@testset "covariance function      " begin

m = Matern(0.1,0.5)
c = CovarianceFunction(2,m)
@test isa(c,CovarianceFunction)

e = Exponential(0.1)
c = CovarianceFunction(1,e)
@test isa(c,CovarianceFunction)

s = SquaredExponential(0.1)
c = CovarianceFunction(2,s)
@test isa(c,CovarianceFunction)

@test_throws DomainError CovarianceFunction(0,m)
@test_throws DomainError CovarianceFunction(-1,e)
end
