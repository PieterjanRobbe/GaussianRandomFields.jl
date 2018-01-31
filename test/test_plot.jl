## test_plot.jl : test plotting commands

## 1D ##
m = Matern(0.75,1,p=1.)
cov = CovarianceFunction(1,m)
pts = 0:0.01:1
grf1 = GaussianRandomField(cov,KarhunenLoeve(500),pts)
plot(grf1,n=10); close()

## 2D ##
e = Exponential(0.01)
cov = CovarianceFunction(2,e)
pts = 0:0.05:1
grf2 = GaussianRandomField(cov,KarhunenLoeve(10),pts,pts)
plot(grf2)
surf(grf2)
contourf(grf2)
contour(grf2)

## 3D ##
m = Exponential(0.5)
cov = CovarianceFunction(3,m)
pts = 0:0.1:1
grf3 = GaussianRandomField(cov,KarhunenLoeve(10),pts,pts,pts)
plot(grf3); close()
plot(grf3,ix=1); close()

## Separable ##
e1 = Exponential(0.1,p=1)
e2 = Exponential(0.01,p=1)
e3 = Exponential(0.001,p=1)
sp1 = SeparableCovarianceFunction(e1)
sp2 = SeparableCovarianceFunction(e1,e2)
sp3 = SeparableCovarianceFunction(e1,e2,e3)
pts = 0:0.01:1
sg1 = GaussianRandomField(sp1,KarhunenLoeve(10),pts)
sg2 = GaussianRandomField(sp2,KarhunenLoeve(10),pts,pts)
sg3 = GaussianRandomField(sp3,KarhunenLoeve(10),pts,pts,pts)

## Eigenvalues/functions ##
plot_eigenvalues(grf1); close()
plot_eigenvalues(grf2); close()
plot_eigenvalues(sg2); close()
plot_eigenfunction(grf1,1); close()
plot_eigenfunction(grf2,1); close()
plot_eigenfunction(grf3,1); close()
plot_eigenfunction(sg1,1); close()
plot_eigenfunction(sg2,1); close()
plot_eigenfunction(sg3,1); close()
