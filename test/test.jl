using GaussianRandomFields
m = Matern(0.5,2.0,2)
c = CovarianceFunction(1,m)
pts = 0:0.001:1
grf = GaussianRandomField(c,Cholesky(),pts)

plot(grf,n=5)
