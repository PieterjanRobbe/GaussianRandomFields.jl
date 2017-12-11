using GaussianRandomFields
using PyPlot

m = Matern(0.3,2.0,2)
c = CovarianceFunction(2,m)
pts1 = 0:0.05:1
pts2 = 0:0.05:1
grf = GaussianRandomField(c,Spectral(),pts1,pts2)

contour(pts1,pts2,grf)
plt[:show]()
