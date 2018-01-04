using GaussianRandomFields
using PyPlot

#
m = Matern(0.3,2.0,2)
c = CovarianceFunction(3,m)
pts1 = 0:0.075:1
pts2 = 0:0.075:1
pts3 = 0:0.075:1
#pts1 = 0:0.1:1
#pts2 = 0:0.1:1
#pts3 = 0:0.1:1
nx = length(pts1)
nx2 = round(Int,nx/2)
ny = length(pts2)
ny2 = round(Int,ny/2)
nz = length(pts3)
nz2 = round(Int,nz/2)
grf = GaussianRandomField(c,Spectral(),pts1,pts2,pts3)

plot(pts1,pts2,pts3,nx2,ny2,nz2,grf,edgecolor="white",linewidth=1.0)
plt[:show]()
#
#=
m1 = Matern(1,2.0,2)
m2 = Matern(0.3,2.0,2)
c = SeparableCovarianceFunction(m1,m2)
pts1 = 0:0.05:1
pts2 = 0:0.05:1
grf = GaussianRandomField(c,Spectral(),pts1,pts2)
=#
