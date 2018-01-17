using GaussianRandomFields
#=
p = readdlm("../data/star.p")
t = readdlm("../data/star.t",Int64)
@show size(p)
@show size(t)
m = Matern(0.75,2.0)
c = CovarianceFunction(2,m)
g = GaussianRandomField(c,KarhunenLoeve(300),p,t,quad=Trapezoidal())
#plot_trisurf(g); show()
tricontourf(g); show()
plot_eigenvalues(g); show()
#for i = 1:25
#    plot_eigenfunction(g,i); show()
#end
=#


exp1 = Exponential(0.5,p=1)
exp2 = Exponential(0.5,p=1)
#exp1 = Exponential(0.5)
#exp2 = Exponential(0.5)
scov = SeparableCovarianceFunction(exp1,exp2)
grf = GaussianRandomField(scov,KarhunenLoeve(300),0:0.01:1,0:0.01:1)
@show grf
#plot_eigenvalues(grf); show()
#for i = 1:25
#plot_eigenfunction(grf,i); show()
#end
#
#contourf(grf); show()
plot_eigenvalues(grf); show()
for i = 1:5
	plot(grf); show()
end
#

#=
exp = Exponential(0.5)
cov = CovarianceFunction(3,exp)
grf = GaussianRandomField(cov,KarhunenLoeve(300),0:0.05:0.5,0:0.05:2,0:0.05:1)
#=
for i = 1:25
	plot_eigenfunction(grf,i); show()
end
=#
#plot_eigenvalues(grf); show()
#for i = 1:5
	plot(grf); show()
#end
=#
