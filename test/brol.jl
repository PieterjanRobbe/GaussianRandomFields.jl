using GaussianRandomFields

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
