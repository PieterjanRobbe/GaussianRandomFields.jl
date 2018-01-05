using GaussianRandomFields

p = readdlm("../data/star.p")
t = readdlm("../data/star.t",Int64)
@show size(p)
@show size(t)
m = Matern(0.75,2.0)
c = CovarianceFunction(2,m)
g = GaussianRandomField(c,Spectral(),p,t)
plot_trisurf(g); show()
tricontourf(g); show()
