## test_plot.jl : test plotting commands

@testset "Random field plotting    " begin

    ## 1D ##
    m = Matern(0.75,1,p=1.)
    cov = CovarianceFunction(1,m)
    pts = 0:0.01:1
    grf1 = GaussianRandomField(cov,KarhunenLoeve(500),pts)
    plot(grf1)
    plot_eigenvalues(grf1)
    plot_eigenfunction(grf1, 3)
    plot_eigenfunction(grf1, 25)
    plot_covariance_matrix(grf1)
    grf2 = GaussianRandomField(CovarianceFunction(1,Exponential(0.1)),CirculantEmbedding(),pts)
    plot(grf2)
    plot_eigenvalues(grf2)
    # plot_eigenfunction(grf2, 3)
    plot_covariance_matrix(grf2)
    plot(pts, m) # for #48

    ## 2D ##
    e = Exponential(4)
    cov = CovarianceFunction(2,e)
    pts = 0:0.05:1
    grf3 = GaussianRandomField(cov,KarhunenLoeve(10),pts,pts)
    surface(grf3)
    heatmap(grf3)
    contourf(grf3)
    contour(grf3)
    plot_eigenvalues(grf3)
    plot_eigenfunction(grf3, 2)
    plot_eigenfunction(grf3, 10)

    ## Separable ##
    e1 = Exponential(0.1, p=1)
    sp1 = SeparableCovarianceFunction(e1)
    pts = 0:0.01:1
    sg1 = GaussianRandomField(sp1,KarhunenLoeve(10),pts)
    plot_eigenvalues(sg1)
    plot_eigenfunction(sg1, 1)
    plot_eigenfunction(sg1, 4)
    plot_eigenfunction(sg1, 7)

    e2 = Exponential(0.01, p=1)
    sp2 = SeparableCovarianceFunction(e1,e2)
    sg2 = GaussianRandomField(sp2,KarhunenLoeve(10),pts,pts)
    plot_eigenvalues(sg2)
    plot_eigenfunction(sg2, 2)
    plot_eigenfunction(sg2, 5)
    plot_eigenfunction(sg2, 8)

end
