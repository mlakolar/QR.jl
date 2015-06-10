addprocs(4);

@everywhere using HDF5, JLD
@everywhere using Distributions
@everywhere using QR
@everywhere using Gurobi
@everywhere using CDLasso

@everywhere function generate_data(
    rep;
    corType = 1,
    noiseType = 1
    )

  # random seed
  srand(rep)

  # generate data
  p = 1500
  n = 1000
  s = 10

  if corType == 1
    # equal correlation
    rho = 0.5
    Sigma = full(blkdiag(sparse(rho * ones(s+5,s+5) + (1-rho) * eye(s+5)), speye(p-s-5)))
  elseif corType == 2
    # toeplitz
    rho = 0.8
    Sigma = zeros(Float64, p,p)
    for a=1:p
      for b=a:p
        t = rho^abs(a-b)
        Sigma[a,b] = t
        Sigma[b,a] = t
      end
    end
  elseif corType == 3
    # toeplitz
    rho = 0.1
    Sigma = zeros(Float64, p,p)
    for a=1:p
      for b=a:p
        t = rho^abs(a-b)
        Sigma[a,b] = t
        Sigma[b,a] = t
      end
    end
  else
    error("Not implemented")
  end
  sqSigma = sqrtm(Sigma)

  beta = linspace(1., 0.5, s)
  X = randn(n, p) * sqSigma
  tX = hcat(ones(n), X)
  if noiseType == 1
    err_d = Normal()
  elseif noiseType == 2
    err_d = TDist(1)
  elseif noiseType == 3
    err_d = TDist(3)
  elseif noiseType == 4
    err_d = MixtureModel(Normal, [(0., 1.), (3., 0.4), (-3.0, 0.4)], [0.6, 0.2, 0.2])
  else
    error("Not implemented")
  end
  errTerm = rand(err_d, n)
  Y = X[:,1:s] * beta + errTerm

  Y, X, beta, n, p, s
end


@everywhere function estimCoeff(
    rep;
    tau = 0.5,
    j = 1,
    lambdaQR = 0.1,
    lambdaLasso = 0.1,
    corType = 1,
    noiseType = 1,
    h = 0.06
    )

  Y, X, true_beta, n, p, s = generate_data(rep; corType=corType, noiseType=noiseType)
  p = 10
  X = X[:, 1:p]

  solver = GurobiSolver(Method=1, OutputFlag=0)
  qr_problem = QRProblem(solver, X, Y)
  @time QR.solve!(qr_problem, 0., tau)
  intercept, ebeta = getBeta(qr_problem)
  a2 = getXi(qr_problem)

  lambda = fill(0., p)
  solve!(qr_problem, lambda, tau - 2*h)
  a3 = getXi(qr_problem)

  solve!(qr_problem, lambda, tau + 2*h)
  a1 = getXi(qr_problem)

  spF = dot(Y, a1 - 2*a2 + a3) / 4 / h^2

  A = cov(X, corrected=false, mean=0.)

  hb = ebeta[j]
  eSigma = inv(A)[j,j] * tau * (1 - tau) * spF^2 / n

  tb = true_beta[j]
  hb, eSigma, spF, (hb - tb) / sqrt(eSigma)
end


@everywhere function estimSparsityFunction(
    rep;
    tauArray = 0.2:0.01:0.8,
    lambdaQR = 0.1,
    corType = 1,
    noiseType = 1,
    h = 0.06
    )

  Y, X, true_beta, n, p, s = generate_data(rep; corType=corType, noiseType=noiseType)
  p = 10
  X = X[:, 1:p]
  lambda = fill(0., p)

  solver = GurobiSolver(Method=1, OutputFlag=0)
  qr_problem = QRProblem(solver, X, Y)

  spF = zeros(length(tauArray))
  for indTau=1:length(tauArray)
    tau = tauArray[indTau]
    QR.solve!(qr_problem, lambda, tau)
    a2 = getXi(qr_problem)
    solve!(qr_problem, lambda, tau - 2*h)
    a3 = getXi(qr_problem)
    solve!(qr_problem, lambda, tau + 2*h)
    a1 = getXi(qr_problem)

    spF[indTau] = dot(Y, a1 - 2*a2 + a3) / 4 / h^2
  end
  spF
end


numTests = 500

# ores = zeros(numTests)
# for i=1:numTests
#   ores[i] = res[i][4]
# end

# plt.hist(ores, min(30,numTests))

# qq = qqbuild(ores, Normal())
# scatter(qq.qx, qq.qy)
# ax = gca()
# ax[:set_ylim]([-3.5, 3.5])
# ax[:set_xlim]([-3.5, 3.5])
# xx=-3.5:0.01:3.5
# plot(xx,xx, color="red")

###  tau = 0.5, cor = 1, noise = 1

res = pmap(x -> estimCoeff(
             x;
             tau = 0.5,
             j = 1,
             lambdaQR = 0.1,
             lambdaLasso = 0.1,
             corType = 1,
             noiseType = 1,
             h = 0.06
             ),
           [1:numTests])
save("oracle_noise_1_cor_1_var_1.jld", "res", res)
res = pmap(x -> estimCoeff(
             x;
             tau = 0.5,
             j = 10,
             lambdaQR = 0.1,
             lambdaLasso = 0.1,
             corType = 1,
             noiseType = 1,
             h = 0.06
             ),
           [1:numTests])
save("oracle_noise_1_cor_1_var_10.jld", "res", res)

###  tau = 0.5, cor = 1, noise = 2

res = pmap(x -> estimCoeff(
             x;
             tau = 0.5,
             j = 1,
             lambdaQR = 0.1,
             lambdaLasso = 0.1,
             corType = 1,
             noiseType = 2,
             h = 0.06
             ),
           [1:numTests])
save("oracle_noise_2_cor_1_var_1.jld", "res", res)
res = pmap(x -> estimCoeff(
             x;
             tau = 0.5,
             j = 10,
             lambdaQR = 0.1,
             lambdaLasso = 0.1,
             corType = 1,
             noiseType = 2,
             h = 0.06
             ),
           [1:numTests])
save("oracle_noise_2_cor_1_var_10.jld", "res", res)

###  tau = 0.5, cor = 3, noise = 1

res = pmap(x -> estimCoeff(
             x;
             tau = 0.5,
             j = 1,
             lambdaQR = 0.1,
             lambdaLasso = 0.1,
             corType = 3,
             noiseType = 1,
             h = 0.06
             ),
           [1:numTests])
save("oracle_noise_1_cor_3_var_1.jld", "res", res)
res = pmap(x -> estimCoeff(
             x;
             tau = 0.5,
             j = 10,
             lambdaQR = 0.1,
             lambdaLasso = 0.1,
             corType = 3,
             noiseType = 1,
             h = 0.06
             ),
           [1:numTests])
save("oracle_noise_1_cor_3_var_10.jld", "res", res)

###  tau = 0.5, cor = 3, noise = 2

res = pmap(x -> estimCoeff(
             x;
             tau = 0.5,
             j = 1,
             lambdaQR = 0.1,
             lambdaLasso = 0.1,
             corType = 3,
             noiseType = 2,
             h = 0.06
             ),
           [1:numTests])
save("oracle_noise_2_cor_3_var_1.jld", "res", res)
res = pmap(x -> estimCoeff(
             x;
             tau = 0.5,
             j = 10,
             lambdaQR = 0.1,
             lambdaLasso = 0.1,
             corType = 3,
             noiseType = 2,
             h = 0.06
             ),
           [1:numTests])
save("oracle_noise_2_cor_3_var_10.jld", "res", res)


###################################################################################################
###################################################################################################
###################################################################################################



using PyPlot
using Cubature

d = MixtureModel(Normal, [(0., 1.), (3., 0.4), (-3.0, 0.4)], [0.6, 0.2, 0.2])

xxCDF = -7.:0.00001:7.
cdfMixture = [hquadrature(x -> pdf(d, x), -7., y)[1] for y in xxCDF];

res = pmap(x -> estimSparsityFunction(x;
                                      tauArray = 0.2:0.01:0.8,
                                      lambdaQR = 0.1,
                                      corType = 1,
                                      noiseType = 1,
                                      h = 0.03
                                      ),
           [1:5]
           )



plot(-3.:0.1:3., 1./pdf(d, -3.:0.1:3.), linewidth=2, color="red")
tauArr = 0.2:0.01:0.8
xx = zeros(length(tauArr))
for j=1:length(tauArr)
  ii = findfirst(x -> abs(x-tauArr[j]) < 1e-4, cdfMixture)
  xx[j] = xxCDF[ii]
end

for i=1:5
  plot(xx, res[i])
end

