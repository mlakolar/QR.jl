# helper functions for simulation 1

function generate_data(
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

  beta = collect(linspace(1., 0.5, s))
  X = randn(n, p) * sqSigma
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


function estimCoeff(
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

  solver = GurobiSolver(Method=1, OutputFlag=0)
#   solver = MosekSolver(LOG=0,
#                        OPTIMIZER=MSK_OPTIMIZER_FREE_SIMPLEX,
#                        PRESOLVE_USE=MSK_PRESOLVE_MODE_OFF)
  qr_problem = QRProblem(solver, X, Y)
  @time QR.solve!(qr_problem, lambdaQR, tau)
  intercept, ebeta = getBeta(qr_problem)

  A = cov(X, corrected=false, mean=0.)
  b = zeros(Float64, p)
  b[j] = 1.
  gamma = zeros(p)
  lasso!(gamma, A, b, lambdaLasso)

  support_nz = find(gamma)
  gamma_refit = zeros(p)
  gamma_refit[support_nz] = A[support_nz, support_nz] \ b[support_nz]

  # refit
  support_nz = find(ebeta)
  lambda = fill(1.e10, p)
  lambda[support_nz] = 0.

  solve!(qr_problem, lambda, tau - 2*h)
  a3 = getXi(qr_problem)

  solve!(qr_problem, lambda, tau)
  a2 = getXi(qr_problem)
  intercept, ebeta_refit = getBeta(qr_problem)

  solve!(qr_problem, lambda, tau + 2*h)
  a1 = getXi(qr_problem)

  spF = dot(Y, a1 - 2*a2 + a3) / 4 / h^2

  # gradient
  residuals = Array(Float64, n)
  gradient = Array(Float64, p)

  for i=1:n
    tmp = Y[i] - intercept - dot(slice(X, i, :), ebeta_refit)
    if tmp < 0.
      residuals[i] = tau - 1
    else
      residuals[i] = tau
    end
  end
  At_mul_B!(gradient, X, residuals)

  # debiased
  hb = ebeta_refit[j] + dot(gamma_refit, gradient) * spF / n
  eSigma = dot(gamma_refit, A * gamma_refit) * tau * (1 - tau) * spF^2 / n

  @show "Done $(rep)"
  tb = j > s ? 0. : true_beta[j]
  hb, eSigma, spF, (hb - tb) / sqrt(eSigma)
end

function estimCoeffOracle(
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
  if j > 10
    p = 11
    X = X[:, [1:10, j]]
  else
    p = 10
    X = X[:, 1:p]
  end

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

function estimSparsityFunctionOracle(
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
