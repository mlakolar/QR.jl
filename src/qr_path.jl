#input args :: outputDir rep correlationType noiseType

using HDF5, JLD
using QR, HD
using Mosek
using Distributions

# input arguments
outDir = ARGS[1]
rep = int(ARGS[2])
corType = int(ARGS[3])
noiseType = int(ARGS[4])

# random seed
srand(rep)

# generate data
p = 1500
n = 1000
s = 10

if corType == 1
  # equal correlation
  Sigma = 0.8 * ones(p,p) + 0.2 * eye(p)
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
  errTerm = randn(n)
elseif noiseType == 2
  tdist = TDist(1)
  errTerm = rand(tdist, n)
else
  error("Not implemented")
end

Y = X[:,1:s] * beta + errTerm

# save("$(outDir)/data_rep_$(rep)_cor_$(corType)_noise_$(noiseType).jld", "X", X, "Y", Y, "s", s, "beta", beta)

solver = MosekSolver(LOG=0,
                     OPTIMIZER=MSK_OPTIMIZER_FREE_SIMPLEX,
                     PRESOLVE_USE=MSK_PRESOLVE_MODE_OFF)
qr_problem = QRProblem(solver, X, Y)

# solves penalized quantile regression
lambdaArr = logspace(-0.3, -2, 100)
tauArr = [0.1:0.02:0.9]
# tauArr = [0.5]
qr_tau_path = cell(length(tauArr))
indTau = 0
for tau=tauArr
  indTau = indTau + 1
  println("tau = $(tau) --> $(indTau)/$(length(tauArr))")
  qr_tau_path[indTau] = compute_qr_path!(qr_problem, lambdaArr, tau; max_hat_s=15)
end

# inverts XX for debiasing
println("Inverting rows")
lasso_var_path = cell(p)
XX = cov(X, corrected=false, mean=0.)
Xy = zeros(Float64, p)
for j=1:p
  fill!(Xy, 0.)
  Xy[j] = -1.
  lasso_var_path[j] = compute_lasso_path(XX, Xy, lambdaArr; max_hat_s=10)
end

# refit quantile regression for each support
qr_refit_tau_path = cell(length(tauArr))
indTau = 0
lambda = fill(1.e10, p)
println("Refitting QR")
for tau=tauArr
  indTau = indTau + 1
  println("tau = $(tau) --> $(indTau)/$(length(tauArr))")
  qr_refit_tau_path[indTau] = compute_qr_path_refit!(qr_problem, qr_tau_path[indTau])
end


# inverts XX for debiasing
println("Inverting rows -- refit")
lasso_refit_var_path = cell(p)
for j=1:p
  fill!(Xy, 0.)
  Xy[j] = 1.
  lasso_refit_var_path[j] = compute_lasso_path_refit(lasso_var_path[j], XX, Xy)
end

save("$(outDir)/path_rep_$(rep)_cor_$(corType)_noise_$(noiseType).jld",
     "qr_tau_path", qr_tau_path,
     "qr_refit_tau_path", qr_refit_tau_path,
     "lasso_var_path", lasso_var_path,
     "lasso_refit_var_path", lasso_refit_var_path
     )

