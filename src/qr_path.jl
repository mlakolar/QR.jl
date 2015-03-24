#input args :: outputDir rep correlationType noiseType

using HDF5, JLD
using QR
using Mosek
using Distributions

solver = MosekSolver(LOG=0,OPTIMIZER=Mosek.MSK_OPTIMIZER_FREE_SIMPLEX)

# input arguments
outDir = ARGS[1]
rep = int(ARGS[2])
corType = int(ARGS[3])
noiseType = int(ARGS[4])

# random seed
srand(rep)

if corType == 1
  # equal correlation
  Sigma = 0.8 * ones(p,p) + 0.2 * eye(p)
elseif corType == 2
  # toeplitz
  rho = 0.8
  Sigma = zeros(Float64, p,p)
  for a=1:p
    for b=a:p
      t = rho^(a-b)
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
      t = rho^(a-b)
      Sigma[a,b] = t
      Sigma[b,a] = t
    end
  end
end

sqSigma = sqrtm(Sigma)

# generate data
p = 1500
n = 1000
s = 10
beta = linspace(1., 0.5, s)
X = randn(n, p) * sqSigma
tX = hcat(ones(n), X)
if noiseType == 1
  errTerm = randn(n)
elseif noiseType == 2
  tdist = TDist(1)
  errTerm = rand(tdist, n)
end

Y = X[:,1:s] * beta + noiseType

# save("$(outDir)/data_rep_$(rep)_cor_$(corType)_noise_$(noiseType).jld", "X", X, "Y", Y, "s", s, "beta", beta)

lambdaArr = logspace(-1.7, -2.3, 50)
tauArr = [0.1:0.02:0.9]
sol_qr_path = QR.qr_path(tX, Y, lambdaArr, tauArr, solver)

save("$(outDir)/path_rep_$(rep)_cor_$(corType)_noise_$(noiseType).jld", "sol_qr_path", sol_qr_path)

