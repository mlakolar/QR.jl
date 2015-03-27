#input args :: outputDir rep correlationType noiseType

using HDF5, JLD
using QR, HD

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
  Sigma = 0.5 * ones(p,p) + 0.2 * eye(p)
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

save("$(outDir)/data_rep_$(rep)_cor_$(corType)_noise_$(noiseType).jld", "X", X, "Y", Y, "s", s, "beta", beta)

