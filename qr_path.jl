import HD
using Mosek

type QRPath
  lambdaArr
  tauArr
  hBeta
  hBetaRefit
  hXi
  hXiRefit
end

# assumes that the first row of X is equal to all ones
# lambdaArr is in decreasing order
# tauArr is ordered
# these requirements are not strict, however, it may be useful for
function qr_path(X, Y, lambdaArr, tauArr, solver)
  n, p = size(X)
  varX = vec(mapslices(norm, X, 1))
  varX[1] = 0.

  insBeta = zeros(Float64, p)
  insXi = zeros(Float64, n)

  numLambda  = length(lambdaArr)
  numTau = length(tauArr)
  hBeta = cell(numLambda, numTau)
  hBetaRefit = cell(numLambda, numTau)
  hXi = cell(numLambda, numTau)
  hXiRefit = cell(numLambda, numTau)

  qr_problem = HD.QR(solver, X, Y)
  forwardTau = 1
  for indLambda=1:numLambda
    if forwardTau == 1
      tauRange = 1:numTau
    else
      tauRange = numTau:-1:1
    end
    for indTau=tauRange
      tmpBeta, tmpXi = HD.solve!(qr_problem, lambdaArr[indLambda] * sqrt(tauArr[indTau]*(1-tauArr[indTau])) * varX, tauArr[indTau])

      fill!(insBeta, 0.)
      for kv=tmpBeta
        insBeta[kv[1]] = kv[2]
      end
      hBeta[indLambda, indTau] = sparse(insBeta)

      fill!(insXi, 0.)
      for kv=tmpXi
        insXi[kv[1]] = kv[2]
      end
      hXi[indLambda, indTau] = sparse(insXi)

      # refit
      indNonZero = find(abs(insBeta) .> 1e-4)
      hat_s = length(indNonZero)
      refit_problem = HD.QR(solver, X[:, indNonZero], Y)
      tmpBeta, tmpXi = HD.solve!(refit_problem, zeros(Float64, hat_s), tauArr[indTau])

      fill!(insXi, 0.)
      for kv=tmpXi
        insXi[kv[1]] = kv[2]
      end
      hXiRefit[indLambda, indTau] = sparse(insXi)

      insBetaR = Array(Float64, hat_s)
      for i=1:hat_s
        insBetaR[i] = tmpBeta[i]
      end
      hBetaRefit[indLambda, indTau] = sparsevec(indNonZero, insBetaR)

    end
    forwardTau = 1 - forwardTau
  end

  QRPath(lambdaArr, tauArr, hBeta, hBetaRefit, hXi, hXiRefit)
end

p = 20
n = 100
X = randn(n, p)
tX = hcat(ones(n), X)
Y = X[:,1:10] * ones(10) + 0.1 * randn(n)

solver = MosekSolver(LOG=0,OPTIMIZER=Mosek.MSK_OPTIMIZER_FREE_SIMPLEX)
solver = MosekSolver(LOG=0)
lambdaArr = logspace(-1, -3, 20)
tauArr = 0.1:0.02:0.9
@time ans = qr_path(tX, Y, lambdaArr, tauArr, solver)

ans.hBeta[7, 1]
ans.hBetaRefit[7,1]

