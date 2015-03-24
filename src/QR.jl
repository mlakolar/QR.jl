module QR

import Mosek
import JuMP

export
  QRProblem,
  solve,
  qr_path

######################################################################
#
#  Penalized QR
#
######################################################################

type QRProblem
  problem::JuMP.Model
  beta
  t         # p variables -- penalty
  up        # n variables
  un        # n variables
  xi_dual   #
  n::Int64
  p::Int64

  function QRProblem(solver, X, Y, simplex=true)
    n, p = size(X)
    problem = JuMP.Model(solver=solver)

    @JuMP.defVar(problem, beta[1:p])
    @JuMP.defVar(problem, t[1:p])
    @JuMP.defVar(problem, up[1:n])
    @JuMP.defVar(problem, un[1:n])
    @JuMP.addConstraint(problem, xi_dual[i=1:n], Y[i] - dot(vec(X[i,:]), beta) == up[i] - un[i])
    for i=1:n
        @JuMP.addConstraint(problem, up[i] >= 0)
        @JuMP.addConstraint(problem, un[i] >= 0)
    end
    for i=1:p
        @JuMP.addConstraint(problem, -beta[i] <= t[i])
        @JuMP.addConstraint(problem, beta[i] <= t[i])
    end

    new(problem, beta, t, up, un, xi_dual, n, p)
  end

end

function solve!(qr_problem::QRProblem, lambda::Array{Float64, 1}, tau::Float64)
  oneN = ones(qr_problem.n)
  @JuMP.setObjective(qr_problem.problem, Min, (tau*dot(oneN, qr_problem.up) + (1-tau)*dot(oneN, qr_problem.un)) / qr_problem.n + dot(lambda, qr_problem.t))

  JuMP.solve(qr_problem.problem)
  JuMP.getValue(qr_problem.beta), JuMP.getDual(qr_problem.xi_dual)
end

######################################################################
#
#  Penalized QR path
#
######################################################################


type QRPath
  lambdaArr
  tauArr
  hBeta
  hBetaRefit
  hXi
  hXiRefit
  optval
  optvalRefit
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
  optval = cell(numLambda, numTau)
  optvalRefit = cell(numLambda, numTau)


  qr_problem = QRProblem(solver, X, Y)
  forwardTau = 1
  for indLambda=1:numLambda
    @show "$(indLambda)/$(numLambda)"
    if forwardTau == 1
      tauRange = 1:numTau
    else
      tauRange = numTau:-1:1
    end
    for indTau=tauRange
      tmpBeta, tmpXi = solve!(qr_problem, lambdaArr[indLambda] * sqrt(tauArr[indTau]*(1-tauArr[indTau])) * varX, tauArr[indTau])

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

      optval[indLambda, indTau] = JuMP.getObjectiveValue(qr_problem.problem)

      # refit
      indNonZero = find(abs(insBeta) .> 1e-4)
      hat_s = length(indNonZero)
      refit_problem = QRProblem(solver, X[:, indNonZero], Y)
      tmpBeta, tmpXi = solve!(refit_problem, zeros(Float64, hat_s), tauArr[indTau])

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

      optvalRefit[indLambda, indTau] = JuMP.getObjectiveValue(refit_problem.problem)


    end
    forwardTau = 1 - forwardTau
  end

  QRPath(lambdaArr, tauArr, hBeta, hBetaRefit, hXi, hXiRefit, optval, optvalRefit)
end


end
