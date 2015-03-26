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

# refit variables are dictionaries with keys equal to support set
type QRPath
  lambdaArr
  tau
  hBeta
  hXi
  optval
end

# assumes that the first row of X is equal to all ones
# lambdaArr is in decreasing order
# tauArr is ordered
# these requirements are not strict, however, it may be useful for
function qr_path(X::Array{Float64, 2}, Y::Array{Float64, 2},
                 lambdaArr::Array{Float64, 1}, tauArr::Float64,
                 solver::AbstractMathProgSolve; max_hat_s=Inf, zero_thr=1e-4)

  n, p = size(X)
  varX = vec(mapslices(norm, X, 1)) / sqrt(n)
  varX[1] = 0.

  insBeta = zeros(Float64, p)
  insXi = zeros(Float64, n)

  _lambdaArr = copy(lambdaArr)
  numLambda  = length(lambdaArr)
  hBeta = cell(numLambda)
  hXi = cell(numLambda)
  optval = cell(numLambda)

  qr_problem = QRProblem(solver, X, Y)

  for indLambda=1:numLambda
    @show "$(indLambda)/$(numLambda)"
    tmpBeta, tmpXi = solve!(qr_problem, lambdaArr[indLambda] * sqrt(tau*(1-tau)) * varX, tau)

    fill!(insBeta, 0.)
    for kv=tmpBeta
      if abs(kv[2]) > zero_thr
        insBeta[kv[1]] = kv[2]
      end
    end
    hBeta[indLambda] = sparse(insBeta)

    fill!(insXi, 0.)
    for kv=tmpXi
      insXi[kv[1]] = kv[2]
    end
    hXi[indLambda, indTau] = sparse(insXi)

    optval[indLambda] = JuMP.getObjectiveValue(qr_problem.problem)

    if nnz(hBeta[indLambda]) > max_hat_s
      _lambdaArr = lambdaArr[1:indLambda-1]
      hBeta = hBeta[1:indLambda-1]
      hXi = hXi[1:indLambda-1]
      optval = optval[1:indLambda-1]
      break
    end
  end

  QRPath(_lambdaArr, tau, hBeta, hXi, optval)
end


end
