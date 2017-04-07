######################################################################
#
#  Penalized QR
#
######################################################################

type QRProblem
  problem::JuMP.Model
  intercept
  beta
  t         # p variables -- penalty
  up        # n variables
  un        # n variables
  xi_dual   #
  n::Int64
  p::Int64

  function QRProblem(solver::AbstractMathProgSolver, X::Array{Float64, 2}, Y::Array{Float64, 1})
    n, p = size(X)
    problem = JuMP.Model(solver=solver)

    @JuMP.variable(problem, intercept)
    @JuMP.variable(problem, beta[1:p])
    @JuMP.variable(problem, t[1:p])
    @JuMP.variable(problem, up[1:n])
    @JuMP.variable(problem, un[1:n])
    @JuMP.constraint(problem, xi_dual[i=1:n], Y[i] - intercept - dot(vec(X[i,:]), beta) == up[i] - un[i])
    for i=1:n
        @JuMP.constraint(problem, up[i] >= 0)
        @JuMP.constraint(problem, un[i] >= 0)
    end
    for i=1:p
        @JuMP.constraint(problem, -beta[i] <= t[i])
        @JuMP.constraint(problem,  beta[i] <= t[i])
    end

    new(problem, intercept, beta, t, up, un, xi_dual, n, p)
  end
end


solve!(qr_problem::QRProblem, lambda::Float64, tau::Float64) = solve!(qr_problem, fill(lambda, qr_problem.p), tau, fill(1, qr_problem.n))
function solve!(
    qr_problem::QRProblem,
    lambda::Array{Float64, 1},
    tau::Float64,
    weights::Array{Float64, 1}   ## observations weights --- in iid setting should be a vector of ones
    )
  @assert 0. < tau < 1.
  @JuMP.objective(qr_problem.problem,
                     Min,
                     (tau*dot(weights, qr_problem.up) + (1-tau)*dot(weights, qr_problem.un)) / qr_problem.n
                         + dot(lambda, qr_problem.t)
                     )

  JuMP.solve(qr_problem.problem)
  nothing
end


function getBeta!(tmpBeta::Vector{Float64}, qr_problem::QRProblem; zero_thr=1e-4)
  @assert length(tmpBeta) == qr_problem.p
  copy!(tmpBeta, JuMP.getvalue(qr_problem.beta))
  for i=1:qr_problem.p
    tmpBeta[i] = abs(tmpBeta[i]) > zero_thr ? tmpBeta[i] : 0
  end
  JuMP.getvalue(qr_problem.intercept), tmpBeta
end
getBeta(qr_problem::QRProblem; zero_thr=1e-4) = getBeta!(zeros(qr_problem.p), qr_problem; zero_thr=zero_thr)


function getXi!(xi::Vector{Float64}, qr_problem::QRProblem)
  @assert length(xi) == qr_problem.n
  copy!(xi, JuMP.getdual(qr_problem.xi_dual))
  xi
end
getXi(qr_problem::QRProblem) = getXi!(zeros(qr_problem.n), qr_problem)

######################################################################
#
#  Penalized QR path
#
######################################################################

# refit variables are dictionaries with keys equal to support set
type QRPath
  lambdaArr
  tau
  intercept
  beta
  xi
  optval
end

# lambdaArr is in decreasing order
function compute_qr_path!(qr_problem::QRProblem,
                          lambdaArr::Array{Float64, 1},
                          tau::Float64,
                          stdX::Array{Float64, 1};
                          max_hat_s=Inf, zero_thr=1e-4)

  p = qr_problem.p
  n = qr_problem.n
  oneN = ones(n)

  _lambdaArr = copy(lambdaArr)
  numLambda  = length(lambdaArr)
  intercept = Array(Float64, numLambda)
  beta = Array{Any}(numLambda)
  xi = Array(Float64, numLambda, n)
  optval = Array(Float64, numLambda)

  for indLambda=1:numLambda
    if mod(indLambda, 10) == 1
      print("Solving $(indLambda)/$(numLambda)")
    end
    solve!(qr_problem, lambdaArr[indLambda] * stdX, tau, oneN)

    intercept[indLambda], beta[indLambda] = getBeta(qr_problem; zero_thr=zero_thr)
    if mod(indLambda, 10) == 1
      println("    nnz ==  $(countnz(beta[indLambda]))")
    end
    xi[indLambda, :] = getXi(qr_problem)
    optval[indLambda] = JuMP.getobjectivevalue(qr_problem.problem)

    if countnz(beta[indLambda]) > max_hat_s
      _lambdaArr = lambdaArr[1:indLambda-1]
      beta = beta[1:indLambda-1]
      xi = xi[1:indLambda-1, :]
      optval = optval[1:indLambda-1]
      break
    end
  end

  QRPath(_lambdaArr, tau, intercept, beta, xi, optval)
end


type QRRefit
  intercept
  beta
  xi
end

function compute_qr_path_refit!(qr_problem::QRProblem, qr_path::QRPath; zero_thr=1e-4)
  tau = qr_path.tau
  lambda = fill(1.e6, qr_problem.p)
  lambdaArr = qr_path.lambdaArr

  oneN = ones(qr_problem.n)

  tmpDict = Dict()
  for i=1:length(lambdaArr)
    support_nz = find(qr_path.beta[i])
    if haskey(tmpDict, support_nz)
      continue
    end
    fill!(lambda, 1.e6)
    lambda[support_nz] = 0.
    solve!(qr_problem, lambda, tau, oneN)
    _intercept, _beta = getBeta(qr_problem; zero_thr=zero_thr)
    _xi = getXi(qr_problem)
    tmpDict[support_nz] = QRRefit(_intercept, _beta, _xi)
  end
  tmpDict
end
