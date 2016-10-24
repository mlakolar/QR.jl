using FactCheck

using QR
using Convex, Mosek

cvx_solver = MosekSolver(LOG=0,OPTIMIZER=Mosek.MSK_OPTIMIZER_FREE_SIMPLEX)
# cvx_solver = GurobiSolver(Method=1, OutputFlag=0)
set_default_solver(cvx_solver)

solver = MosekSolver(LOG=0,OPTIMIZER=Mosek.MSK_OPTIMIZER_FREE_SIMPLEX)
# solver = GurobiSolver(Method=1, OutputFlag=0)

function qr_primal_cvx(X, Y, lambda, tau)
  n, p = size(X)
  theta = Convex.Variable(p);
  intercept = Convex.Variable(1);
  u = vec(Y-intercept-X*theta)
  prob = Convex.minimize(sum( abs(u) / 2. + (tau - 0.5) * u ) / n  + dot(lambda, abs(theta)))
  Convex.solve!(prob)

  prob, intercept, theta
end

facts("path") do
  p = 200
  n = 100
  s = 5
  beta = randn(s)
  X = randn(n, p)
  Y = X[:,1:s] * beta + 0.1 * randn(n)

  std_col = zeros(Float64, p)
  std_col[1] = 0.
  for j=1:p
    std_col[j] = norm(X[:, j]) / sqrt(n)
  end

  lambdaArr = [0.3, 0.1, 0.05, 0.01]
  tauArr = [0.5;]
  qr_prob = QRProblem(solver, X, Y)

  for indTau=1:length(tauArr)
    sol_qr_path = QR.compute_qr_path!(qr_prob, lambdaArr, tauArr[indTau])

    for indLambda=1:length(lambdaArr)
      lambda = lambdaArr[indLambda] * std_col
      hat_prob, hat_intercept_cvx, hat_beta_cvx = qr_primal_cvx(X, Y, lambda, tauArr[indTau])

      @fact abs(sol_qr_path.optval[indLambda]-hat_prob.optval) --> roughly(0., 1e-5)
    end
  end

end

FactCheck.exitstatus()
