using FactCheck

using QR
using Convex, Mosek

cvx_solver = MosekSolver(LOG=0,OPTIMIZER=Mosek.MSK_OPTIMIZER_FREE_SIMPLEX)
set_default_solver(cvx_solver)

solver = MosekSolver(LOG=0,OPTIMIZER=Mosek.MSK_OPTIMIZER_FREE_SIMPLEX)

function qr_primal_cvx(X, Y, lambda, tau)
  n, p = size(X)
  theta = Convex.Variable(p);
  u = vec(Y-X*theta)
  prob = Convex.minimize(sum( abs(u) / 2. + (tau - 0.5) * u ) / n  + dot(lambda, abs(theta)))
  Convex.solve!(prob)

  prob, theta
end

facts("path") do
  p = 200
  n = 100
  s = 5
  beta = randn(s)
  X = randn(n, p)
  tX = hcat(ones(n), X)
  Y = X[:,1:s] * beta + 0.1 * randn(n)

  std_col = zeros(Float64, p+1)
  std_col[1] = 0.
  for j=2:p+1
    std_col[j] = norm(tX[:, j])
  end

  lambdaArr = [0.3, 0.1, 0.05, 0.01]
  tauArr = [0.1:0.05:0.9]
  sol_qr_path = QR.qr_path(tX, Y, lambdaArr, tauArr, solver)

  for indLambda=1:length(lambdaArr)
    for indTau=1:length(tauArr)
      lambda = lambdaArr[indLambda] * sqrt(tauArr[indTau]*(1-tauArr[indTau])) * std_col
      hat_prob, hat_beta_cvx = qr_primal_cvx(tX, Y, lambda, tauArr[indTau])

      @fact abs(sol_qr_path.optval[indLambda,indTau]-hat_prob.optval) => roughly(0., 1e-5)
    end
  end

end

FactCheck.exitstatus()
