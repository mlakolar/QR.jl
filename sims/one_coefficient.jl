using HDF5, JLD
using Distributions
import QR
import Gurobi
import CDLasso

# generate data
p = 1500
n = 1000
s = 10

rho = 0.1
Sigma = zeros(Float64, p,p)
for a=1:p
  for b=a:p
    t = rho^abs(a-b)
    Sigma[a,b] = t
    Sigma[b,a] = t
  end
end

sqSigma = sqrtm(Sigma)

beta = linspace(1., 0.5, s)
X = randn(n, p) * sqSigma
tX = hcat(ones(n), X)
errTerm = randn(n)


Y = X[:,1:s] * beta + errTerm

##

solver = Gurobi.GurobiSolver(Method=1, OutputFlag=1)
qr_problem = QR.QRProblem(solver, X, Y)

lambda = fill(0.1, p)
@time QR.solve!(qr_problem, lambda, 0.5)
QR.getBeta(qr_problem)

XX = cov(X, corrected=false, mean=0.)
Xy = zeros(Float64, p)
Xy[1] = 1.
gamma = zeros(p)
CDLasso.lasso!(gamma, XX, Xy, 0.1)
