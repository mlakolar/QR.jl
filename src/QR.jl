module QR

# import Mosek
# import Gurobi
import JuMP
using MathProgBase.SolverInterface

export
  QRProblem,
  solve!,
  getBeta, getBeta!,
  getXi, getXi!,
  compute_qr_path!,
  compute_qr_path_refit!


include("iidQR.jl")
include("VarCoefQR.jl")

end
