using PyPlot
using JLD
fileDir = "/home/mkolar/tmp/qr_sim1_tau_0.5"


n = 1  # noise
c = 1  # correlation type
v = 1  # variable

res = load("$(fileDir)/noise_$(n)_cor_$(c)_var_$(v).jld", "res")


