using JLD
using Distributions
using Formatting

function createTableElem(inFile)

  res = load(inFile, "res")

  numRep = 500
  coverage =  [abs(res[i][4]) <= 1.96 for i=1:numRep]
  width =  [2*1.96*res[i][2] for i=1:numRep]
  string(" & ", floor(mean(coverage)*1000)/10, " & (", floor(mean(width)*sqrt(1000)*1000)/1000, ") ")

end

fileDir = "/home/mkolar/tmp/qr_sim1_tau_0.5"
plotParams = Array(Any, 24)
plotParams[1] = "$(fileDir)/noise_1_cor_1_var_1.jld"
plotParams[2] = "$(fileDir)/oracle_noise_1_cor_1_var_1.jld"
plotParams[3] = "$(fileDir)/noise_1_cor_1_var_10.jld"
plotParams[4] = "$(fileDir)/oracle_noise_1_cor_1_var_10.jld"
plotParams[5] = "$(fileDir)/noise_1_cor_1_var_20.jld"
plotParams[6] = "$(fileDir)/oracle_noise_1_cor_1_var_20.jld"

plotParams[7] = "$(fileDir)/noise_1_cor_3_var_1.jld"
plotParams[8] = "$(fileDir)/oracle_noise_1_cor_3_var_1.jld"
plotParams[9] = "$(fileDir)/noise_1_cor_3_var_10.jld"
plotParams[10] = "$(fileDir)/oracle_noise_1_cor_3_var_10.jld"
plotParams[11] = "$(fileDir)/noise_1_cor_3_var_20.jld"
plotParams[12] = "$(fileDir)/oracle_noise_1_cor_3_var_20.jld"

####  Equicorrelation, Tdist
plotParams[13] = "$(fileDir)/noise_2_cor_1_var_1.jld"
plotParams[14] = "$(fileDir)/oracle_noise_2_cor_1_var_1.jld"
plotParams[15] = "$(fileDir)/noise_2_cor_1_var_10.jld"
plotParams[16] = "$(fileDir)/oracle_noise_2_cor_1_var_10.jld"
plotParams[17] = "$(fileDir)/noise_2_cor_1_var_20.jld"
plotParams[18] = "$(fileDir)/oracle_noise_2_cor_1_var_20.jld"

####  Toeplitz, Tdist
plotParams[19] = "$(fileDir)/noise_2_cor_3_var_1.jld"
plotParams[20] = "$(fileDir)/oracle_noise_2_cor_3_var_1.jld"
plotParams[21] = "$(fileDir)/noise_2_cor_3_var_10.jld"
plotParams[22] = "$(fileDir)/oracle_noise_2_cor_3_var_10.jld"
plotParams[23] = "$(fileDir)/noise_2_cor_3_var_20.jld"
plotParams[24] = "$(fileDir)/oracle_noise_2_cor_3_var_20.jld"

latexStr = ""
rowName = ["EQ, \$N(0,1)\$", "Toeplitz, \$N(0,1)\$", "EQ, \$t_1\$", "Toeplitz, \$t_1\$"]
for r=1:4
  latexStr = string(latexStr, rowName[r], " ")
  for c=1:3
    latexStr = string(latexStr, createTableElem(plotParams[(r-1)*6+(c-1)*2+1]))
  end
  latexStr = string(latexStr, "\\\\\n")
end

latexStr = string(latexStr, "\n\n")

for r=1:4
  latexStr = string(latexStr, rowName[r], " ")
  for c=1:3
    latexStr = string(latexStr, createTableElem(plotParams[(r-1)*6+(c-1)*2+2]))
  end
  latexStr = string(latexStr, "\\\\\n")
end


print(latexStr)
