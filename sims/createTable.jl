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

fileDirArr = Array(Any, 3)
fileDirArr[1] = "/home/mkolar/tmp/qr_sim1_tau_0.3"
fileDirArr[2] = "/home/mkolar/tmp/qr_sim1_tau_0.5"
fileDirArr[3] = "/home/mkolar/tmp/qr_sim1_tau_0.6"

latexStr = ""
rowName = ["EQ, \$N(0,1)\$", "Toeplitz, \$N(0,1)\$", "EQ, \$t_1\$", "Toeplitz, \$t_1\$"]

for fNameInd=1:3

  fileDir = fileDirArr[fNameInd]

  plotParams = Array(Any, 24)
  if fNameInd == 2
    append = ""
  elseif fNameInd == 1
    append = "_tau_0.3"
  else
    append = "_tau_0.6"
  end

  plotParams[1] = "$(fileDir)/noise_1_cor_1_var_1$(append).jld"
  plotParams[2] = "$(fileDir)/oracle_noise_1_cor_1_var_1$(append).jld"
  plotParams[3] = "$(fileDir)/noise_1_cor_1_var_10$(append).jld"
  plotParams[4] = "$(fileDir)/oracle_noise_1_cor_1_var_10$(append).jld"
  plotParams[5] = "$(fileDir)/noise_1_cor_1_var_20$(append).jld"
  plotParams[6] = "$(fileDir)/oracle_noise_1_cor_1_var_20$(append).jld"

  plotParams[7] = "$(fileDir)/noise_1_cor_3_var_1$(append).jld"
  plotParams[8] = "$(fileDir)/oracle_noise_1_cor_3_var_1$(append).jld"
  plotParams[9] = "$(fileDir)/noise_1_cor_3_var_10$(append).jld"
  plotParams[10] = "$(fileDir)/oracle_noise_1_cor_3_var_10$(append).jld"
  plotParams[11] = "$(fileDir)/noise_1_cor_3_var_20$(append).jld"
  plotParams[12] = "$(fileDir)/oracle_noise_1_cor_3_var_20$(append).jld"

  ####  Equicorrelation, Tdist
  plotParams[13] = "$(fileDir)/noise_2_cor_1_var_1$(append).jld"
  plotParams[14] = "$(fileDir)/oracle_noise_2_cor_1_var_1$(append).jld"
  plotParams[15] = "$(fileDir)/noise_2_cor_1_var_10$(append).jld"
  plotParams[16] = "$(fileDir)/oracle_noise_2_cor_1_var_10$(append).jld"
  plotParams[17] = "$(fileDir)/noise_2_cor_1_var_20$(append).jld"
  plotParams[18] = "$(fileDir)/oracle_noise_2_cor_1_var_20$(append).jld"

  ####  Toeplitz, Tdist
  plotParams[19] = "$(fileDir)/noise_2_cor_3_var_1$(append).jld"
  plotParams[20] = "$(fileDir)/oracle_noise_2_cor_3_var_1$(append).jld"
  plotParams[21] = "$(fileDir)/noise_2_cor_3_var_10$(append).jld"
  plotParams[22] = "$(fileDir)/oracle_noise_2_cor_3_var_10$(append).jld"
  plotParams[23] = "$(fileDir)/noise_2_cor_3_var_20$(append).jld"
  plotParams[24] = "$(fileDir)/oracle_noise_2_cor_3_var_20$(append).jld"

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

  latexStr = string(latexStr, "\n\n")
  latexStr = string(latexStr, "\n\n")
  latexStr = string(latexStr, "\n\n")

end

print(latexStr)
