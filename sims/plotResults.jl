using PyPlot
using JLD
using Distributions


function createFigure(
    inFile,
    outFile,
    xlimA,
    titleText,
    ylabelText,
    )

  res = load(inFile, "res")

  numRep = 500
  stdTest =  [res[i][4] for i=1:numRep]
  plt[:hist](stdTest, 50, normed=true)
  xlim(xlimA)
  d = Normal()
  x = linspace(xlimA[1], xlimA[2], 500)
  y = pdf(d, x)
  h = plt[:plot](x, y, lw=3, color="red")
  title(titleText, fontsize=20)
  ylabel(ylabelText, fontsize=20)
  savefig(outFile, format="pdf")
  plt[:close]()

end


fileDir = "/home/mkolar/tmp/qr_sim1_tau_0.5"
true_beta = collect(linspace(1., 0.5, s))

xlimA = [-4, 4]

plotParams = Array(Any, 6)
plotParams[1] = [
  "$(fileDir)/noise_1_cor_1_var_1.jld",
  "/home/mkolar/tmp/noise_1_cor_1_var_1.pdf",
  L"$\beta_1 = 1$",
  "Equicorrelation design"
  ]

plotParams[2] = [
  "$(fileDir)/oracle_noise_1_cor_1_var_1.jld",
  "/home/mkolar/tmp/oracle_noise_1_cor_1_var_1.pdf",
  "",
  "Equicorrelation design (oracle)"
]

plotParams[3] = [
   "$(fileDir)/noise_1_cor_1_var_10.jld",
   "/home/mkolar/tmp/noise_1_cor_1_var_10.pdf",
   L"$\beta_{10} = 0.5$",
   ""
  ]

plotParams[4] = [
 "$(fileDir)/oracle_noise_1_cor_1_var_10.jld",
  "/home/mkolar/tmp/oracle_noise_1_cor_1_var_10.pdf",
  "",
  ""
]

plotParams[5] = [
  "$(fileDir)/noise_1_cor_1_var_20.jld",
  "/home/mkolar/tmp/noise_1_cor_1_var_20.pdf",
  L"$\beta_{20} = 0$",
  ""
]

####  Equicorrelation, Gaussian

for i=1:5
  inFile = plotParams[i][1]
  outFile = plotParams[i][2]
  titleText = plotParams[i][3]
  ylabelText = plotParams[i][4]
  createFigure(inFile,outFile,xlimA,titleText,ylabelText)
end




