addprocs(15);

using JLD
@everywhere using Distributions
@everywhere using QR
@everywhere using Gurobi
@everywhere using CDLasso

@everywhere include("helper_funcs.jl")

macro m_estimCoeff(coefIndex, corType, noiseType, numTests)
    return :(
       pmap(x -> estimCoeff(
            x;
            tau = 0.5,
            j = $coefIndex,
            lambdaQR = 0.1,
            lambdaLasso = 0.1,
            corType = $corType,
            noiseType = $noiseType,
            h = 0.06
            ),
           [1:$numTests])
    )
end

macro m_estimCoeffOracle(coefIndex, corType, noiseType, numTests)
    return :(
       pmap(x -> estimCoeffOracle(
            x;
            tau = 0.5,
            j = $coefIndex,
            lambdaQR = 0.1,
            lambdaLasso = 0.1,
            corType = $corType,
            noiseType = $noiseType,
            h = 0.06
            ),
           [1:$numTests])
    )
end

vars = [1, 10, 20]
corTypes = [1, 3]
noiseTypes = [1, 2, 4]

## oracle simulations
for v in vars
    for c in corTypes
        for n in noiseTypes
          if !isfile("oracle_noise_$(n)_cor_$(c)_var_$(v).jld")
            @show (v,c,n)
            res = @m_estimCoeffOracle(v,c,n,500)
            save("oracle_noise_$(n)_cor_$(c)_var_$(v).jld", "res", res)
          end
        end
    end
end

## our estimator
for v in vars
  for c in corTypes
    for n in noiseTypes
      if !isfile("noise_$(n)_cor_$(c)_var_$(v).jld")
        @show (v,c,n)
        res = @m_estimCoeff(v,c,n,500)
        save("noise_$(n)_cor_$(c)_var_$(v).jld", "res", res)
      end
    end
  end
end

