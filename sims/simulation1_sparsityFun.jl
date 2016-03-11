addprocs(15);

using JLD
@everywhere using Distributions
@everywhere using QR
# @everywhere using Mosek
@everywhere using Gurobi
@everywhere using CDLasso

@everywhere include("helper_funcs.jl")

macro m_estimCoeffOracle(h, corType, noiseType, numTests)
    return :(
       pmap(x -> estimSparsityFunctionOracle(
            x;
            corType = $corType,
            noiseType = $noiseType,
            h = $h
            ),
           1:$numTests)
    )
end

corTypes = [1, 3]
noiseTypes = [1, 2]
hArr = [0.02, 0.04, 0.06, 0.08]

## oracle simulations
for h in hArr
    for c in corTypes
        for n in noiseTypes
          if !isfile("spf_noise_$(n)_cor_$(c)_h_$(h).jld")
            @show (h,c,n)
            @time res = @m_estimCoeffOracle(h,c,n,200)
            save("spf_noise_$(n)_cor_$(c)_h_$(h).jld", "res", res)
          end
        end
    end
end
