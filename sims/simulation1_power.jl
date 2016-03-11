addprocs(15);

using JLD
@everywhere using Distributions
@everywhere using QR
@everywhere using Gurobi
@everywhere using CDLasso

@everywhere include("helper_funcs.jl")

macro m_powerCurve(corType, noiseType, numTests)
    return :(
       pmap(x -> createPowerCurve(
            x;
            corType = $corType,
            noiseType = $noiseType,
            ),
           1:$numTests)
    )
end

corTypes = [1, 3]
noiseTypes = [1, 2]

## oracle simulations
for c in corTypes
    for n in noiseTypes
      if !isfile("power_noise_$(n)_cor_$(c).jld")
        @show (c,n)
        @time res = @m_powerCurve(c,n,500)
        save("power_noise_$(n)_cor_$(c).jld", "res", res)
      end
    end
end
