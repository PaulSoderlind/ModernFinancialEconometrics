module FinEcmt_KernelRegression

using Statistics

export EpanechnikovKernel, GaussianKernel, KernelDensity, KernelRegression,
LocalLinearRegression, UniformKernel, hRuleOfThumb, CrossValidateKernelR

include("CovNW.jl")                   #used here
include("KernelRegression.jl")
include("Ols.jl")                     #used here

end
