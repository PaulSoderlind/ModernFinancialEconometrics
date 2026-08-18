module FinEcmt_KernelRegression

using Statistics, LinearAlgebra

export EpanechnikovKernel, GaussianKernel, KernelDensity, KernelRegression,
LocalLinearRegression, UniformKernel, hRuleOfThumb, CrossValidateKernelR2,
CrossValidateKernelR

include("KernelRegression.jl")
include("CovNW.jl")                   #used here
include("Ols.jl")                     #used here
include("UtilityFunctions.jl")        #used here

end
