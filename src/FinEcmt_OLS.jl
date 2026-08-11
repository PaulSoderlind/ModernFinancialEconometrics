module FinEcmt_OLS

using Statistics, LinearAlgebra, DelimitedFiles, Random, Distributions, StatsBase
import Printf
using FiniteDiff: finite_difference_jacobian as jacobian

export BinScatter, BinDummies,
JarqueBeraTest, CompanionFormAR, CovNW, CovToCor,
DeltaMethod, DrawBlocks, EMA, EWportf,
FindNNPanel, FindNN, FirstDiff,
IndividualDemean, KolSmirTest,
HistcNorm, HistAsh,
NWCovPs, OLSyxReplaceNaN, OlsAutoCorr, OlsBasic, OlsBootstrap, OlsGM, OlsNW, RegressionTable,
OlsR2Test, OlsSure, OlsWhitesTest,RegressionFit, VIF, DiagnosticsTable, DiagnosticsNoniidTable,
PanelOls, PanelReshuffle, DummiesCreate, TimeDummiesTTN,
FixedIndivEffects,FixedTimeEffects, FixedIndivTimeEffects, FixedTimeIndivEffects, FWonZRepeated!,
PutDataInNT, Readcsv,
QuantRegrIRLS, ReturnStats, RidgeRegression, StandardiseYX,
TwoSLS, excise, lag, lag2,
printblue, printlnPs, printmagenta, printmat, printred, printyellow,
rankPs, sortLoHi, cdfNorm, logpdfNorm, ContingencyTable, StandardiseX, @doc2

include("BinScatter.jl")
include("CovNW.jl")
include("DeltaMethod.jl")
include("DistributionTests.jl")
include("HistAsh.jl")
include("Ols.jl")
include("OlsBootstrap.jl")
include("OlsDiagnostics.jl")
include("OlsSure.jl")
include("PanelOls.jl")
include("PanelRegrBasic.jl")
include("PortfolioSorts.jl")
include("printmat.jl")
include("QuantRegrIRLS.jl")
include("RidgeRegression.jl")
include("TwoSLS.jl")
include("UtilityFunctions.jl")

end
