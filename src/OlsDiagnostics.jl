"""
    OlsR2Test(R²,T,df)

Test of all slope coefficients. Notice that the regression must contain
an intercept for R² to be useful.

### Input
- `R²::Number`:    R² value
- `T::Int`:        number of observations
- `df::Number`:    number of (non-constant) regressors

### Output
- `RegrStat::Number`: test statistic
- `pval::Number`:     p-value

"""
function OlsR2Test(R²,T,df)
    RegrStat = T*R²/(1-R²)           #R\^2[TAB]
    pval     = ccdf(Chisq(df),RegrStat)    #same as 1-cdf()
    return RegrStat, pval
end


"""
    OlsWhitesTest(u,x)

Test of heteroskedasticity. Notice that the regression must contain 
an intercept for the test to be useful.

### Input
- `u::Vector`:   T-vector, residuals
- `x::Matrix`:   Txk, regressors

### Output
- `RegrStat::Number`: test statistic
- `pval::Number`:     p-value
- `b::Vector`:        coefficients in regressing `u^2` on w 
- `tstat:Vector`:     t-stats for `b`

"""
function OlsWhitesTest(u,x)

    (T,k) = (size(x,1),size(x,2))

    w = zeros(T,round(Int,k*(k+1)/2))   #matrix of cross products of x
    vv = 1
    for i = 1:k, j = i:k
        w[:,vv] = x[:,i].*x[:,j]        #eg. x1*x1, x1*x2, x2*x2
        vv        = vv + 1
    end

    (b,_,_,V,R²) = OlsGM(u.^2,w)
    df = rank(w) - 1                    #number of independent regressors in w
    WhiteStat = T*R²/(1-R²)
    pval      = ccdf(Chisq(df),WhiteStat)
    #printmat(u,u.^2,w)
    println(R²)
    tstat = b./sqrt.(diag(V))

    return WhiteStat, pval, b, tstat

end


"""
    OlsAutoCorr(u,L=1)

Test the autocorrelation of OLS residuals

### Input
- `u::Vector`:   T-vector, residuals
- `L::Int`:      scalar, number of lags in autocorrelation and Box-Pierce test

### Output
- `AutoCorr::Matrix`:   Lx3, autocorrelation, t-stat and p-value
- `BoxPierce::Matrix`:  1x2, Box-Pierce statistic and p-value
- `DW::Number`:         DW statistic

### Requires
- StatsBase, Distributions

"""
function OlsAutoCorr(u,L=1)

    T = size(u,1)

    Stdu = std(u)
    ρ    = autocor(u,1:L)
    t_ρ  = sqrt(T)*ρ             #t-stat of ρ 

    pval      = 2*ccdf.(Normal(0,1),abs.(t_ρ))
    AutoCorr  = [ρ t_ρ pval]

    BPStat    = T*sum(ρ.^2)
    pval      = ccdf(Chisq(L),BPStat)
    BoxPierce = [BPStat pval]

    DWStat    = mean(diff(u).^2)/Stdu^2

    return AutoCorr, BoxPierce, DWStat

end


"""
    RegressionFit(u,R²,k)

Calculate adjusted R², AIC and BIC from regression residuals.

### Input
- `u::Vector`:      T-vector of residuals
- `R²::Float`:      the R² value
- `k::Int`:         number of regressors

"""
function RegressionFit(u,R²,k)
  T     = length(u)
  σ²    = var(u)
  R²adj = 1 - (1-R²)*(T-1)/(T-k)
  AIC   = log(σ²) + 2*k/T
  BIC   = log(σ²) + k/T * log(T)
  return R²adj, AIC, BIC
end


"""
    VIF(X)

Calculate the variance inflation factor

### Input
- `x::Matrix`:    Txk matrix with regressors

### Output
- `maxVIF::Float`:     highest VIF value
- `allVIF::Vector`:    a k VIF values

"""
function VIF(X)

  k = size(X,2)

  intercept_cols = [first(col) != 0 && allequal(col) for col in eachcol(X)]
  !any(intercept_cols) && throw(ArgumentError("one of the colums of X must be a non-zero constant"))

  R2 = fill(NaN,k)
  for i = 1:k
    if !intercept_cols[i]                 #if X[:,i] is not a constant
      v       = filter(!=(i),1:k)         #exclude i from 1:k
      R2[i] = OlsGM(X[:,i],X[:,v])[5]
    else                                  #if X[:,i] is a constant
      R2[i] = 0
    end
  end

  allVIF = 1.0./(1.0 .- R2)
  maxVIF = maximum(allVIF)

  return maxVIF, allVIF

end


"""
    DiagnosticsTable(X,u,R²,nlags,xNames="")

Compute and print a number of regression diagnostic tests.

### Input
- `X::Matrix`:      Txk matrix of regressors
- `u::Vector`:      T-vector of residuals
- `R²::Float`:      the R² value
- `xNames::Vector`: of strings, regressor names

"""
function DiagnosticsTable(X,u,R²,xNames="")

  (T,k) = size(X)

  isempty(xNames) && (xNames = [string("x",'₀'+i) for i=1:k])    #create rowNames

  printblue("Test of all slopes = 0")
  df = k - 1              #number of slope coefficients
  (RegrStat,pval) = OlsR2Test(R²,T,df)[1:2]
  printmat([RegrStat,pval],rowNames=["stat","p-val"])

  printblue("Measures of fit")
  (R²adj,AIC,BIC)  = RegressionFit(u,R²,k)
  printmat([R²,R²adj,AIC,BIC];rowNames=["R²","R²adj","AIC","BIC"])

  printblue("Test of normality")
  (skewness,kurtosis,JB,pvals) = JarqueBeraTest(u)
  xut = vcat(skewness,kurtosis,JB)
  printmat(xut,collect(pvals);rowNames=["skewness","kurtosis","Jarque-Bera"],colNames=["stat","p-value"])

  printblue("Correlation matrix (checking multicollinearity)")
  printmat(cor(X);colNames=xNames,rowNames=xNames)

  (maxVIF,allVIF) = VIF(X)
  printblue("VIF (checking multicollinearity)")
  printmat(allVIF;rowNames=xNames)

  return nothing

end


"""
    DiagnosticsNoniidTable(X,u,nlags,xNames="")

Compute and print a number of tests for heteroskedasticity and autocorrelation

### Input
- `X::Matrix`:      Txk matrix of regressors
- `u::Vector`:      T-vector of residuals
- `nlags::Int`:     number of lags to use in autocorrelation test
- `xNames::Vector`: of strings, regressor names

"""
function DiagnosticsNoniidTable(X,u,nlags,xNames="")

  (T,k) = size(X)

  isempty(xNames) && (xNames = [string("x",'₀'+i) for i=1:k])    #create rowNames

  printblue("White's test (H₀: heteroskedasticity is not correlated with regressors)")
  (WhiteStat,pval) = OlsWhitesTest(u,X)
  printmat([WhiteStat,pval],rowNames=["stat","p-val"])

  printblue("Testing autocorrelation of residuals (lag 1 to $nlags)")
  (ρStats,BoxPierce,DW) = OlsAutoCorr(u,nlags)
  printmat(ρStats,colNames=["autocorr","t-stat","p-val"],rowNames=1:nlags,cell00="lag")

  printblue("BoxPierce ($nlags lags) ")
  printmat(BoxPierce',rowNames=["stat","p-val"])

  printblue("DW statistic")
  printlnPs(DW,"\n")

  for i in 1:k         #iterate over different regressors
      ρStats, = OlsAutoCorr(X[:,i].*u,nlags)
      printblue("Autocorrelations of $(xNames[i])*u  (lag 1 to $nlags)")
      printmat(ρStats,colNames=["autocorr","t-stat","p-val"],rowNames=1:nlags,cell00="lag")
  end

  return nothing

end
