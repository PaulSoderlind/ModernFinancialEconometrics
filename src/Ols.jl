"""
    OlsGM(Y,X)

LS of Y on X; for one dependent variable, Gauss-Markov assumptions

### Input
- `Y::Vector`:    T-vector, the dependent variable
- `X::Matrix`:    Txk matrix of regressors (including deterministic ones)

### Output
- `b::Vector`:    k-vector, regression coefficients
- `u::Vector`:    T-vector, residuals Y - yhat
- `Yhat::Vector`: T-vector, fitted values X*b
- `V::Matrix`:    kxk matrix, covariance matrix of b
- `R²::Number`:   scalar, R² value

"""
function OlsGM(Y,X)

    (Y,X) = excise(Y,X)             #keep obs with non-missings
    T     = size(Y,1)

    b    = X\Y
    Yhat = X*b
    u    = Y - Yhat

    σ²   = var(u;corrected=false)        #using 1/T, not 1/(T-1)
    V    = inv(X'X)*σ²
    R²   = 1 - var(u)/var(Y)

    return b, u, Yhat, V, R²

end


"""
    OlsNW(Y,X,m=0)

LS of Y on X; for one dependent variable, using Newey-West covariance matrix

### Input
- `Y::Vector`:    T-vector, the dependent variable
- `X::Matrix`:    Txk matrix of regressors (including deterministic ones)
- `m::Int`:       scalar, bandwidth in Newey-West

### Output
- `b::Vector`:    k-vector, regression coefficients
- `u::Vector`:    T-vector, residuals Y - Yhat
- `Yhat::Vector`: T-vector, fitted values X*b
- `V::Matrix`:    kxk matrix, covariance matrix of b
- `R²::Number`:   scalar, R² value

"""
function OlsNW(Y,X,m=0)

    (Y,X) = excise(Y,X)             #keep obs with non-missings
    T    = size(Y,1)

    b    = X\Y
    Yhat = X*b
    u    = Y - Yhat

    S    = CovNW(X.*u,m)           #Newey-West covariance matrix
    Sxx  = X'X
    V    = inv(Sxx)'S*inv(Sxx)     #covariance matrix of b
    R²   = 1 - var(u)/var(Y)

    return b, u, Yhat, V, R²

end


"""

    OLSyxReplaceNaN(Y,X)

Replaces any rows in Y and X with zeros if there is any NaN/missing in any of them.

"""
function OLSyxReplaceNaN(Y,X)

  vv = FindNN(Y,X)             #vv[t] = true if no missing/NaN i (y[t],x[t,:])

  (Yb,Xb)     = (copy(Y),copy(X))    #set both y[t] and x[t,:] to 0 if any missing/NaN for obs. t
  Yb[.!vv]   .= 0
  Xb[.!vv,:] .= 0

  return vv, Yb, Xb

end


"""
    OlsBasic(Y,X,ExciseQ=false)

LS of Y on X; for one dependent variable, only point estimates, fitted values and residuals.
Optionally (`ExciseQ=true`) handles missing values/NaNs.

### Input
- `Y::Vector`:      T-vector, the dependent variable
- `X::Matrix`:      Txk matrix of regressors (including deterministic ones)
- `ExciseQ::Bool`:  true: get rid of missing/NaN cases in estimation, but put them back in (u,Yhat)

### Output
- `b::Vector`:    k-vector, regression coefficients
- `u::Vector`:    T-vector, residuals Y - yhat
- `Yhat::Vector`: T-vector, fitted values X*b

"""
function OlsBasic(Y,X,ExciseQ=false)

    if ExciseQ
        (vv,Y,X) = OLSyxReplaceNaN(Y,X)            #creates new (Y,X)
    end
    b    = X\Y
    Yhat = X*b
    ExciseQ && (Yhat[.!vv] .= NaN)                 #puts obs with missings/NaN to NaN
    u    = Y - Yhat

    return b, u, Yhat

end


"""
    RegressionTable(b,V,xNames="")

### Input
- `b::Vector`:       of k point estimates
- `V::Matrix`:       kxk variance-covariance matrix of b
- `xNames::Vector`:  of k strings, variable names

### Requires
- Distributions, LinearAlgebra, Printf

"""
function RegressionTable(b,V,xNames="")

  k = length(b)
  isempty(xNames) && (xNames = [string("x",'₀'+i) for i=1:k])    #create rowNames

  stderr = sqrt.(diag(V))
  tstat = b./stderr
  pval  = 2*ccdf.(Normal(0,1),abs.(tstat))    # ccdf(x) = 1-cdf(x)

  colNames = ["coef","stderr","t-stat","p-value"]
  printmat(b,stderr,tstat,pval;rowNames=xNames,colNames)

end
