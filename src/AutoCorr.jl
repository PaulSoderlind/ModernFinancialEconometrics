"""
    AutoCorr(x,plags=1)

Estimate autocorrelation function for (handling NaNs/missings)


### Input
- `x::Vector`                 data
- `plags::Number or Vector`:  K vector, lags

### Output
- `autoc::vector or Number`:  K-vector (or Number if K==1) of autocorrelation coefficients

"""
function AutoCorr(x,plags=1)

  K = length(plags)

  vv   = FindNN(x)
  Varx = var(x[vv],corrected=false)     #variance
  xbar = mean(x[vv])                    #mean

  autoc = fill(NaN,K)
  for (i,p) in enumerate(plags)
    x_p       = lag(x,p)
    vv        = FindNN(x,x_p)           #find record with non-missings
    Ti        = sum(vv)
    autcov_p  = dot(x[vv].-xbar,x_p[vv].-xbar)/Ti
    autoc[i]  = autcov_p./Varx                   #autocorrelation
  end

  (length(autoc) == 1) && (autoc = only(autoc))

  return autoc

end


"""
    AutoCorrOls(x,plags,m)

Autocorrelations from OLS of x[t+s] = a + bx[t] + u[t+s], for a single or several s values

## Input
- `x::Vector`:         data
- `plags::Vector`:     integers, lags to estimate corrlations for
- `m::Number`:         bandwidth in Newey-West VCV

## Output
- `ρ::Vector`:             autocorrelation coeffs, estimated by OLS
- `Stdρ::Vector`:          #standard errors
- `Covρ::Matrix`:          #VCV
- `BPstat::Number`:        #Box-Pierce statistocs


"""
function AutoCorrOls(x,plags=1,m=maximum(plags))

  (size(x,2) > 1) && error("x must be vector or Tx1 matrix")

  L = length(plags)

  R = zeros(Int,L,2*L)
  for i in 1:L                   # 1 0 0 0
    R[i,(i-1)*2+1] = 1           # 0 0 1 0, etc to pick out slopes
  end
  #printmat(R)

  xL = lag2(x,plags)
  (xL,x) = excise(xL,x)          #cut rows with any NaN
  T = size(x,1)
  #printmat([xL x][1:3,:])

  (b12,_,_, V,) = OlsSure(xL,[x ones(T)],true,m)      #[x[t+1] x[t+2] etc] on x[t], NW VCV

  ρ     = R*vec(b12)               #same as b12[1,:]
  Covρ  = R*V*R'                   #under iidN, T*Covρ = I
  Stdρ  = sqrt.(diag(Covρ))
  BPstat = ρ'*inv(Covρ)*ρ

  return ρ, Stdρ, Covρ, BPstat

end


"""
    PartialAutoCorr(x,plags=1)

Partial autocorrelation coeffcients

"""
function PartialAutoCorr(x,plags=1)

  (size(x,2) > 1) && error("x must be vector or Tx1 matrix")

  T = size(x,1)
  L = length(plags)

  pautoc = fill(NaN,L)
  for (i,s) in enumerate(plags)
    x_s       = [ones(T) lag2(x,1:s)]        #[1,x(t-1),x(t-2),...]
    b,        = OlsBasic(x,x_s,true)         #OLS, handles missing values
    pautoc[i] = b[end]                       #partial autocorrelation s
  end

  return pautoc

end
