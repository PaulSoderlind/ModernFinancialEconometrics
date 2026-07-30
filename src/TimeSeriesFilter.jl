"""
    EMA(x,m=1)

Equally weighted moving average (over the current and m-1 lagged values) of each column
in a matrix `x`.

"""
function EMA(x,m=1)

    (T,n) = (size(x,1),size(x,2))
    q = m - 1                               #m=2, q=1 (avg over [t-1,t])

    (q < 0) && error("q must be >= 0")

    y = fill(NaN,T,n)                       #y[t] = (x[t-1] + x[t])/2
    for t = 1:T
        vv     = max(1,t-q):t                 #with q=2; 1:1,1:2,1:3,2:4
        y[t,:] = mean(x[vv,:],dims=1)
    end

    return y

end


"""
    ARMAFilter(ϵ,rho=[],θ=[],θ₀=1.0,y₀=[])

Calculate ARMA(p,q) transformation of an input series ϵ. Uses explicit loop (instead of
DSP.filt).


### Input
- `ϵ::Vector`:      T-vector with an input series
- `rho::Vector`:    (optional) p-vector of autoregression coefficients, could be []
- `θ::Vector`:      (optional) q-vector of moving average coefficients (lag 1 to q), could be []
- `θ₀::Number`:     (optional) scalar, coefficient on ϵ[t] in MA part, [1]
- `y₀::Vector`:     (optional) p-vector, initial values of y, eg. [y(-1);y(0)], default: zeros(p)

### Output
- `y::Vector`:      T-vector with output from the filter

### Notice
1. The process is
   `y[t] = rho[1]*y[t-1] + ... + rho[p]*y[t-p] + θ₀*ϵ[t]  +  θ[1]*ϵ[t-1] + ... + θ[q]*ϵ[t-q]`
2. The initial values of `ϵ` are assumed to be zero
3. To calculate impulse response functions, use `ϵ = [1;zeros(T-1,1)]`
4. There are no initial values in a pure MA and the case of `q > p`
   is handled with padding with zeros (see the code below)

"""
function ARMAFilter(ϵ,rho=[],θ=[],θ₀=1,y₀=[])

  T     = length(ϵ)
  (q,p) = (length(θ),length(rho))
  r     = max(p,q)                         #r obs are needed for start-up

  if (p > 0) && isempty(y₀)                #default y₀
    y₀ = zeros(p)
  end
  (length(y₀) != p) && error("length of y₀ must equal p")

  θ_ud = q > 0 ? reverse([θ₀;θ]) : θ₀  #flip upside down
  ρ_ud = p > 1 ? reverse(rho)    : rho

  ϵ  = [zeros(r);ϵ]                          #pad with zeros, clumsy but easy
  Tb = length(ϵ)                             #effective sample with padding

  y = r > p ? [zeros(r-p);y₀;zeros(T)] : [y₀;zeros(T)]      #to store output in

  for t in r+1:Tb
      if p == 0                              #pure MA
        y[t] = dot(θ_ud,ϵ[t-q:t])
      else                                   #AR or ARMA
        y[t] = dot(ρ_ud,y[t-p:t-1]) + dot(θ_ud,ϵ[t-q:t])
    end
  end
  y = y[r+1:Tb]                              #cut padded part

  return y

end


"""
    VARFilter(ϵ,A,y₀)

Create y Txn matrix from VAR model where
`y[t,:] = A1*y[t-1,:] +...+Ap*y[t-p,:] + ϵ[t,:]`

A is an `nxnxp` array with `cat(A1,A2,...,dims=3)`
`y₀` is `pxn` initial values of y (for [t=-2;t=-1;t=0] for a VAR(3))
"""
function VARFilter(ϵ,A,y₀)

    p     = size(A,3)            #lag order
    (T,n) = (size(ϵ,1),size(ϵ,2))
    (p != size(y₀,1)) && error("y₀ must be $p x $n")

    y     = [y₀;zeros(T,n)]
    ϵ     = [zeros(p,n);ϵ]
    Tb    = size(y,1)
    for t in p+1:Tb               #loop over t
        for s in 1:p              #loop over lags, y[t-1],y[t-2],...,y[t-p]
            y[t,:] = y[t,:] + A[:,:,s]*y[t-s,:]
        end
        y[t,:] = y[t,:] + ϵ[t,:]
    end
    #printmat(y)
    y = y[p+1:end,:]                #cut padded part

    return y

end

