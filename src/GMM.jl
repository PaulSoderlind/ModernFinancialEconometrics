"""
    meanV(x)

Calculate the sample average of each column of `x` and return a vector of them.

"""
function meanV(x)
    xbar = vec(mean(x,dims=1))    #mean of each column, transformed into a vector
    return xbar
end


"""
    GMMExactlyIdentified(GmmMomFn::Function,x,par0,m)

Estimates GMM coeffs and variance-covariance matrix from an exactly identified model. The Jacobian
is calculated numerically.

### Input
- `GmmMomFn::Function`:    for the moment conditions, called as `GmmMomFn(p,x)` where `p`
  are the coefficients and `x` is the data.
- `x::VecOrMat`:            data
- `par0::Vector`:           initial guess
- `m::Int`:                 number of lags in NW covariance matrix

"""
function GMMExactlyIdentified(GmmMomFn::Function,x,par0,m)

    T = size(x,1)

    #Sol  = nlsolve(p->meanV(GmmMomFn(p,x)),par0)   #numerically solve for the estimates
    #par1 = Sol.zero
    par1  = nlsolvePs(p->meanV(GmmMomFn(p,x)),par0)   #numerically solve for the estimates

    g = GmmMomFn(par1,x)        #Tx2, moment conditions
    Σ = CovNW(g,m,1)          #variance of sqrt(T)*gbar, NW with m lags

    D   = jacobian(par->meanV(GmmMomFn(par,x)),par1)  #Numerical Jacobian
    V_T = inv(D'inv(Σ)*D)/T
    StdErr = sqrt.(diag(V_T))

    return par1, StdErr, V_T, Σ

end


"""
    GMMgbarWgbar(GmmMomFn::Function,W,x,par0,m;SkipCovQ=false)

Estimates GMM coeffs and variance-covariance matrix from gbar'W*gbar. The Jacobian
is calculated numerically.

### Input
- `GmmMomFn::Function`:    for the moment conditions, called as `GmmMomFn(p,x)` where `p`
  are the coefficients and `x` is the data.
- `W::Matrix`:              length(gbar)xlength(gbar)
- `x::VecOrMat`:            data
- `par0::Vector`:           initial guess
- `m::Int`:                 number of lags in NW covariance matrix
- `SkipCovQ::Bool`:         if true: the Jacobian and variance-covariance matrix are
  not calculated. This can be used to speed up calculations in iterative computations.

"""
function GMMgbarWgbar(GmmMomFn::Function,W,x,par0,m;SkipCovQ=false)

    function GmmMomLossFn(GmmMomFn::Function,p,x,W=1)
        gbar = meanV(GmmMomFn(p,x))
        Loss = gbar'W*gbar      #to be minimized
        return Loss
    end

    T = size(x,1)

    Sol  = optimize(p->GmmMomLossFn(GmmMomFn,p,x,W),par0,BFGS())
    par1 = Optim.minimizer(Sol)

    g = GmmMomFn(par1,x)        #Txq, moment conditions
    Σ = CovNW(g,m,1)            #variance of sqrt(T)*gbar, NW with m lags

    if SkipCovQ
        (D,V_T,StdErr) = (NaN,NaN,NaN)
    else
        D      = jacobian(par->meanV(GmmMomFn(par,x)),par1)  #Numerical Jacobian
        Γ      = inv(D'W*D)*D'W
        V_T    = Γ*Σ*Γ'/T
        StdErr = sqrt.(diag(V_T))
        if size(g,2) > length(par0)            #overidentified
            Ψ     = (I-D*Γ)*Σ*(I-D*Γ)'
            gbar  = meanV(GmmMomFn(par1,x))
            gTest = T*gbar'pinv(Ψ)*gbar
        else
            gTest = NaN
        end
    end

    return par1, StdErr, V_T, Σ, D, gTest

end


"""
    GMMAgbar(GmmMomFn::Function,A,x,par0,m)

Estimates GMM coeffs and variance-covariance matrix from A*gbar. The Jacobian
is calculated numerically.

### Input
- `GmmMomFn::Function`:    for the moment conditions, called as `GmmMomFn(p,x)` where `p`
  are the coefficients and `x` is the data.
- `A::Matrix`:              length(p) x length(gbar)
- `x::VecOrMat`:            data
- `par0::Vector`:           initial guess
- `m::Int`:                 number of lags in NW covariance matrix

"""
function GMMAgbar(GmmMomFn::Function,A,x,par0,m)

    T = size(x,1)

    #Sol  = nlsolve(p->A*meanV(GmmMomFn(p,x)),par0)   #numerically solve for the estimates
    #par1 = Sol.zero
    par1 = nlsolvePs(p->A*meanV(GmmMomFn(p,x)),par0)   #numerically solve for the estimates

    g = GmmMomFn(par1,x)        #Tx2, moment conditions
    Σ = CovNW(g,m,1)          #variance of sqrt(T)*gbar, NW with m lags

    D      = jacobian(par->meanV(GmmMomFn(par,x)),par1)  #Numerical Jacobian
    Γ      = inv(A*D)*A
    V_T    = Γ*Σ*Γ'/T
    StdErr = sqrt.(diag(V_T))
    if size(g,2) > length(par0)            #overidentified
        Ψ     = (I-D*Γ)*Σ*(I-D*Γ)'
        gbar  = meanV(GmmMomFn(par1,x))
        gTest = T*gbar'pinv(Ψ)*gbar
    else
        gTest = NaN
    end

    return par1, StdErr, V_T, Σ, gTest

end


"""
    nlsolvePs(fn::Function,x0,ConCrit=1e-6)

Solve the function `fn` for the vector of roots, using Optim.jl rather than NLsolve.jl.
"""
function nlsolvePs(fn::Function,x0,ConCrit=1e-6)
  Sol = optimize(p->sum(abs2,fn(p)),x0,BFGS())
  x1   = Optim.minimizer(Sol)
  res = fn(x1)
  x = maximum(abs.(res)) < ConCrit ? x1 : NaN
  return x
end
