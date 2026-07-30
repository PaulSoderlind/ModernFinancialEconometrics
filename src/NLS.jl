
"""
    NLS(NlsF::Function,NlsdF::Function,y,x,par0,m)

Non-linear least squares, using non-linear optimization

### Input
- `NlsF::Function`:    nonlinear regression function, `NlsF(β,x)` where `β`
  are the coefficients and `x` is the data on the regressors, should output T-vector
- `NlsdF::Function`:   derivatives of nonlinear regression function, `NlsdF(β,x)`
- `y::VecOrMat`:       Tx1, data, dependent variable
- `x::VecOrMat`:       Txk, data, regressors
- `par0::Vector`:      initial guess of parameters
- `m::Int`:            number of lags in NW covariance matrix

### Examples
for OLS
- NlsF(β,x)  = vec(x*β)            #Txk kx1
- NlsdF(β,x) = x
for probit:
- NlsF(β,x)  = cdf.(Normal(0,1),vec(x*β))
- NlsdF(β,x) = pdf.(Normal(0,1),vec(x*β)).*x

"""
function NLS(NlsF::Function,NlsdF::Function,y,x,par0,m)

    function NlsLossFn(NlsF::Function,β,y,x)
        T = size(x,1)
        u = y - NlsF(β,x)
        Loss = (u'*u)/T      #to be minimized
        return Loss, u
    end

    T = size(x,1)

    Sol  = optimize(b->NlsLossFn(NlsF,b,y,x)[1],par0,BFGS())
    par1 = Optim.minimizer(Sol)                        #point estimates
    
    u    = NlsLossFn(NlsF,par1,y,x)[2]                    #residuals
    yhat = y - u                                          #fitted values, y = yhat + u
    s    = NlsdF(par1,x)                                  #dF/db

    S   = CovNW(s.*u,m)                                #Cov(s_1u_1+s_2u_2+...)
    Sss = s'*s
    V   = inv(Sss)*S*inv(Sss)                          #VCV

    StdErr = sqrt.(diag(V))
    #printmat(par1,StdErr)

    return par1, StdErr, V, yhat

end
##------------------------------------------------------------------------------
