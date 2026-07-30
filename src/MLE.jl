"""
    MLE(LLtFun,par0,y,x,lower,upper)

Calculate ML point estimates of K parameters and three different types of standard errors:
from the Information matrix, from the gradients and the sandwich approach.

### Input
- `LLtFun::Function`:  name of log-likelihood function
- `par0::Vector`:      K-vector, starting guess of the parameters
- `y::VecOrMat`:       vector or matrix with the dependent variable
- `x::VecOrMat`:       vector or matrix with data, use `nothing` if not needed
- `lower::Vector`:     lower bounds on the parameters, nothing or fill(-Inf,K) if no bounds
- `upper::Vector`:     upper bounds on the parameters, nothing or fill(Inf,K) if no bounds

### Requires
- `using FiniteDiff: finite_difference_hessian as hessian, finite_difference_jacobian as jacobian`

### Notice
The `LLtFun` should take `(par,y,x)` as inputs and generate a T-vector `LLt` as output.

"""
function MLE(LLtFun::Function,par0,y,x,lower=nothing,upper=nothing)

    T    = size(y,1)

    NoBounds = (isnothing(lower) && isnothing(upper)) || (all(!isfinite,lower) && all(!isfinite,upper))

    if NoBounds
        Sol = optimize(par->-sum(LLtFun(par,y,x)),par0,BFGS()) #minimize -sum(LLt)
    else
        Sol = optimize(par->-sum(LLtFun(par,y,x)),lower,upper,par0,Fminbox(BFGS()))
    end
    parHat = Optim.converged(Sol) ? Optim.minimizer(Sol) : fill(NaN,length(par0))  #the optimal solution
    LL_t = LLtFun(parHat,y,x)

    Ia = -hessian(par->mean(LLtFun(par,y,x)),parHat)   #2nd derivatives of mean(LLt)
    Ia       = (Ia+Ia')/2                            #to guarantee symmetry
    vcv      = inv(Ia)/T
    std_hess = sqrt.(diag(vcv))

    δL       = jacobian(par->LLtFun(par,y,x),parHat)   #TxK
    J        = δL'δL/T                               #KxT * TxK
    vcv      = inv(J)/T
    std_grad = sqrt.(diag(vcv))

    vcv       = inv(Ia) * J * inv(Ia)/T
    std_sandw = sqrt.(diag(vcv))                     #std from sandwich

   return parHat, std_hess, std_grad, std_sandw, LL_t

end
