"""
    TwoSLS(y,x,z,NWQ=true,m=0)

### Input
- `y::VecOrMat`:      Tx1 or T-vector of the dependent variable
- `x::VecOrMat`:      Txk matrix (or vector) of regressors
- `z::VecOrMat`:      TxL matrix (or vector) of instruments
- `NWQ:Bool`:         if true, then Newey-West's covariance matrix is used, otherwise Gauss-Markov
- `m::Int`:           scalar, bandwidth in Newey-West; 0 means White's method

### Output
- `b::Vector`:             k-vector, regression coefficients
- `fnOutput::NamedTuple`:  with
  - res                Tx1 or Txn matrix, residuals y - yhat
  - yhat               Tx1 or Txn matrix, fitted values
  - Covb               matrix, covariance matrix of vec(b) = [beq1;beq2;...]
  - R2                 1xn, R2
  - R2_stage1          k-vector, R2 of each x[:,i] in first stage regression on z
  - δ_stage1           Lxk matrix, coeffs from 1st stage x = z'δ
  - Stdδ_stage1        Lxk matrix, std of δ

### Requires
- Statistics, LinearAlgebra
- CovNW


"""
function TwoSLS(y,x,z,NWQ=true,m=0)

    (Ty,n) = (size(y,1),size(y,2))
    (k,L)  = (size(x,2),size(z,2))

    δ         = z\x             #stage 1 estimates, Lxk, one column per regression
    xhat      = z*δ             #TxL * Lxk - > Txk
    resx      = x - xhat        #Txk
    R2_stage1 = [cor(x[:,i],xhat[:,i])^2  for i=1:k]

    Szz_1 = inv(z'z)             #stage 1 standard errors
    Stdδ  = similar(δ)           #Lxk standard errors of δ
    for i = 1:k                  #loop over columns in x
        if NWQ                   #NW standard errors
            S      = CovNW(resx[:,i].*z,m)
            Covδ_i = Szz_1*S*Szz_1
        else                     #standard errors assuming iid
            Covδ_i = Szz_1*var(resx[:,i])
        end
        Stdδ[:,i] = sqrt.(diag(Covδ_i))
    end

    b    = xhat\y            #stage 2 estimates
    yhat = x*b               #notice: from y=x'b+u, not 2nd stage regression
    res  = y - yhat

    R2   = cor(y,yhat)^2
    Sxz  = x'z              #stage 2 standard errors 
    if NWQ     #Cov(b) using Newey-West 
        S    = CovNW(res.*z,m)
        B    = inv(Sxz*Szz_1*Sxz')*Sxz*Szz_1
        Covb = B*S*B'
    else       #Cov(b) assuming iid residuals, independent of z
        Covb = var(res)*inv(Sxz*Szz_1*Sxz')
    end


    _isxNot0(A,atol=1e-10) = any(x -> !isapprox(x,0;atol),A)      #used in DWH test

    DWHtest = NaN
    cols = findall(_isxNot0,eachcol(resx))             #Durbin-Wu-Hausman test
    k2   = length(cols)
    if k2 >= 1
        resxb = view(resx,:,cols)
        R = [zeros(k2,k) I]
        (bx,_,_,Vx,_) = OlsNW(y,[x resxb],m)          #regress y on (x,residual stage 1)
        DWHtest = (R*bx)'*inv(R*Vx*R')*(R*bx)
    end


    fnOutput = (;res,yhat,Covb,R2,R2_stage1,δ_stage1=δ,Stdδ_stage1=Stdδ,DWHtest)

    return b, fnOutput

end
