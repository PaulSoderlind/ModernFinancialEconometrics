"""
    CompanionFormAR(ap)

Companion form of AR(p), that is, write an AR(p) as a VAR(1)
"""
function CompanionFormAR(ap,σ²=1)
    p = length(ap)
    if p > 1                      #if order > 1
        A = vcat(ap',
                 [I  zeros(p-1)])
        Ω      = zeros(p,p)
        Ω[1,1] = σ²
    else                          #if AR(1)
        A = [ap;;]
        Ω = [σ²;;]
    end
    return A,Ω
end


"""
    ARpEst(y,p)

Estimate an AR(p) model (with an intercept) on the data in a vector `y`.

Output: the slope coefficients (not the intercept).

"""
function ARpEst(y,p)

    T    = length(y)

    x = [lag2(y,1:p) ones(T)]       #build matrix of regressors, add constant last
    #printmat([y x][1:10,:])        #uncomment to see the regressors

    b = x[p+1:end,:]\y[p+1:end]     #OLS, cut the first p observatioms
    a = b[1:end-1]                  #slopes

    return a

end


"""
    CompanionFormVAR(Ap,Ωp)

Rewrite VAR(p) as VAR(1), that is, on companion form.

# Input
- `Ap::Array`:   nxnxp matrix of VAR(p) coefficients, i.e. a nxn matrix per lag
- `Ωp::Matrix`:  (optional) nxn covariance matrix of VAR(p) residuals

# Output
- `A::Matrix`:  (n*p)x(n*p) matrix of VAR(1) coefficients
- `Ω::Matrix`:  (n*p)x(n*p) covariance matrix of VAR(1) residuals

"""
function CompanionFormVAR(Ap,Ωp=one(Ap[:,:,1]))

  (n,p) = (size(Ap,1),size(Ap,3))                 #number of variables, order of VAR(p)

  if p > 1                      #if order > 1
    A          = vcat(reshape(Ap,n,n*p),
                      [I  zeros((p-1)*n,n)])
    Ω          = zeros(p*n,p*n)
    Ω[1:n,1:n] = Ωp
  else                                           #if already VAR(1)
    A = copy(reshape(Ap,n,n))                    #guarantee nxn matrix
    Ω = Matrix(Ωp)                               #full matrix, even if I()
  end

  return A,Ω

end


"""
    VAR1IRF(A,L,B;m)

Calculate the impulse response function of a VAR(1) system. Gives (1+L) x n x k output



# Input
- `A::Matrix`:    nxn, VAR(1) matrix in x(t) = A*x(t-1) + e(t)
- `L::Int`:       last period to calculate for
- `B::Matrix`:    (optional) n x k matrix, such that e(t) = B*u(t), where u(t) is kx1.
                  u(t) could, for instance, be structural shocks
- `m::Int`:       output only results for series 1:m

# Output
- `C::3DArray`:   (1+L) x n x k, 3-dimensional matrix with VMA

"""
function VAR1IRF(A,L,B=one(A);m=size(A,2))

  n = size(A,1)
  k = size(B,2)

  C      = fill(NaN,1+L,n,k)     #to put results in
  AA     = copy(B)               #period 0 (same as shock)
  C[1,:,:] = AA
  for s in 2:1+L                 #periods 1,3,...,L
    AA       = A*AA              #A^(s-1) * B, quicker than A^(s-1)
    C[s,:,:] = AA
  end

  if m != n
    C = C[:,1:m,1:m]                         #pick out first m series
    (m == 1) && (C = dropdims(C,dims=(2,3))) #if m==1, to vector
  end

  return C

end


"""
    VAR1Cov(A,Ω)

Calculate unconditional covariance matrix of a VAR(1) system.

# Input
- `A::Matrix`:          nxn, VAR(1) matrix in x(t) = A*x(t-1) + e(t)
- `Ω::Matrix`:          nxn, optional, Cov[e(t)]

"""
function VAR1Cov(A,Ω=one(A))

  n  = size(A,1)

  Δ  = Inf
  Γ₀ = zeros(n,n)
  convcrit = sqrt(eps())

  while Δ > convcrit
    Γ₀_1 = copy(Γ₀)                 #old
    Γ₀   = Ω + A*Γ₀*A'
    Δ    = maximum(abs,Γ₀-Γ₀_1)     #comparing
  end

  return Γ₀

end


"""
    VAR1AutoCov(A,L,Ω;m)

# Input
- `A::Matrix`:          nxn, VAR(1) matrix in x(t) = A*x(t-1) + e(t)
- `L::Int`:             no. lags
- `Ω::Matrix`:          nxn, Cov[e(t)]
- `m::Int`:             output only results for series 1:m

"""
function VAR1AutoCov(A,L,Ω=one(A);m=size(A,2))

  n = size(A,1)

  Γ₀ = VAR1Cov(A,Ω)
  σ  = sqrt.(diag(Γ₀))

  (Γ,R) = (fill(NaN,L,n,n),fill(NaN,L,n,n))
  AA = 1.0I(n)
  for s in 1:L
    AA       = A*AA
    Γ[s,:,:] = AA*Γ₀                     #autocovariances, A^s * Γ₀
    R[s,:,:] = Γ[s,:,:]./(σ*σ')            #autocorrelations
  end

  if m != n
    (Γ₀,Γ,R) = (Γ₀[1:m,1:m],Γ[:,1:m,1:m],R[:,1:m,1:m])
    if m == 1           #to vectors
      (Γ,R) = (dropdims(Γ,dims=(2,3)),dropdims(R,dims=(2,3)))
    end
  end

  return Γ₀,Γ,R

end


"""
    VAR1Forecast(x₀,A,L,Ω;m)

# Input
- `x₀::Vector`:         n-vector
- `A::Matrix`:          nxn, VAR(1) matrix in x(t) = A*x(t-1) + e(t)
- `L::Int`:             no. lags
- `Ω::Matrix`:          nxn, Cov[e(t)]
- `m::Int`:             output only results for series 1:m

"""
function VAR1Forecast(x₀,A,L,Ω=one(A);m=size(A,2))

  n = size(A,1)

  (Ex,Varx) = (fill(NaN,L,n),fill(NaN,L,n,n))
  AA = 1.0I(n)
  Γ  = zeros(n,n)
  for s in 1:L
    AA          = A*AA
    Ex[s,:]     = AA*x₀
    Γ           = A*Γ*A' + Ω
    Varx[s,:,:] = Γ
  end
  Ex   = vcat(vec(x₀)',Ex)             #padding with period 0
  Varx = vcat(zeros(1,n,n),Varx)

  if m != n
    (Ex,Varx) = (Ex[:,1:m],Varx[:,1:m,1:m])
    (m == 1) && (Varx = dropdims(Varx,dims=(2,3)))   #if m==1, to vector
  end

  return Ex,Varx

end
