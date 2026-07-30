"""
    LassoEN(Y,X,γM=0,λ=0,β₀=0)

Do Lasso (set γ>0,λ=0), ridge (set γ=0,λ>0) or elastic net regression (set γ>0,λ>0).
The function loops over the values in a vector γM, but requires λ to be a number (scalar).

### Input
- `Y::Vector`:: T-vector, zero mean dependent variable
- `X::Matrix`:  TxK matrix, zero mean regressors
- `γM::Vector`: nγ-vector with different values of γ (could also be a number)
- `λ::Number`:  value of λ (a number)
- `β₀::Vector`: K-vector of target levels for the coeffs (could also be a common number)


### Remark (details on the coding)

Choice variables `z = [b;t]` with lengths K and K respectively

The objective

`0.5*z'P*z + q'z`

effectively involves

`0.5*z'P*z =  b'(X'X/T+λI)b` and

`q'z = (-2X'Y/T-2λβ₀)'b + γ1't`

The restrictions `lb <= Az <= ub` imply

`-∞ <= β-t    <= β₀`

`β₀ <= β-t    <= ∞`

### Requires
using OSQP, SparseArrays, LinearAlgebra

"""
function LassoEN(Y,X,γM=0,λ=0,β₀=0)

  (T,K) = (size(X,1),size(X,2))
  nγ = length(γM)

  b_ls = X\Y
  βM₀ = isa(β₀,Number) ? fill(β₀,K) : β₀   #expand to vector, if needed

  P1 = X'X/T + λ*I                         #from (Xb-Y)'(Xb-Y) and λb'b
  P2 = -2X'Y/T

  P = blockdiag(sparse(2*P1),spzeros(K,K)) #2* to cancel the 0.5 in 0.5z'Pz
  q = [P2-2*λ*βM₀;zeros(K)]                #we replace zeros(K) by γ (see below)
  A = [sparse(I,K,K)  -sparse(I,K,K);
       sparse(I,K,K)   sparse(I,K,K)]
  lb = [fill(-Inf,K); βM₀]
  ub = [βM₀; fill(Inf,K)]
  prob = OSQP.Model()

  settings = Dict(:verbose => false)
  OSQP.setup!(prob;P=P,q=q,A=A,l=lb,u=ub,settings...)

  bM   = fill(NaN,K,nγ)               #loop over γ values
  for i = 1:nγ
    q[end-K+1:end] .= γM[i]              #change q
    OSQP.update!(prob;q=q)
    res = OSQP.solve!(prob)
    (res.info.status == :Solved) && (bM[:,i] = res.x[1:K])
  end

  return bM, b_ls

end
