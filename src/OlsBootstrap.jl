"""
    OlsBootstrap(y,x,bLS,BlockSize,NSim,ExciseIt=false;yxPairType=0)

Do a bootstrap of OLS regression, w/wo blocks and of residual/pairs for (y,x)

# Input
- `y::Array`:            Txn matrix of n dependent variables
- `x::Array`:            TxK matrix of (common) regressors
- `bLS::Array`:          OLS estimates (will be generated if not supplied)
- `BlockSize::Int`:      scalar, size of blocks for block bootstrap
                         (eg. 1 for no blocks, 20 for long blocks)
- `NSim::Int`:           scalar, number of simulations
- `ExciseIt::Bool`:      true for excise on [y,x]; default: false
- `yxPairType::Int`:     1 for drawing pairs of (y(s),x(s)); 2 for wild bootstrap;
  else residuals are drawn; default: 0

# Output
- `CovbBoot::Array`:     K*n x K*n covariance matrix of vec(b)
- `bBoot::Array`:        NSim x K*n matrix, vec(b) in row i

# Requires
- FindNN

"""
function OlsBootstrap(y,x,bLS,BlockSize,NSim,ExciseIt=false;yxPairType=0)

  n = size(y,2)

  if ExciseIt
    vv    = FindNN(y,x)
    (y,x) = (y[vv,:],x[vv,:])
  end

  if any(isunordered,y) || any(isunordered,x)
    @warn("OlsBootstrapPs: there are NaNs/missings")
  end

  if isempty(bLS)
    bLS = x\y
  end
  yhat = x*bLS
  res  = y - yhat

  (T,K) = (size(x,1),size(x,2))

  (yxPairType == 2) && (z_i = zeros(Int,T))
  (yxPairType != 1) && (y_i = fill(NaN,T,n))
  bBoot = fill(NaN,NSim,K*n)                            #vec(b), [beq1;beq2;..;beqn]
  for i = 1:NSim                       #loop over simulations
    local b_i
    if yxPairType == 1                                  #draw pairs of (y(s),x(s))
      t_j  = BootstrapBlock(T,BlockSize)
      b_i  = view(x,t_j,:)\view(y,t_j,:)
    elseif yxPairType == 2                              #wild bootstrap
      Random.rand!(z_i,[-1;1])
      y_i .= yhat .+ res.*z_i
      b_i  = x\y_i
    else                                                #draw residuals and create y(s)
      t_j  = BootstrapBlock(T,BlockSize)
      y_i .= yhat .+ view(res,t_j,:)
      b_i  = x\y_i
    end
    bBoot[i,:] = b_i
  end   #i

  #Avgb = mean(bBoot)
  CovbBoot = cov(bBoot)

  return CovbBoot,bBoot

end
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
"""
    BootstrapBlock(T,BlockSize=1,TSim=T)

Calculates (row) indices for a block boostrap simulation


# Input
- `T::Int`:          sample size of data, the data is drawn from 1:T
- `BlockSize::Int`:  number of periods in each block [1]
- `TSim::Int`:       sample size to simulate [TSim=T].
                     Notice that T>TSim, T==TSim and T<TSim all work.

# Output
- `v::Vector{Int}`:  TSim-vector of indices (in blocks)

# Remark
- could also use vcat([range(i,length=BlockSize) for i in v0]...)

"""
function BootstrapBlock(T,BlockSize=1,TSim=T)

  nBlocks = cld(TSim,BlockSize)                #how many blocks
  Tv      = nBlocks*BlockSize
  v       = zeros(Int,Tv)

  if BlockSize == 1
    Random.rand!(v,1:T)
  else
    v0 = rand(1:T,nBlocks)                                   #starting row of blocks
    for (t,value) in enumerate(v0)
      i    = (t-1)*BlockSize+1:t*BlockSize
      v[i] = value:value+BlockSize-1
    end
    #v  = vec(v0' .+ vec(0:BlockSize-1))                     #alternative approach
    v = replace(z -> z>T ? z-T : z,v)                        #wrap around if index > T
  end

  if Tv > TSim                                               #get exact sample length
    v = deleteat!(v,TSim+1:Tv)
  end

  return v

end
#------------------------------------------------------------------------------
