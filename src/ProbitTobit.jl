"""
    ProbitLL(b,y,x)

Loglikelihood fn for probit model

"""
function ProbitLL(b,y,x)
  Prob_i = cdfNorm(xbFn(x,b))
  LL_i   = y.*log.(Prob_i) + (1.0.-y).*log.(1.0.-Prob_i)
  return LL_i,Prob_i
end


"""
    LogitLL(b,y,x)

Loglikelihood fn for logit model

"""
function LogitLL(b,y,x)
  Prob_i = logisticFn(xbFn(x,b))
  LL_i   = y.*log.(Prob_i) + (1.0.-y).*log.(1.0.-Prob_i)
  return LL_i,Prob_i
end


"""
    BinLLConst(y)

Loglikelihood of a constant probability model from binary data

"""
function BinLLConst(y)
  p  = mean(y)           #fraction of y=1
  N₀ = count(==(0),y)
  N₁ = count(==(1),y)
  LL = N₀*log(1-p) + N₁*log(p)
  return LL,p
end


"""
    BinaryChoiceR2(y,predHat)

Prediction R2 for a binary choice model
"""
function BinaryChoiceR2pred(y,predHat)
  (_,cTab) = ContingencyTable(y,predHat,[0,1],[0,1];DoRelativeQ=false)
  (N₀,N₁,N) = cTab[:,end]
  (n₀₁,n₁₀) = (cTab[1,2],cTab[2,1])
  p    = N₁/N
  Ndiv = p < 0.5 ? N₁ : N₀
  R2   = 1 - (n₀₁+n₁₀)/Ndiv
  return R2,cTab
end


"""
    TruncRegrLL(par,y,x,c=0)

Loglikelihood of a truncated regression model

"""
function TruncRegrLL(par,y,x,c=0)
  b = par[1:end-1]
  σ = abs.(par[end])
  ϵ = y - xbFn(x,b)
  LLa  = logpdfNorm.(ϵ/σ) .- log(σ)
  LLb  = -log.(cdfNorm((x*b.-c)/σ))
  LL_i = LLa .+ LLb
  return LL_i,ϵ
end


"""
    CensRegrLL(par,y,x,c=0)

Loglikelihood of a censored regression model

"""
function CensRegrLL(par,y,x,c=0)
  b = par[1:end-1]
  σ = abs.(par[end])
  ϵ = y - xbFn(x,b)
  δ   = y .== c            #dummies for censored data points
  LL0 = log.(cdfNorm((c.-x*b)/σ))
  LL1 = logpdfNorm.(ϵ/σ) .- log(σ)
  LL_i = δ.*LL0 + (1.0.-δ).*LL1
  return LL_i,ϵ
end


"""
    xbFn

Calculate x*b when x is a VecOrMat and b is vector.

# Remark
- Notice that x*b fails if x and b both are vectors.
- The current MLE implementation assumes that b is a vector

"""
function xbFn(x,b)
  xb = length(b) == 1 ? x*only(b) : x*b
  return xb
end
