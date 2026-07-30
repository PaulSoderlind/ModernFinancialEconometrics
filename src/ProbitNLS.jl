LpmRegrF(β,x)  = vec(x*β)            #OLS as NLS, just to check
LpmdRegrF(β,x) = x
##----------------------------------------------------------

ProbitRegrF(β,x)  = cdf.(Normal(0,1),vec(x*β))            #probit
ProbitdRegrF(β,x) = pdf.(Normal(0,1),vec(x*β)).*x

function ProbitMeF(β,x,Binaryx=[])
  me = pdf.(Normal(0,1),vec(x*β)).*β
  if !isempty(Binaryx)                    #handle binary regressors
    for i in Binaryx
      x1 = copy(x)
      x1[i] = 1
      F1 = ProbitRegrF(β,x1)
      x1[i] = 0
      F0 = ProbitRegrF(β,x1)
      me[i] = only(F1 - F0)
    end
  end
  return me
end
##----------------------------------------------------------

"""
    logisticFn(v)

logistic function

"""
logisticFn(v) = 1.0/(1.0 + exp(-v))

LogitRegrF(β,x) = logisticFn.(vec(x*β))            #logit

function LogitdRegrF(β,x)
  F = logisticFn.(vec(x*β))
  s = (1.0.-F).*F.*x
  return s
end

function LogitMeF(β,x,Binaryx=[])
  F = LogitRegrF(β,x)
  me = (1.0.-F).*F.*β
  if !isempty(Binaryx)                    #handle binary regressors
    for i in Binaryx
      x1 = copy(x)
      x1[i] = 1
      F1 = logisticFn.(vec(x1*β))
      x1[i] = 0
      F0 = logisticFn.(vec(x1*β))
      me[i] = only(F1 - F0)
    end
  end
  return me
end
##------------------------------------------------------------------------------
