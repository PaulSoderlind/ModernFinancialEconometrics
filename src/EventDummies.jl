"""
    EventDummy(dN_i,dN,p,q)

Create a Tx(p+1+q) matrix of dummies for event days

# Input
- `dN_i::Date or Int`: event Date (DateTime) or index (Int)
- `dN::Vector`:        of Date (DateTime) or Int, matching dN_i. Can also be a StepRange (eg. Date(2000):Month(1):Date(2001))
- `p and q::Int`:      create dummies for p periods before to q periods after event


"""
function EventDummy(dN_i::Union{TimeType,Int},dN,p,q)

  nBefore = count(<(dN_i),dN)               #checking the inputs
  nAfter  = count(>(dN_i),dN)
  (nBefore < p || nAfter<q) && throw(ArgumentError("event window is outside the date vector"))

  dN = dN isa AbstractRange ? collect(dN) : dN    #convert to vector if a StepRange
  T  = length(dN)
  tˣ = findfirst(==(dN_i),dN)    #dN must be a vector, not a StepRange

  D = zeros(Int,T,p+1+q)
  if !isnothing(tˣ)
    p0qM  = vcat(p:-1:0,-1:-1:-q)
    for (i,s) in enumerate(p0qM)
      D[tˣ-s,i] = 1                  #1 in t=3 if tˣ=5 and s=2
    end
  end

  return D,tˣ

end

function EventDummy(dN_i::Union{AbstractArray, Tuple},dN,p,q)

  T = length(dN)
  N = length(dN_i)

  (D,tˣ) = (zeros(Int,T,p+1+q),zeros(Int,N))
  for (i,dN_ii) in enumerate(dN_i)
     (D_,tˣ[i]) = EventDummy(dN_ii,dN,p,q)
     D .|= D_
  end

  return D,tˣ

end
##------------------------------------------------------------------------------

