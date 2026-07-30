"""
    rankPs(x)

Calculates the ordinal rank of eack element in a vector `x`. As an aternative,
use `ordinalrank` from the `StatsBase.jl` package.

"""
rankPs(x) = invperm(sortperm(x))


"""
    sortLoHi(x,v,m)

Create vectors `vL` and `vH` with trues/falses indicating membership of the Lo and Hi
groups. It sorts according to `x[v]`, setting the m lowest (in `vL`) and m
highest values (in `vH`) to `true`. All other elements 
(also those in `x[.!v]`) are set to false.

### Input
- `x::Vector`:    n-vector, sorting variable
- `v::Vector`:    n-vector of true/false. Sorting is done within x[v]
- `m::Int`:       number of assets in Lo/Hi portfolio

### Output
- `vL::Vector`:   n-vector of true/false, indicating membership of Lo portfolio
- `vH::Vector`:   n-vector of true/false, indicating membership of Hi portfolio

"""
function sortLoHi(x,v,m)

    x  = Float64.(x)           #independent copy of the input (x), float
    nv = sum(v)
    (nv < 2m) && error("sum(v) < 2m")

    x[.!v] .= Inf                     #v[i] = false are put to Inf to sort last
    r       = rankPs(x)               #lowest are first
    vL      = r .<= m                 #eg. 1:5, with m=5
    vH      = (nv-m+1) .<= r .<= nv   #eg. 8:12, with nv=12
    
    return vL, vH
end


"""
    EWportf(v)

Create (equal) portfolio weights from a vector of trues/falses. If all elements are falses,
then the weights are NaNs.

### Examples
- EWportf([true,false,true]) gives [0.5,0.0,0.5]. 
- EWportf([false,false]) gives [NaN,NaN]

"""
function EWportf(v) 
    w = ifelse( all(.!v), fill(NaN,length(v)), v/sum(v) )
    return w
end
