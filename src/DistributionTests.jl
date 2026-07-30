"""
    JarqueBeraTest(x)

Calculate the JB test for each column in a matrix. Reports `(skewness,kurtosis,JB)`.

"""
function JarqueBeraTest(x)
    (T,n) = (size(x,1),size(x,2))    #number of columns in x
    μ     = mean(x,dims=1)
    σ     = std(x,dims=1,corrected=false)
    xStd  = (x .- μ)./σ               #first normalize to a zero mean, unit std variable
    skewness = mean(z->z^3,xStd,dims=1)
    kurtosis = mean(z->z^4,xStd,dims=1)
    JB       = (T/6)*skewness.^2 + (T/24)*(kurtosis.-3).^2   #Jarque-Bera, Chisq(2)
    if n == 1 
        (skewness,kurtosis,JB) = (only(skewness),only(kurtosis),only(JB))  #to numbers if n=1
    end

    pskew = 2*ccdf.(Normal(0,sqrt(6/T)), abs.(skewness))
    pkurt = 2*ccdf.(Normal(3,sqrt(24/T)),abs.(kurtosis))
    pJB   = ccdf.(Chisq(2),JB)
    pvals = (;pskew,pkurt,pJB)

    return skewness, kurtosis, JB, pvals
end   


"""
    KolSmirTest(x1,TheoryCdf::Function)

Calculate the Kolmogorov-Smirnov test

### Output
- `KSstat::Float64`:     KS test statistic
- `xD::Number`:          x value with the largest diff beteen empirical and theoretical cdf

"""
function KolSmirTest(x1,TheoryCdf::Function)

    T        = length(x1)
    !issorted(x1) && (x1 = sort(x1))

    TheoryCdf_x  = TheoryCdf.(x1)
    edfH         = 1/T:1/T:1                        #empirical cdf for x1
    edfL         = 0:1/T:(1-1/T)

    D_candidates = [edfH;edfL] - repeat(TheoryCdf_x,2)
    (D,vD)       = findmax(abs,D_candidates)

    KSstat       = sqrt(T)*D

    if vD <= T                                      #if max is at a jump
        xD = x1[vD]
    else                                            #if max is just before a jump
        xD = prevfloat(x1[vD-T])
    end

    return KSstat, xD

end
