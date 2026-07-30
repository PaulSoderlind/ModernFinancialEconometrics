"""
Four different kernels for use with the kernel density and regression
"""
GaussianKernel(z)     = exp(-abs2(z)/2)/sqrt(2*pi)
UniformKernel(z)      = ifelse(abs(z) < sqrt(3),1/(2*sqrt(3)),0.0)                   #[-sqrt(3),sqrt(3)]
EpanechnikovKernel(z) = ifelse(abs(z) < sqrt(5),(1-abs2(z)/5)*3/(4*sqrt(5)),0.0)     #[-sqrt(5),sqrt(5)]
TriangularKernel(z)   = ifelse(abs(z) < sqrt(6),(1-abs(z)/sqrt(6))/sqrt(6),0.0)      #[-sqrt(6),sqrt(6)]


"""
    KernelDensity(x,xGrid,h=[],KernelFun=GaussianKernel)

Compute a kernel density estimate at each value of the grid `xGrid`, using the data in vector `x`.
The bandwidth `h` can be specified (otherwise a default value is used). The kernel function
defaults to a standard normal density function, but other choices are available.

"""
function KernelDensity(x,xGrid,h=[],KernelFun=GaussianKernel)

    T = length(x)
    if isempty(h)
        h = 1.06*std(x)/T^0.2
    end

    Ngrid = length(xGrid)                          #number of grid points
    fx    = fill(NaN,Ngrid)
    for j = 1:Ngrid                                #loop over elements in xGrid
        xa    = (x .- xGrid[j])/h
        Kh    = KernelFun.(xa)
        fx[j] = mean(Kh)/h
    end

    Varfx = fx./(T*h) * 1/(2*sqrt(π))
    Stdfx = sqrt.(Varfx)                            #std[f(x)]

  return fx, Stdfx

end


"""
    KernelRegression(y,x,xGrid,h,vv = :all,DoCovb=true,KernelFun=GaussianKernel)

Do kernel regression `y[vv] = b(x[vv])`, evaluated at
each point in the `xGrid` vector, using bandwidth `h`.
Implemented as weighted least squares (WLS), which also provide heteroskedasticity
robust standard errors.

### Input
- `y::Vector`:      T-vector with data for the dependent variable
- `x::Vector`:      T-vector with data for the regressor
- `xGrid::Vector`:  Ngrid-vector with grid points where the estimates are done
- `vv::Symbol or Vector`: If `vv = :all`, then all data points are used, otherwise supply indices.
- `DoCovb::Bool`:    If true, the standard error of the estimate is also calculated
- `KernelFun::Function`: Function used as kernel.

### Remark
- The `vv` and `DoCovb=false` options are useful for speeding up the cross-validation below.

"""
function KernelRegression(y,x,xGrid,h,vv = :all,DoCovb=true,KernelFun=GaussianKernel)

    if vv != :all
        (y,x) = (y[vv],x[vv])
    end

    Ngrid = length(xGrid)                  #number of grid points

    (bHat,StdbHat) = (fill(NaN,Ngrid),fill(NaN,Ngrid))         #b[x(t)]
    for i = 1:Ngrid                        #loop over elements in xGrid
        zi  = (x .- xGrid[i])/h
        w   = KernelFun.(zi)
        w05 = sqrt.(w)
        if DoCovb                          #point estimate and standard error
            (b_i,_,_,Covb_i,) = OlsNW(w05.*y,w05,0)
            bHat[i]    = b_i
            StdbHat[i] = sqrt(Covb_i)
        else                               #point estimate only
            bHat[i] = w05\(w05.*y)
        end
    end

    return bHat, StdbHat

end


"""
    hRuleOfThumb(y,x)

Rule of thumb bandwidth for regressing `y` on `x`.

"""
function hRuleOfThumb(y,x)

    T            = length(y)
    (b,res,)     = OlsGM(y,[x.^2 x ones(T)])
    (σ,γ)        = (std(res), b[1])
    (x_10,x_90)  = quantile(x,[0.1,0.9])             #10th and 90th percentiles

    h_rot = 0.6*σ^(2/5)*abs(γ)^(-2/5)*(x_90-x_10)^(1/5)*T^(-1/5)

    return h_rot
end


"""
    LocalLinearRegression(y,x,xGrid,h,vv = :all,DoCovb=true,KernelFun=GaussianKernel)

Do local linear regression `y = a + b(x-xGrid[i])`, where both `a` and `b` will differ
across `xGrid[i]` values. The estimates of `a` and their standard errors are
exported.

See `KernRegrFn()` for further comments

"""
function LocalLinearRegression(y,x,xGrid,h,vv = :all,DoCovb=true,KernelFun=GaussianKernel)

    if vv != :all
        (y,x) = (y[vv],x[vv])
    end
    c = ones(length(y))

    Ngrid = length(xGrid)                  #number of grid points

    (aHat,StdaHat) = (fill(NaN,Ngrid),fill(NaN,Ngrid))         #b[x(t)]
    for i = 1:Ngrid                        #loop over elements in xGrid
        zi  = (x .- xGrid[i])/h
        w   = KernelFun.(zi)
        w05 = sqrt.(w)
        x2  = hcat(c,x .- xGrid[i])
        if DoCovb
            (b_i,_,_,Covb_i,) = OlsNW(w05.*y,w05.*x2,0)
            aHat[i]    = b_i[1]
            StdaHat[i] = sqrt(Covb_i[1,1])
        else
            b_i     = (w05.*x2)\(w05.*y)
            aHat[i] = b_i[1]
        end
    end

    return aHat, StdaHat

end


"""
  CrossValidateKernelR(y,x,hM)

### Input
- `y::Vector`:      T-vector with data for the dependent variable
- `x::Vector`:      T-vector with data for the regressor
- `hM::Vector`:     Nh-vector of bandwidth values (h) to investigate

"""
function CrossValidateKernelR(y,x,hM)

    T    = length(y)
    Nh   = length(hM)

    EPEM = fill(NaN,T,Nh)
    for t in 1:T
        local v_No_t
        v_No_t = setdiff(1:T,t)     #exclude t from estimation
        for (j,h) in enumerate(hM)                #loop over hM[j] values
            local b_t
            b_t,      = KernelRegression(y,x,x[t],h,v_No_t,false)  #calculate fitted b(x[t])
            EPEM[t,j] = (y[t] - b_t[1])^2     #out-of-sample error for obs t
        end
    end

    EPE = mean(EPEM,dims=1)'

    return EPE
end
