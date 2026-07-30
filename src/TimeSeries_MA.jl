"""
    MAqLL(par,y)

Log likelihood function for MA(q) process. Same as
`LL_i  = -1/2*log(2*π) .- 1/2*log(σ^2) .- 1/2*ϵ.^2/σ^2`

"""
function MAqLL(par::Vector,y)
    (θ,σ) = (par[1:end-1],abs(par[end]))
    q     = length(θ)
    ϵ     = ARMAFilter(y,-θ)                #ϵ is AR(q) with coefs -θ
    LL_i  = logpdfNorm.(ϵ./σ) .- log(σ)     #log of N(0,1) pdf - log(σ)
    return LL_i, ϵ
end


"""
    MAqToAutocorr(θ,L)

Calculate implied `L` autocorrerlations from MA(q) model, where
`y[t] = ϵ[t] + θ1*ϵ[t-1] + ... +  θq*ϵ[t-q]`
"""
function MAqToAutocorr(θ,L)

  q = length(θ)
  θ = vcat(1,θ)           #θ₀ = 1

  γ = fill(NaN,1+L)       #variance and L aucovariances
  for s in 0:L
    γ_s = 0
    for j in 0:q-s
      γ_s += θ[1+j+s]*θ[1+j]     #+1 to conform with 1-based indexing
    end
    γ[1+s] = γ_s
  end
  ρ = γ[2:end]/γ[1]

  return ρ

end


"""
    YuleWalker(γ0p)

Calculate the autoregressive coeffs from the autocovariances by using the Yule-Walker equations.

# Arguments
- `γ0p`        1+p vector, autocovariances or autocorrelations from lag 0 to p

# Output
- `a`          p vector,  autoregressive coeffs from lag 1 to p

"""
function YuleWalker(γ0p)

  γ0p_1 = γ0p[1:end-1]          #autocovariances lag 0 to p-1
  γ1p   = γ0p[2:end]            #autocovariances lag 1 to p

  G2 = Toeplitz(γ0p_1)           #pxp Toeplitz matrix
  a  = G2\γ1p                    #a = inv(G2)*γ1p, but \ is better

  return a

end


"""
    Toeplitz(vc)

Create a symmetric Toeplitz matrix.

"""
function Toeplitz(v::Vector)
    n = length(v)
    T = similar(v,n,n)
    for i in 1:n, j in 1:n
        T[i,j] = v[abs(i-j) + 1]
    end
    return T
end
