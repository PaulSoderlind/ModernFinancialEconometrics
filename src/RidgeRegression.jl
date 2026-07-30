"""
    RidgeRegression(Y,X,λ,β₀=0)

Calculate ridge regression estimate with penalty vector `λ` and target vector `β₀`.
"""
function RidgeRegression(Y,X,λ,β₀=0)
    (T,K) = (size(X,1),size(X,2))
    isa(λ,Number)  && (λ=fill(λ,K))
    isa(β₀,Number) && (β₀=fill(β₀,K))
    Λ = diagm(λ)
    b = (X'X/T+Λ)\(X'Y/T+Λ*β₀)         #same as inv(X'X/T+Λ)*(X'Y/T+Λ*β₀)
    return b
end
