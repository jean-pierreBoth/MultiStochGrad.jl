# logistic regression use case

using LinearAlgebra




# We have to implement 
# (observations: Observations, position: Array{Float64}, term : Int64) -> Float64
"""
# function term_value(observations:: Observations, position:: Array{Float64}, term :: Int64)

Our position data is encoded in an array of diemsnsion (nbclass - 1 ,  1 + length of an observation)
```math
 term value = log(1+\\sum_{1}^{K-1} exp(a_{i} \\dot x_{k})) - \\sum_{k=1}^{K-1} 1_{y_{i}=k} a_{i} \\dot x_{k}
```
"""
function logistic_term_value(observations:: Observations, position:: Array{Float64,2}, term :: Int64)
    #
    dims = size(position)
    nbclass = dims[2]
    #
    term_data = observations.datas[term]
    term_class = round(Int64, observations.value_at_data[term])
    # compute dot_k[k] = dot(position[:,k], obs_term)
    dot_k = zeros(Float64, nbclass)
    BLAS.gemm!('T', 'N', 1., position, term_data, 0., dot_k)
    logarg = 1. + sum(exp.(dot_k))
    #
    other_term = 0.
    # This relies on our convention that class 0 bears the identifiability constraint.
    if term_class >= 1
        other_term  = dot_k[term_class]
    end
    log(logarg) - other_term
end



"""
# function logistic_term_gradient(observations:: Observations, position:: Array{Float64,2}, 
        term :: Int64, gradient ::  Array{Float64,2})


"""
function logistic_term_gradient(observations:: Observations, position:: Array{Float64,2}, 
                term :: Int64, gradient ::  Array{Float64,2})
    #
    dims = size(position)
    nbclass = dims[2]
    obs_term = observations.datas[term]
    #
#    @assert length(position[:,1]) == length(obs_term)
    #
    class_term = round(Int64, observations.value_at_data[term])
    #
    den = 1.;
    dot_k = zeros(Float64, nbclass)
    # compute dot_k[k] = dot(position[:,k], obs_term)
    BLAS.gemm!('T', 'N', 1., position, obs_term, 0., dot_k)
    dot_k = exp.(dot_k)
    den = den + sum(dot_k)
    dot_k = dot_k / den
    obs_dim =  length(obs_term)
    for k in 1:nbclass
        gradient[:,k] .= obs_term .* dot_k[k]
        if k == class_term
            gradient[:,k] .= gradient[:,k] .- obs_term
        end
    end
end


