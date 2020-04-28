# logistic regression use case

using LinearAlgebra
using BLAS

include("../evaluator.jl")



# We have to implement 
# (observations: Observations, position: Array{Float64}, term : Int64) -> Float64
"""
# function term_value(observations:: Observations, position:: Array{Float64}, term :: Int64)

The dimension of our position vector is (nbclass - 1 ,  1 + length of an observation
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
    @debug "exiting logistic_term_gradient"
end




"""
# function LogisticRegression

Logistic Regression on nbclass classes.
Classes are supposed numbered from 0 to nbclass-1. (C/Rust indexing)


## Fields

- datas

    1. vectors in data have been augmented by one (with valuse of 1.) for interception term so that the length of
        vectors in datas is 1 + length of data in Observations

    2. The coefficients of the first class have been assumed to be 0 to take into account
        for the identifiability constraint (Cf  Machine Learning Murphy par 9.2.2.1-2)
        Our unknow is a 2 dimensional array  (length(Observations.datas[1]) + 1, nbclass -1)
        so that column 1 corresponds to class 1. (so we fall back to Julia indexing...)


- term_value

- term_gradient

"""

mutable struct LogisticRegression{F <: Function, G <: Function}
    nbclass :: Int64
    # length of observation + 1. Values 1. in first slot  of arrays.
    datas :: Vector{Tuple{Vector{Float64}, Float64}}
    #
    term_value :: F
    #
    term_gradient ::G
    function LogisticRegression{F,G}(nbclass :: Int64, observations::Observations, logistic_term_value::F, logistic_term_gradient::G) where {F,G}
        nbobs = length(observations.datas)
        datas = Vector{Tuple{Vector{Float64}, Float64}}(undef, nbobs)
        # add interception term
        obs_dim = length(observations.datas[1])
        # we search coefficients as 2 dimensional arrays , one column by class
        # and each column corresponds to observations type
        for i in 1:nbobs
            obs = zeros(Float64, obs_dim)
            obs[1:end] = observations.datas[i][1:end]
            # CAVEAT we shout check that class numbering goes from 0 to nbclass-1
            datas[i] = (obs, observations.value_at_data[i])
        end
        new(nbclass, datas, logistic_term_value, logistic_term_gradient)
    end
end

