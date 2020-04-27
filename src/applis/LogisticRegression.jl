# logistic regression use case

using LinearAlgebra


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
    term_data = observations.datas[term]
    term_class = round(Int64, observations.value_at_data[term])
    #
    logarg = 1.
    for k in 1:nbclass
        logarg += exp(dot(term_data, position[:,k]))
    end
    other_term = 0.
    # This relies on our convention that class 0 bears the identifiability constraint.
    if term_class >= 1
        @assert size(term_data) == size(position[:,term_class])
        other_term  = dot(term_data, position[:,term_class])
    end
    log(logarg) - other_term
end


function logistic_term_gradient(observations:: Observations, position:: Array{Float64,2}, 
                term :: Int64, gradient ::  Array{Float64,2})
    #
    dims = size(position)
    nbclass = dims[2]
    obs_term = observations.datas[term]
    class_term = round(Int64, observations.value_at_data[term])
    #
    den = 1.;
    for k in 1:nbclass
        dot_k = dot(position[:,k], obs_term)
        den += exp(dot_k)
    end
    obs_dim =  length(obs_term)
    for k in 1:nbclass
        @assert length(position[:,k]) == length(obs_term)
        dot_k  = dot(obs_term, position[:,k])
        for j in 1:obs_dim
            g_term = obs_term[j] * exp(dot_k)/den;
            # keep term corresponding to term_class (class of term passed as arg)
            if class_term == k 
                g_term -= obs_term[j];
            end
            gradient[j, k] = g_term;
        end
    end
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

