# logistic regression use case

include("../evaluator.jl")

"""
    data dimension have been augmented by one for interception term.
    The coefficients of the last class have been assumed to be 0 to take into account
    for the identifiability constraint (Cf  Machine Learning Murphy par 9.2.2.1-2)
"""

mutable struct LogisticRegression
    nbclass :: Int64
    # length of observation + 1. Values 1. in first slot  of arrays.
    observations :: Vector{Tuple{Vector{Float64}, Float64}}
end



# We have to implement 
# (observations: Observations, position: Vector{Float64}, term : Int64) -> Float64

function term_value(observations:: Observations, position:: Vector{Float64}, term :: Int64)
    term_data = observations.
end


function term_gradient(observations:: Observations, position:: Vector{Float64}, 
        term :: Int64, gradient ::  Vector{Float64})
end