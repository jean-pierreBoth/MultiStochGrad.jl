# evaluation of objective function




"""

# Observations

datas list of data vector
a vector of value at each data

"""

mutable struct Observations 
    # 
    datas :: Vector{Vector{Float64}}
    value_at_data :: Vector{Float64}
end


"""

# TermFunction

A structure grouping observations and an evaluation function

## Fields

"""
mutable struct TermFunction{F<:Function}
    eval :: F
    observations :: Observations
    function TermFunction(eval : F, observations : Observations)
        # check signature
        dist = eval(observations, observations.datas[1], 1)
        @assert abs(dist - observations.value_at_data[1]) < 1.E-10 "incoherent function"
        new(eval, observations)
    end
end


"""
# struct TermGradient

This structure is is dedicated to do all gradient computations

## Fiels

- eval is a function of signature Fn(Observations, Vector{Float64}, Int64, Vector{Float64}) 
    taking as arguments : 
        . observations
        . position
        . a term rank
        . a vector for gradient to be returned in (avoid reallocations as we sum the result in loops)
    
- observations : the observations of the problem

"""

mutable struct TermGradient{F<:Function}
    eval :: F
    observations :: Observations
    function TermGradient(eval : F, observations : Observations)
        # check signature
        direction = rand(length(observations.datas[1]))
        dist = eval(observations, observations.datas[1], 1, direction)
        # find an assertion
        new(eval, observations)
    end
end

"""
the function that will go in generic SCSG iterations

"""
function compute_gradient(termg::TermGradient, position, term)
    termg.eval(termg.observations, position, term)
end


#####################################################################

"""
#  Evaluator

   This structure contains all that is necessary to compute function value at any term
   and gradients total or any partial sum of terms.
   It can be passed as argument in any algorithm for our stochastic gradients.
    
## Fields

- compute_term_value : to do computations of function value
- compute_term_gradient: to do computations of gradients value


"""

mutable struct Evaluator
    #  A vector of observation , associated value
    compute_term_value :: TermFunction
    compute_term_gradient :: TermGradient
    function Evaluator(compute_term_value :: TermFunction, compute_term_gradient :: TermGradient)
        new(compute_term_value, compute_term_gradient)
    end
end





"""
the function that will go in generic SCSG iterations

"""
function compute_gradient(evaluator::Evaluator, position, term)
    gradient = evaluator.compute_term_gradient.eval(evaluator.compute_term_gradient.observations, position, term)
    return gradient
end

