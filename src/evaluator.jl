# evaluation of objective function


using Distributed

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
    function TermFunction(evalf : F, observations : Observations)
        # check signature
        dist = evalf(observations, observations.datas[1], 1)
        @assert abs(dist - observations.value_at_data[1]) < 1.E-10 "incoherent function"
        new(evalf, observations)
    end
end


"""
#  function compute_valuetf :: TermFunction, position :: Vector{Float64})

This function compute value of function at a given position summing over all terms
    
"""
function compute_value(tf :: TermFunction, position :: Vector{Float64})
    nbterm = length(tf.observations)
    # pamp with batch size = 1000
    values = pmap(i->tf.eval(tf.observations[i], position), 1:nbterm; batch_size = 1000)
    value = sum(values)/nbterm
    value
end


function compute_value(tf :: TermFunction, position :: Vector{Float64}, terms::Vector{Int64})
    nbterm = length(tf.observations)
    # pamp with batch size = 1000. check speed versus a mapreduce
    values = pmap(i->tf.eval(tf.observations[i], position), terms; batch_size = 1000)
    value = sum(values)/nbterm
    value
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
    function TermGradient(evalg : F, observations : Observations)
        # check signature
        direction = rand(length(observations.datas[1]))
        evalg(observations, observations.datas[1], 1, direction)
        # find an assertion
        new(evalg, observations)
    end
end

"""
the function that will go in generic SCSG iterations

"""
function compute_gradient!(termg::TermGradient, position, term, gradient)
    termg.eval(termg.observations, position, term, gradient)
end



function compute_gradient!(termg::TermGradient, position : Vector{Float64}, terms::Vector{Float64}, gradient)
    batchsize=1000
    nbterms = length(terms)
     # split in blocks
    nbblocks = nbterms % batchsize
    nbblocks = rem(nbblocks,batchsize) > 0 nbblocks+1 : nbblocks
    gradblocks = zeros(length(gradient), nbblocks)
    for i in 1::nbblocks 
        first = (i-1) * batchsize +1
        last = min(i*batchsize, nbterms)
        gradtmp = zeros(Float64, length(gradient))
        for j in first:last
            compute_gradient!(termg, position, terms[j], gradtmp)
            gradblocks[:,i]= gradblocks[:,i] + gradtmp
        end
    end
    # recall that in julia is column oriented so summing along rows is sum(,2)
    gradient = sum(gradblocks,2)/nbterms
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
function compute_gradient!(evaluator::Evaluator, position, term, gradient)
    evaluator.compute_term_gradient.eval(evaluator.compute_term_gradient.observations, position, term, gradient)
end


function compute_gradient!(evaluator::Evaluator, position : Vector{Float64}, terms::Vector{Float64}, gradient)
    evaluator.compute_term_gradient.compute_gradient!(evaluator.compute_term_gradient.observations, position, terms, gradient)
end