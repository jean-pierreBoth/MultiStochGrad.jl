# evaluation of objective function


using Distributed

# we export here, so every submodule including this file gets visibility
export Observations,
    TermFunction,
    compute_value,
    TermGradient,
    compute_gradient!,
    Evaluator,
    get_nbterms



    
"""

# Observations


## Fields

- datas : list of data vector
- value_at_data : value for each observations

"""

mutable struct Observations 
    # 
    datas :: Vector{Vector{Float64}}
    value_at_data :: Vector{Float64}
end


"""

# TermFunction

A structure grouping observations and an evaluation function

The evaluation function must have signature
    (observations: Observations, position: Array{Float64, N}, term : Int64) -> Float64

## Args:

-  position : is position we want the function value at. So it is a vector
    of the same dimension as datas vector in observations.

- term : the rank of term in sum representing objective function

## Fields

"""


mutable struct TermFunction
    eval :: Function
    observations :: Observations
    dim :: Dims
    function TermFunction(evalf :: Function, observations :: Observations, d::Dims)
        # check signature
        z = zeros(d)
        dist = evalf(observations, observations.datas[1], 1)
        new(evalf, observations, d)
    end
end



"""
#  function compute_valuetf :: TermFunction, position :: Vector{Float64})

This function compute value of function at a given position summing over all terms
    
"""
function compute_value(tf :: TermFunction, position :: Array{Float64,1})
    nbterm = length(tf.observations.datas)
    # pamp with batch size = 1000
    values = pmap(i->tf.eval(tf.observations, position, i), 1:nbterm; batch_size = 1000)
    value = sum(values)/nbterm
    value
end


function compute_value(tf :: TermFunction, position :: Array{Float64,1}, terms::Vector{Int64})
    nbterm = length(tf.observations)
    # pamp with batch size = 1000. check speed versus a mapreduce
    values = pmap(i->tf.eval(tf.observations, position, i), terms; batch_size = 1000)
    value = sum(values)/nbterm
    value
end




"""
# struct TermGradient

This structure is is dedicated to do all gradient computations

## Fields

- eval is a function of signature Fn(Observations, Array{Float64}, Int64, Array{Float64}) 
    taking as arguments : 
        . observations
        . position
        . a term rank
        . a vector for gradient to be returned in (avoid reallocations as we sum the result in loops)
            Array for gradient has the same dimension as array for position (...)
    
- observations : the observations of the problem
- dims : ccharacterize the dimensions on variable for which we do a minimization

"""

mutable struct TermGradient
    eval :: Function
    observations :: Observations
    dim :: Dims
    function TermGradient(evalg :: Function, observations :: Observations, d::Dims)
        # check signature
        z = zeros(d)
        evalg(observations, observations.datas[1], 1, z)
        # find an assertion
        new(evalg, observations, d)
    end
end

"""
the function that will go in generic SCSG iterations

"""
function compute_gradient!(termg::TermGradient, position::Array{Float64,1} , term:: Int64, gradient:: Array{Float64,1})
    @debug " in  compute_gradient!(termg::TermGradient, position : ...terms::Int64 " 
    termg.eval(termg.observations, position, term, gradient)
end



function compute_gradient!(termg::TermGradient, position :: Array{Float64,1}, terms::Vector{Int64}, gradient::Array{Float64,1})
    @debug " in  compute_gradient!(termg::TermGradient, position : ...terms::Vector{Int64} " 
    batchsize=1000
    nbterms = length(terms)
     # split in blocks
    nbblocks = nbterms % batchsize
    nbblocks = rem(nbblocks,batchsize) > 0  ? nbblocks+1 : nbblocks
    @debug " nbblocks  " nbblocks
        # CAVEAT , if gradient 2d array bug.
    gradblocks = zeros(length(gradient), nbblocks)
    # CAVEAT to be threaded
    for i in 1::nbblocks 
        first = (i-1) * batchsize +1
        last = min(i*batchsize, nbterms)
        # CAVEAT , if gradient 2d array bug.
        gradtmp = zeros(Float64, length(gradient))
        for j in first:last
            termg.eval(termg.observations, position, terms[j], gradtmp)
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
function compute_gradient!(evaluator::Evaluator, position :: Array{Float64,1}, term::Int64 , gradient :: Array{Float64,1})
    @debug " in  compute_gradient!(evaluator::Evaluator, position : ...terms::Int64 " 
    compute_gradient!(evaluator.compute_term_gradient, position, term, gradient)
end


function compute_gradient!(evaluator::Evaluator, position :: Array{Float64,1}, terms::Vector{Int64}, gradient :: Array{Float64,1})
    @debug " in  compute_gradient!(evaluator::Evaluator, position : ...terms::Vector{Int64} " 
#    compute_gradient!(evaluator.compute_term_gradient, position, terms, gradient)
    @debug " in  compute_gradient!(termg::TermGradient, position : ...terms::Vector{Int64} " 
    termg = evaluator.compute_term_gradient
    batchsize=1000
    nbterms = length(terms)
     # split in blocks
    nbblocks = floor(Int64, nbterms / batchsize)
    nbblocks = nbterms % batchsize > 0  ? nbblocks+1 : nbblocks
    @debug " nbblocks  " nbblocks
        # CAVEAT , if gradient 2d array bug.
    gradblocks = zeros(length(gradient), nbblocks)
    # CAVEAT to be threaded
    for i in 1:nbblocks 
        first = (i-1) * batchsize +1
        last = min(i*batchsize, nbterms)
        # CAVEAT , if gradient 2d array bug.
        gradtmp = zeros(Float64, length(gradient))
        for j in first:last
            termg.eval(termg.observations, position, terms[j], gradtmp)
            @debug " gradtmp" gradtmp
            gradblocks[: ,i] .= gradblocks[:,i] + gradtmp
        end
    end
    # recall that in julia is column oriented so summing along rows is sum(,dims=2)
    copy!(gradient, sum(gradblocks,dims = 2)[:,1]/nbterms)
    @debug "gradient sum blocks" gradient
end


function compute_value(evaluator::Evaluator, position :: Array{Float64,1})
    compute_value(evaluator.compute_term_value, position)
end


function get_nbterms(evaluator::Evaluator)
    length(evaluator.compute_term_gradient.observations.datas)
end