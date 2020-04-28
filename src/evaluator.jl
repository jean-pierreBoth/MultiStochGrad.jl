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

- datas : list of data vector one for each observations
- value_at_data : value for each observations

For regressions problems for example length(datas) is number of observations.
    and length(datas[1]) is 1+dimension of observations data beccause of interception terms.

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

- eval is a function of signature Fn(Observations, Array{Float64}, Int64)::Float64
    taking as arguments (in this order): 
        . observations
        . position
        . a term rank
    it returns the value of the component of rank term in the sum of objective function
        at position position given in args

- dims : characterize the dimensions on variable for which we do a minimization
        it is not the same as observations which is always a one dimensional array.
        It is also the dimension of gradients we compute.
        For example on a logistic regression with nbclass dims = (d , nbclass-1)
        with d  is lenght(observations)
        nbclass -1 beccause of solvability constraints.

"""


mutable struct TermFunction{F <: Function}
    eval :: F
    observations :: Observations
    dim :: Dims
    function TermFunction{F}(evalf :: F, observations :: Observations, d::Dims) where{F}
        # check signature
        z = zeros(d)
        dist = evalf(observations, z, 1)
        new(evalf, observations, d)
    end
end



"""
#  function compute_valuetf :: TermFunction, position :: Vector{Float64})

This function compute value of function at a given position summing over all terms
    
"""
function compute_value(tf :: TermFunction{F}, position :: Array{Float64,N}) where {F,N}
    nbterm = length(tf.observations.datas)
    value = compute_value(tf, position, Vector{Int64}((1:nbterm)))
    value
end


function compute_value(tf :: TermFunction{F}, position :: Array{Float64,N}, terms::Vector{Int64}) where {F,N}
    nbterms = length(terms)
    # split in blocks
    batchsize = 2000
    nbblocks = floor(Int64, nbterms / batchsize)
    nbblocks = nbterms % batchsize > 0  ? nbblocks+1 : nbblocks
    blockvalue = zeros(Float64, nbblocks)
    # 
    Threads.@threads for i in 1:nbblocks 
        first = (i-1) * batchsize +1
        last = min(i*batchsize, nbterms)
        blockvalue[i] = mapreduce(i->tf.eval(tf.observations, position, i), + , terms[first:last])
    end
    # pamp with batch size = 1000. check speed versus a mapreduce
    value = sum(blockvalue)/nbterms
    value
end




"""
# struct TermGradient

This structure is is dedicated to do all gradient computations

## Fields

- eval is a function of signature Fn(Observations, Array{Float64}, Int64, Array{Float64}) 
    taking as arguments (in this order): 
        . observations
        . position
        . a term rank
        . a vector for gradient to be returned in (avoid reallocations as we sum the result in loops)
            Array for gradient has the same dimension as array for position (...)
    
- observations : the observations of the problem
- dims : characterize the dimensions on variable for which we do a minimization

"""

mutable struct TermGradient{G<:Function}
    eval :: G
    observations :: Observations
    dim :: Dims
    function TermGradient{G}(evalg :: G, observations :: Observations, d::Dims) where {G}
        # check signature
        z = zeros(d)
        grad = zeros(d)
        evalg(observations, z, 1, grad)
        # find an assertion
        new(evalg, observations, d)
    end
end

"""
the function that will go in generic SCSG iterations

"""
function compute_gradient!(termg::TermGradient{G}, position::Array{Float64,N} , term:: Int64, gradient:: Array{Float64,N}) where {G,N}
    @debug " in  compute_gradient!(termg::TermGradient, position : ...terms::Int64 " 
    termg.eval(termg.observations, position, term, gradient)
end



function compute_gradient!(termg::TermGradient{G}, position :: Array{Float64,N}, terms::Vector{Int64}, gradient::Array{Float64,N}) where {G,N}
        @debug " in  compute_gradient!(termg::TermGradient, position : ...terms::Vector{Int64} "
        # 
        batchsize=1500
        nbterms = length(terms)
         # split in blocks
        nbblocks = floor(Int64, nbterms / batchsize)
        nbblocks = nbterms % batchsize > 0  ? nbblocks+1 : nbblocks
#        @debug " nbblocks  " nbblocks
        # we must allocate an array of nbblocks arrays of dimensions size(gradient)
        dimg = ndims(gradient)
        gradblocks = Vector{Array{Float64,dimg}}(undef, nbblocks)
        # CAVEAT to be threaded
        Threads.@threads for i in 1:nbblocks 
            first = (i-1) * batchsize +1
            last = min(i*batchsize, nbterms)
            # must respect dimensions of gradient
            gradtmp = zeros(Float64, size(gradient))
            gradblocks[i] = zeros(Float64, size(gradient))
            for j in first:last
                termg.eval(termg.observations, position, terms[j], gradtmp)
                gradblocks[i] = gradblocks[i] + gradtmp
            end
        end
        # recall that in julia is column oriented so summing along rows is sum(,dims=2)
        copy!(gradient, sum(gradblocks)/nbterms)
 #       @debug "gradient sum blocks" gradient
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

mutable struct Evaluator{F,G}
    #  A vector of observation , associated value
    compute_term_value :: TermFunction{F}
    compute_term_gradient :: TermGradient{G}
    function Evaluator{F,G}(compute_term_value :: TermFunction{F}, compute_term_gradient :: TermGradient{G}) where {F,G}
        new(compute_term_value, compute_term_gradient)
    end
end





"""
the function that will go in generic SCSG iterations

"""
function compute_gradient!(evaluator::Evaluator{F,G}, position :: Array{Float64,N}, term::Int64 , gradient :: Array{Float64,N}) where {F,G,N}
    @debug " in  compute_gradient!(evaluator::Evaluator, position : ...terms::Int64 " 
    compute_gradient!(evaluator.compute_term_gradient, position, term, gradient)
end


function compute_gradient!(evaluator::Evaluator{F,G}, position :: Array{Float64,N}, terms::Vector{Int64}, gradient :: Array{Float64,N}) where {F,G,N}
    @debug " in  compute_gradient!(evaluator::Evaluator, position : ...terms::Vector{Int64} " 
    compute_gradient!(evaluator.compute_term_gradient, position, terms, gradient)
end


function compute_value(evaluator::Evaluator{F,G}, position :: Array{Float64,N}) where {F,G,N}
    compute_value(evaluator.compute_term_value, position)
end


function compute_value(evaluator::Evaluator{F,G}, position :: Array{Float64,N}, terms::Vector{Int64})  where {F,G,N}
    compute_value(evaluator.compute_term_value, position, terms)
end


function get_nbterms(evaluator::Evaluator{F,G}) where {F,G}
    length(evaluator.compute_term_gradient.observations.datas)
end