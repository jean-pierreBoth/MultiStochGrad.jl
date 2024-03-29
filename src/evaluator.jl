# evaluation of objective function






"""
# TermFunction{F}  with F<: Function

This structure stores a function that compute a term of serie representing the objective function given 
the position and the term of the serie to evaluate.  

This **eval** function can be a closure caching observations data (see regressions examples)

## Fields

- eval is a function of signature   **Fn(Array{Float64, N}, Int64)::Float64** 
    taking as arguments (in this order):
    - position    
    - a term rank    

    This function returns the value of the component of rank term in the sum of objective function
        at position position given in args  

- nbterms in the sum

- dims : characterize the dimensions on variable for which we do a minimization
    It is not the same as observations which is always a one dimensional array.  

    For example on a logistic regression with nbclass we used (Cf applis) dims = (d , nbclass-1)
    with *d* the length(observations) and nbclass -1 beccause of solvability constraints.  

    It is also the dimension of gradients we compute.  

Note:
    When instantiating a TermFunction you just write:  
    *tf = TermFunction{typeof(yourfunction)}(yourfunction, nbterms, dims)*

"""
mutable struct TermFunction{F <: Function}
    eval :: F
    nbterms :: Int64
    dim :: Dims
    function TermFunction{F}(evalf :: F, nbterms:: Int64, d::Dims) where{F}
        # check signature
        z = zeros(d)
        evalf(z, 1)
        # we can initialize object
        new(evalf, nbterms, d)
    end
end



"""
    compute_value(tf :: TermFunction{F}, position :: Vector{Float64})

This function computes value of function at a given position summing over all terms
    
"""
function compute_value(tf :: TermFunction{F}, position :: Array{Float64,N}) where {F,N}
    nbterm = tf.nbterms
    value = compute_value(tf, position, Vector{Int64}((1:nbterm)))
    value
end




"""
    compute_value(tf :: TermFunction{F}, position :: Array{Float64,N}, terms::Vector{Int64})

This function computes value of function at a given position summing over all terms passed as arg
    
"""
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
        blockvalue[i] = mapreduce(i->tf.eval(position, i), + , terms[first:last])
    end
    # pamp with batch size = 1000. check speed versus a mapreduce
    value = sum(blockvalue)/nbterms
    value
end




"""
# struct TermGradient{G}  with G <: Function

This structure is the building block for all gradient computations.  
It stores a **eval** function that computes the gradient of term of serie representing the objective function given 
the position and the term of the serie to evaluate.  

This function **eval** can be a closure caching observations data (see regressions examples)

## Fields

- eval is a function of signature **Fn(Array{Float64,N}, Int64, Array{Float64,N})**   
    taking as arguments (in this order):   
    - position  
    - a term rank  
    - a vector for gradient to be returned in. This avoid reallocations as we loop on this function summing
            the resulting Array on required terms.  
            Array for gradient has the same dimension as array for position (...)
    
- nbterms : the number of terms in the sum defining the objective function
- dims : characterize the dimensions on variable for which we do a minimization

"""
struct TermGradient{G<:Function}
    eval :: G
    nbterms :: Int64
    dim :: Dims
    function TermGradient{G}(evalg :: G, nbterms :: Int64, d::Dims) where {G}
        # we check that signature of function defined by the user is correct
        z = zeros(d)
        grad = zeros(d)
        evalg(z, 1, grad)
        # we can allocate our object
        new(evalg, nbterms, d)
    end
end


"""
    compute_gradient!(termg::TermGradient{G}, position::Array{Float64,N} , term:: Int64, gradient:: Array{Float64,N}) where {G,N}


the function  dispatches to TermGradient  the actual gradient computation for a term.

"""
function compute_gradient!(termg::TermGradient{G}, position::Array{Float64,N} , term:: Int64, gradient:: Array{Float64,N}) where {G,N}
    termg.eval(position, term, gradient)
end



"""
    compute_gradient!(termg::TermGradient{G}, position :: Array{Float64,N}, terms::Vector{Int64}, gradient::Array{Float64,N}) where {G,N}

This function computes a gradient Array at a given position summing over all terms passed as arg.

NOTA: This function computes the mean of gradients returned by the termg function returned on each term.
So that the gradient computed on a batch is an estimator of the gradient computed on the whole objective function.

It is multithreaded and computes gradient by blocks of 1500 terms.

"""
function compute_gradient!(termg::TermGradient{G}, position :: Array{Float64,N}, terms::Vector{Int64}, gradient::Array{Float64,N}) where {G,N}
        # 
        batchsize=1500
        nbterms = length(terms)
         # split in blocks
        nbblocks = floor(Int64, nbterms / batchsize)
        nbblocks = nbterms % batchsize > 0  ? nbblocks+1 : nbblocks
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
                termg.eval(position, terms[j], gradtmp)
                gradblocks[i] = gradblocks[i] + gradtmp
            end
        end
        # recall that in julia is column oriented so summing along rows is sum(,dims=2)
        copy!(gradient, sum(gradblocks)/nbterms)
 #       @debug "gradient sum blocks" gradient
end




"""
    compute_gradient!(termg::TermGradient{G}, position :: Array{Float64,N}, gradient::Array{Float64,N}) where {G,N}

This function computes a gradient Array at a given position summing over all terms
"""
function compute_gradient!(termg::TermGradient{G}, position :: Array{Float64,N}, gradient::Array{Float64,N}) where {G,N}
    nbterms = termg.nbterms
    compute_gradient!(termg, position, collect(1:nbterms), gradient)
end

#####################################################################

"""
#  Evaluator{F,G}

   This structure contains all that is necessary to compute function value at any term
   and gradient total or any partial sum of terms.
   It can be passed as argument in any algorithm for our stochastic gradients.
    
## Fields

- compute\\_term\\_value : a TermFunction{F}
- compute\\_term\\_gradient: a TermGradient{G}

It is associated to various functions dispatching computations to it.

"""
struct Evaluator{F,G}
    #  A vector of observation , associated value
    compute_term_value :: TermFunction{F}
    compute_term_gradient :: TermGradient{G}
    function Evaluator{F,G}(compute_term_value :: TermFunction{F}, compute_term_gradient :: TermGradient{G}) where {F,G}
        new(compute_term_value, compute_term_gradient)
    end
end





"""
    compute_gradient!(evaluator::Evaluator{F,G}, position :: Array{Float64,N}, term::Int64 , gradient :: Array{Float64,N}) where {F,G,N}


the function computes gradient given an evaluator, a position, and a term

"""
function compute_gradient!(evaluator::Evaluator{F,G}, position :: Array{Float64,N}, term::Int64 , gradient :: Array{Float64,N}) where {F,G,N}
    compute_gradient!(evaluator.compute_term_gradient, position, term, gradient)
end



"""
    compute_gradient!(evaluator::Evaluator{F,G}, position :: Array{Float64,N}, terms::Vector{Int64}, gradient :: Array{Float64,N}) where {F,G,N}

the function computes a gradient given an evaluator, a position, and a vector of rank term

"""
function compute_gradient!(evaluator::Evaluator{F,G}, position :: Array{Float64,N}, terms::Vector{Int64}, gradient :: Array{Float64,N}) where {F,G,N}
    compute_gradient!(evaluator.compute_term_gradient, position, terms, gradient)
end



"""
    compute_gradient!(evaluator::Evaluator{F,G}, position :: Array{Float64,N}, gradient::Array{Float64,N}) where {F, G,N}

This function computes a gradient Array at a given position summing over all terms
"""
function compute_gradient!(evaluator::Evaluator{F,G} , position :: Array{Float64,N}, gradient::Array{Float64,N}) where {F, G,N}
    compute_gradient!(evaluator.compute_term_gradient, position, gradient)
end


"""
    compute_value(evaluator::Evaluator{F,G}, position :: Array{Float64,N}) where {F,G,N}

this function computes a value given an evaluator, and a position using all terms
"""
function compute_value(evaluator::Evaluator{F,G}, position :: Array{Float64,N}) where {F,G,N}
    compute_value(evaluator.compute_term_value, position)
end


"""
    compute_value(evaluator::Evaluator{F,G}, position :: Array{Float64,N}, terms::Vector{Int64})  where {F,G,N}

this function computes a value given an evaluator, and a position and a list of terms

"""
function compute_value(evaluator::Evaluator{F,G}, position :: Array{Float64,N}, terms::Vector{Int64})  where {F,G,N}
    compute_value(evaluator.compute_term_value, position, terms)
end





"""
    function get_nbterms(evaluator::Evaluator{F,G}) where {F,G}

retrieves number of terms in the sum defining objective function
"""
function get_nbterms(evaluator::Evaluator{F,G}) where {F,G}
    evaluator.compute_term_gradient.nbterms
end