# SCSG implementation

using Random


"""

# SCSG

This is the structure for Stochastic Controlled Stochastic Gradient

"""

mutable struct SCSG
    # initial step size
    eta_zero :: Float64
    # fraction of nbterms to consider in initialization of mⱼ governing evolution of nb small iterations
    m_zero :: Float64
    # m_0 in the paper
    mini_batch_size_init :: Int64
    # related to B_0 in Paper. Fraction of terms to consider in initialisation of B_0
    large_batch_size_init :: Float64
end


# returns growingfactor for computing size of large batch ...
# 
function getgrowingfactor(scsg::SCSG , nbmaxiter::Int64, nbterms)
end


"""
# struct BatchSizeInfo

stores all parameters describing batch characteristis


"""
mutable struct BatchSizeInfo
    iteration :: Int64
    largebatchsize:: Int64
    minibatchsize:: Int64
    # parameter used in determining number of mini batch
    nbminibatchparam :: Int64
    # step used in position update
    stepsize :: Float64
end


"""
returns a struct BatchSizeInfo for a given iteration
"""
function getbatchinfo(scsc :: SCSG, iteration :: Int64)
end

# select sizeasked termsselected without replacement from growingfactor
# returns the selected terms in a Vector{Int64}
function samplewithoutreplacement(sizeasked::Int64, terms:: Vector{Int64})
end


"""
`function estimate_batch_growing_factor(batchinfo:: BatchSizeInfo, nbiterations::Int64, nbterms::Int64)`

computes batch growing factor used to control large and mini batch sizes.  
It cannot be too large, so it must be adjusted according to nbterms value, and nb_max_iterations
We use the following rules for large batch sizes:  
    1.   B_0 max(10, nbterms/100)
    2.   B_0 * alfa^(2T) < nbterms

Then mini batch
"""
function estimate_batch_growing_factor(batchinfo:: BatchSizeInfo, nbiterations::Int64, nbterms::Int64)
end

function minimize(scsg_pb::SCSG, evaluation::Evaluator, initial_position::Vector{Float64})
end