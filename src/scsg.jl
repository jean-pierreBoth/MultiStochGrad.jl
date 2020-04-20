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
function get_batchsizeinfo(scsg_pb :: SCSG, batch_growing_factor :: Int64 , nbterms::Int64, iteration :: Int64)
    alfa_j = batch_growing_factor^iteration
    # max size of large batch is 100 or 0.1 * the number of terms
    max_large_batch_size = nbterms > 100 ? ceil(Int64, nbterms/10.) : nbterms
    # ensure max_mini_batch_size is at least 1.
    max_mini_batch_size = ceil(Int64, nbterms/100.)
    # B_j
    large_batch_size = min(ceil(Int64, scsg_pb.large_batch_size_init * nbterms * alfa_j * alfa_j), max_large_batch_size)
    # b_j  grow slowly 
    mini_batch_size = min(floor(Int64, scsg_pb.mini_batch_size_init * alfa_j), max_mini_batch_size)
    # m_j  computed to ensure mean number of mini batch < large_batch_size as mini_batch_size_init < large_batch_size_init is enfored
    nb_mini_batch_parameter = scsg_pb.m_zero * nbterms * alfa_j^1.5
    step_size = scsg_pb.eta_zero / alfa_j
    #
    BatchSizeInfo(iteration, large_batch_size, mini_batch_size, nb_mini_batch_parameter, step_size)
    end


# select sizeasked terms selected without replacement
# returns the selected terms in a Vector{Int64}
function samplewithoutreplacement(size_asked::Int64, terms:: UnitRange{Int64})
    size_in = length(terms)
    out_terms = Vector{Int64}()
    for t in terms 
        if rand() * (size_in - t) < size_asked - length(out_terms)
            push!(out_terms, t)
        end
    end
    @assert size_asked == length(out_terms)
    out_terms 
end


"""
# function estimate_batch_growing_factor(batchinfo:: BatchSizeInfo, nbiterations::Int64, nbterms::Int64)

computes batch growing factor used to control large and mini batch sizes.  
It cannot be too large, so it must be adjusted according to nbterms value, and nb_max_iterations
The growing factor alfa should be such that :
       B_0 * alfa^(2nbiterations) < nbterms

"""
function getgrowingfactor(scsg_pb::SCSG, nbiterations::Int64, nbterms::Int64)
    if scsg_pb.m_zero > nbterms
        error("m_zero > nbterms in function to minimize, exiting")
    end
    log_alfa = -log(scsg_pb.large_batch_size_init) / (2. * nbiterations)
    batch_growing_factor = exp(log_alfa)
    if batch_growing_factor <= 1.
        @warn("batch growing factor shoud be greater than 1. , possibly you can reduce number of iterations ")
    end
    @debug "batch growing factor : " batch_growing_factor
    batch_growing_factor
end


"""
# function minimize(scsg_pb::SCSG, evaluation::Evaluator, initial_position::Vector{Float64})

## Args
- scsg_pb : the structure describing main parameters of the batch strategy
- scsg_pb : constains the structure containing observations and evaluations function
- initial_position : initial position of the iterations
"""

function minimize(scsg_pb::SCSG, evaluation::Evaluator, initial_position::Vector{Float64})
end