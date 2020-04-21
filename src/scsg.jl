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
    # b_0 in the paper
    mini_batch_size_init :: Int64
    # related to B_0 in Paper. Fraction of terms to consider in initialisation of B_0
    large_batch_size_init :: Float64
end



"""
# struct BatchSizeInfo

stores all parameters describing batch characteristics


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
function get_batchgrowingfactor(scsg_pb::SCSG, nbiterations::Int64, nbterms::Int64)
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

"""

function get_nbminibatch(batch_info::BatchSizeInfo)
    m_j =  batch_info.nb_mini_batch_parameter
    b_j = batch_info.mini_batch_size
    # we return mean of geometric. Sampling too much instable due to large variance of geometric distribution.
    # ensure that it is at least 1 with ceil
    n_j = ceil(Int64, b_j/m_j)
    n_j = min(n_j, batch_info.large_batch_size) 
    @debug " nb mini batch = " n_j
    n_j
end

"""
# function minimize(scsg_pb::SCSG, evaluation::Evaluator, max_iter :: Int64, initial_position::Vector{Float64})

## Args
- scsg_pb : the structure describing main parameters of the batch strategy
- max_iter : maximum number of iterations
- initial_position : initial position of the iterations

"""

function minimize(scsg_pb::SCSG, evaluation::Evaluator, max_iterations, initial_position::Vector{Float64})
    direction = zeros(Float64, length(position))
    large_batch_gradient = zeros(Float64, length(position))
    mini_batch_gradient_current = zeros(Float64, length(position))
    mini_batch_gradient_origin = zeros(Float64, length(position))
    nbterms = get_nbterms(evaluation)
    batch_growing_factor = get_batchgrowingfactor(scsg_pb, max_iterations, evaluation)
    #
    position = Vector{Float64}(initial_position)
    iteration = 0
    more = true
    while more 
        iteration += 1
        # get iteration parameters
        batch_info = get_batchsizeinfo(scsg_pb, batch_growing_factor, nbterms, iteration)
        # batch sampling
        batch_indexes = samplewithoutreplacement(batch_info.large_batch_size, 1:nbterms)
        # compute gradient on large batch index set and store initial position
        compute_gradient!(evaluation.compute_term_gradient, position , batch_indexes, large_batch_gradient)
        # sample binomial law for number Nj of small batch iterations
        nb_mini_batch = get_nbminibatch(scsg_pb)
        position_before_mini_batch = Vector{Float64}(position)
        # loop on small batch iterations
        for i in 1:nb_mini_batch
            # sample mini batch terms
            batch_indexes = samplewithoutreplacement(batch_info.minibatchsize, 1:nbterms)
            compute_gradient!(evaluation.compute_term_gradient, position , batch_indexes, mini_batch_gradient_current)
            compute_gradient!(evaluation.compute_term_gradient, position_before_mini_batch , batch_indexes, mini_batch_gradient_origin)
            direction = mini_batch_gradient_current - mini_batch_gradient_origin + large_batch_gradient;
            position = position - batch_info.stepsize * direction;
        end
        value = compute_value(evaluation, position)
        if iteration >= max_iterations 
            @info("Reached maximal number of iterations required , stopping optimization");
            return position, value
        end
    end

end