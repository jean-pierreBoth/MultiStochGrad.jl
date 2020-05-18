# SCSG implementation

using Random

export SCSG,
        minimize,
        BatchSizeInfo,
        get_batchsizeinfo

include("evaluator.jl")


"""
# SCSG

This is the structure describing parameters used in Stochastic Controlled Stochastic Gradient

## Fields

- eta_zero : initial step size
- m_zero : fraction of terms to consider in initialisation of the number of mini batch to run for each large batch (mⱼ)
- mini\\_batch\\_size\\_init : governs size of mini batches (1 is a good initialization)
- large\\_batch\\_size\\_init : fraction of terms to consider in initialisation of large batch size

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

stores all parameters describing one batch characteristics

"""
mutable struct BatchSizeInfo
    iteration :: Int64
    large_batchsize:: Int64
    mini_batchsize:: Int64
    # parameter used in determining number of mini batch
    nbminibatchparam :: Float64   
    # step used in position update
    stepsize :: Float64
end


"""
returns a struct BatchSizeInfo for a given iteration
"""
function get_batchsizeinfo(scsg_pb :: SCSG, batch_growing_factor :: Float64 , nbterms::Int64, iteration :: Int64)
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
    step_size = scsg_pb.eta_zero / sqrt(alfa_j)
    #
    BatchSizeInfo(iteration, large_batch_size, mini_batch_size, nb_mini_batch_parameter, step_size)
end


# select sizeasked terms selected without replacement
# returns the selected terms in a Vector{Int64}
function samplewithoutreplacement(size_asked::Int64, terms:: Vector{Int64})
    size_in = length(terms)
    out_terms = Vector{Int64}()
    for i in 1:length(terms) 
        if rand() * (size_in - i) < size_asked - length(out_terms)
            push!(out_terms, terms[i])
        end
    end
    @assert size_asked == length(out_terms)
    out_terms 
end


# used to sample mini batches, Really faster.
# See:
#  1. Faster methods for Random Sampling J.S Vitter Comm ACM 1984
#  2. Kim-Hung Li Reservoir Sampling Algorithms : Comm ACM Vol 20, 4 December 1994
#  3. https://en.wikipedia.org/wiki/Reservoir_sampling

function samplewithoutreplacement_reservoir(size_asked::Int64, in_terms::Vector{Int64})
    out_terms = in_terms[1:size_asked]
    w  = exp(log(rand()/(size_asked)))
    s = size_asked
    while s <= length(in_terms)
        s = s + floor(Int64, log(rand())/log(1-w)) + 1
        if s <= length(in_terms)
            k = 1 + floor(Int64, size_asked * rand())
            out_terms[k] = in_terms[s]
            w = w * exp(log(rand())/size_asked)
        end
    end
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
# function get_nbminibatch(batch_info::BatchSizeInfo)
"""
function get_nbminibatch(batch_info::BatchSizeInfo)
    m_j =  batch_info.nbminibatchparam
    b_j = batch_info.mini_batchsize
    # we return mean of geometric. Sampling too much instable due to large variance of geometric distribution.
    # ensure that it is at least 1 with ceil
    n_j = ceil(Int64, m_j/b_j)
    n_j = min(n_j, batch_info.large_batchsize) 
    @debug " nb mini batch = " n_j
    n_j
end




"""
# function minimize(scsgpb::SCSG, evaluation::Evaluator{F,G}, max_iter::Int64, initialposition::Array{Float64,N})

Generic function used in minimisations for all algorithms.  
    
The function has 3 types parameters F, G and N. 

- F and G corresponds to the types of function used in instantiating the structure TermFunction{F} and TermmGradient{G}
(see in test and examples directories the cases: TestLinearRegression or TestLogisticRegression). 

- N is the dimension of array for searched parameters (position and gradients) which depends upon the problem and the way we modelize it.
(See Logistic regression where we used N=2)


## Args
- scsgpb : the structure describing main parameters of the batch strategy
- Evaluator{F,G}
- max_iter : maximum number of iterations
- initialposition : initial position of the iterations

"""
function minimize(scsg_pb::SCSG, evaluation::Evaluator{F,G}, max_iterations, initial_position::Array{Float64,N}) where {F,G,N}
    @debug "scsg_pb" scsg_pb
    #
    direction = zeros(Float64, size(initial_position))
    large_batch_gradient = zeros(Float64, size(initial_position))
    mini_batch_gradient_current = zeros(Float64, size(initial_position))
    mini_batch_gradient_origin = zeros(Float64, size(initial_position))
    nbterms = get_nbterms(evaluation)
    all_index = collect(1:nbterms)
    batch_growing_factor = get_batchgrowingfactor(scsg_pb, max_iterations, nbterms)
    #
    position = Array{Float64}(initial_position)
    iteration = 0
    more = true
    while more 
        iteration += 1
        # get iteration parameters
        batch_info = get_batchsizeinfo(scsg_pb, batch_growing_factor, nbterms, iteration)
        @debug "batch_info" batch_info
        # batch sampling
        batch_indexes = samplewithoutreplacement(batch_info.large_batchsize, all_index)
        # compute gradient on large batch index set and store initial position
        compute_gradient!(evaluation, position , batch_indexes, large_batch_gradient)

        # get mean number  binomial law for number Nj of small batch iterations
        nb_mini_batch = get_nbminibatch(batch_info)
        @info "batch info " batch_info.large_batchsize batch_info.mini_batchsize nb_mini_batch batch_info.stepsize
        position_before_mini_batch = Array{Float64}(position)
        # loop on small batch iterations
        for i in 1:nb_mini_batch
            # sample mini batch terms
            batch_indexes = samplewithoutreplacement_reservoir(batch_info.mini_batchsize, all_index)
            compute_gradient!(evaluation, position , batch_indexes, mini_batch_gradient_current)
            compute_gradient!(evaluation, position_before_mini_batch , batch_indexes, mini_batch_gradient_origin)
            direction = mini_batch_gradient_current - mini_batch_gradient_origin + large_batch_gradient;
            position = position - batch_info.stepsize * direction;
        end
        @debug "norm L2 direction" norm(direction)
        value = compute_value(evaluation, position)
        @info "iteration  value " iteration value
        if iteration >= max_iterations 
            @info("Reached maximal number of iterations required , stopping optimization");
            return position, value
        end
    end

end