# svrg implementation

export SVRG

"""
# struct SVRG

The structure describing iterations for stochastic variance reduced gradient.

## Method

We have the following sequence:    
   - a batch_gradient as the full gradient at current position  
   - storing gradient and position before the mini batch sequence  
   - then for `nb_mini_batch` :  
       - uniform sampling of **one** term of the summation  
       - computation of the gradient of the term at current position and the gradient
                        at position before mini batch
       - computation of direction of propagation as the batch gradient + gradient of term at current 
                        position -  gradient of term at position before mini batch sequence
       - update of position with adequate step size
    
   The step size used in the algorithm is constant and according to the ref paper it should be of the order of
   L/4 where L is the lipschitz constant of the function to minimize

## FIELDS

- nb\\_minibatch : The number of minibatch to run after one full batch run
- stepsize: the step to use in 
"""
mutable struct SVRG 
    # number of minibatch of one term.
    nb_minibatch :: Int64
    # step_size
    stepsize :: Float64
end





"""
    minimize(svrgpb::SVRG, evaluation::Evaluator{F,G}, max_iter::Int64, initialposition::Array{Float64,N})

Generic function used in minimisations for all algorithms.  
    
The function has 3 type parameters F, G and N: 

- F and G corresponds to the types of function used in instantiating the structure TermFunction{F} and TermmGradient{G}
(see in test and examples directories the cases TestLinearRegression or TestLogisticRegression). 

- N is the dimension of array for searched parameters/gradients which depend upon the problem and the way we modelize it.
(See Logistic regression where we used N=2)


# Args
- svrgpb : the structure describing main parameters of the batch strategy
- Evaluator{F,G}
- max_iter : maximum number of iterations
- initialposition : initial position at iteration beginning

"""
function minimize(svrgpb::SVRG, evaluation::Evaluator{F,G}, max_iterations, initial_position::Array{Float64,N}) where {F,G,N}
    direction = zeros(Float64, size(initial_position))
    full_batch_gradient = zeros(Float64, size(initial_position))
    mini_batch_gradient_current = zeros(Float64, size(initial_position))
    mini_batch_gradient_origin = zeros(Float64, size(initial_position))
    nbterms = get_nbterms(evaluation)
    #
    mt = MersenneTwister(117);
    position = Array{Float64,N}(initial_position)
    iteration = 0
    initial_value = 0.
    more = true
    while more 
        iteration += 1
        # get iteration parameters
        # compute gradient on large batch index set and store initial position
        compute_gradient!(evaluation, position ,full_batch_gradient)
        # get number of mini batch
        nb_mini_batch = svrgpb.nb_minibatch
        position_before_mini_batch = Array{Float64,N}(position)
        # loop on small batch iterations
        for i in 1:nb_mini_batch
            # sample mini batch terms
            term = 1 + floor(Int64, nbterms * rand(mt))
            compute_gradient!(evaluation, position , term, mini_batch_gradient_current)
            compute_gradient!(evaluation, position_before_mini_batch , term, mini_batch_gradient_origin)
            direction = mini_batch_gradient_current - mini_batch_gradient_origin + full_batch_gradient;
            position = position - svrgpb.stepsize * direction;
        end
        value = compute_value(evaluation, position)
        if iteration == 1
            initial_value = value
        end
        if iteration >= max_iterations 
            @info "Reached maximal number of iterations required, initial_value final value:" initial_value value
            return position, value
        end
    end

end