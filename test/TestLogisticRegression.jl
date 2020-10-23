# logistic regression on mnist data

using MultiStochGrad

using  LinearAlgebra, Test


using  LinearAlgebra, Test
using Random, Distributions
using Printf
using Statistics

include("../src/applis/LogisticRegression.jl")
include("../src/mnist.jl")


const IMAGE_FNAME_STR  = "/home.1/jpboth/Data/MNIST/train-images-idx3-ubyte"
const LABEL_FNAME_STR  = "/home.1/jpboth/Data/MNIST/train-labels-idx1-ubyte"


function initMnitObservations()
    # check path and load
    if !isfile(IMAGE_FNAME_STR) || !isfile(LABEL_FNAME_STR)
        @warn("\n\n mnist_logistic_regression : bad path to mnist data file, cannot do tests, download and install files")
        return nothing
    end
    mnistdata = mnist.MnistData(IMAGE_FNAME_STR, LABEL_FNAME_STR)
    # transform data into logistic regression problem
    # labels are numbered from 0 to 9.
    datas = Vector{Vector{Float64}}()
    values = Vector{Float64}()
    images = mnistdata.images
    dims = size(images)
    nbimages = dims[3]
    nbpixels = dims[1] * dims[2]
    for i in 1:nbimages
        v = Vector{Float64}()
        push!(v, 1.)
        # reshape and convert to Float64
        v = cat(v, convert(Array{Float64}, reshape(images[:,:,i], nbpixels)), dims = 1)
        v = v / 256.
        push!(datas, v)
        push!(values, Float64(mnistdata.labels[i]))
    end
    observations = Observations(datas,values)
    Some(observations)
end


function mnist_logistic_regression_scsg()
    @printf(stdout, "\n\n\n begining mnist_logistic_regression_scsg ...")
    mnist = initMnitObservations()
    if isnothing(mnist)
        return false
    end
    #
    observations = something(mnist)
    nbclass = 10
    # struct LogisticRegression takes care of interceptions terms and constraint on first class
    # define TermFunction , TermGradient and Evaluator, dims is one less than number of classes for identifiability constraints
    dims = Dims{2}((length(observations.datas[1]) , nbclass-1))
    term_function =  TermFunction{typeof(logistic_term_value)}(logistic_term_value, observations, dims)
    term_gradient = TermGradient{typeof(logistic_term_gradient)}(logistic_term_gradient ,observations, dims)
    evaluator = Evaluator{typeof(logistic_term_value),typeof(logistic_term_gradient)}(term_function, term_gradient)
    # define parameters for scsg
    scsg_pb = SCSG(0.1, 0.006, 1 , 0.02)
    @debug "scsg_pb" scsg_pb
    # solve.
    nb_iter = 70
    initial_position= fill(0.5, dims)
    initial_error = compute_value(evaluator, initial_position)
    @info "initial error " initial_error
    @time position, value = minimize(scsg_pb, evaluator, nb_iter, initial_position)
    @printf(stdout, "value = %f", value)
    @printf(stdout, "\n\n mnist_logistic_regression_scsg done \n")
    value < 0.3 ? true : false
end



function mnist_logistic_regression_svrg()
    @printf(stdout, "\n\n\n begining mnist_logistic_regression_svrg ...")
    mnist = initMnitObservations()
    if isnothing(mnist)
        return false
    end
    #
    observations = something(mnist)
    nbclass = 10
    # struct LogisticRegression takes care of interceptions terms and constraint on first class
    # define TermFunction , TermGradient and Evaluator, dims is one less than number of classes for identifiability constraints
    dims = Dims{2}((length(observations.datas[1]) , nbclass-1))
    term_function =  TermFunction{typeof(logistic_term_value)}(logistic_term_value, observations, dims)
    term_gradient = TermGradient{typeof(logistic_term_gradient)}(logistic_term_gradient ,observations, dims)
    evaluator = Evaluator{typeof(logistic_term_value),typeof(logistic_term_gradient)}(term_function, term_gradient)
    # define parameters for svrg  1000 minibatch , step 0.02
    svrg_pb = SVRG(2000, 0.05)
    # solve.
    nb_iter = 60
    initial_position= fill(0.5, dims)
    initial_error = compute_value(evaluator, initial_position)
    @info "initial error " initial_error
    @time position, value = minimize(svrg_pb, evaluator, nb_iter, initial_position)
    @printf(stdout, "value = %f", value)
    @printf(stdout, "\n\n mnist_logistic_regression_svrg done \n")
    value < 0.3 ? true : false
end