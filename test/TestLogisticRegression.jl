# logistic regression on mnist data

using MultiStochGrad

using  LinearAlgebra, Test



using  LinearAlgebra, Test
using Random, Distributions
using Printf


include("../src/applis/LogisticRegression.jl")

include("../src/scsg.jl")
include("../src/mnist.jl")

const IMAGE_FNAME_STR  = "/home.1/jpboth/Data/MNIST/train-images-idx3-ubyte"
const LABEL_FNAME_STR  = "/home.1/jpboth/Data/MNIST/train-labels-idx1-ubyte"




function mnist_logistic_regression_scsg()
    # check path and load
    if !isfile(IMAGE_FNAME_STR) || !isfile(LABEL_FNAME_STR)
        @warn("mnist_logistic_regression : bad path to mnist data file")
        return false
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
    observations= Observations(datas,values)
    nbclass = 10
    # struct LogisticRegression takes care of interceptions terms and constraint on first class
    logreg = LogisticRegression(nbclass, observations)
    # define TermFunction , TermGradient and Evaluator, dims is one less than number of classes for identifiability constraints
    dims = Dims{2}((length(datas[1]) , nbclass-1))
    term_function =  TermFunction(logreg.term_value, observations, dims)
    term_gradient2 = TermGradient(logreg.term_gradient ,observations, dims)
    evaluator = Evaluator(term_function, term_gradient2)
    # define parameters for scsg
    scsg_pb = SCSG(0.5, 0.0015, 1 , 0.015)
    # solve.
    nb_iter = 50
    initial_position= fill(0.5, dims)
    @info "initial error " compute_value(evaluator, initial_position)
    position, value = minimize(scsg_pb, evaluator, nb_iter, initial_position)
    @printf(stdout, "value = %f, position = %f %f %f ", value , position)
end