module MultiStochGrad

using Printf

using Logging
using Base.CoreLogging
using LinearAlgebra


debug_log = stdout
logger = ConsoleLogger(stdout, CoreLogging.Info)
global_logger(logger)


include("scsg.jl")
include("svrg.jl")

include("applis/LinearRegression.jl")
include("applis/LogisticRegression.jl")

# export list

export Observations,
    TermFunction,
    compute_value,
    TermGradient,
    compute_gradient!,
    Evaluator,
    get_nbterms,
    # from scsg
    SCSG,
    minimize,
    BatchSizeInfo,
    get_batchsizeinfo,
    #
    SVRG
end

