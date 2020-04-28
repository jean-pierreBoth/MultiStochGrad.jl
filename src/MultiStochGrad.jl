module MultiStochGrad

using Printf

using Logging
using Base.CoreLogging
using LinearAlgebra


debug_log = stdout
logger = ConsoleLogger(stdout, CoreLogging.Info)
global_logger(logger)


include("scsg.jl")

include("applis/LinearRegression.jl")
include("applis/LogisticRegression.jl")

# export list



end

