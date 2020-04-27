using Printf

using Logging
using Base.CoreLogging

#include("evaluator.jl")
#include("scsg.jl")

#include("applis/LinearRegression.jl")
#include("applis/LogisticRegression.jl")


debug_log = stdout
logger = ConsoleLogger(stdout, CoreLogging.Debug)
global_logger(logger)

# export list

######################################################################

