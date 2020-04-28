using Printf

using Logging
using Base.CoreLogging

include("scsg.jl")

include("applis/LinearRegression.jl")
include("applis/LogisticRegression.jl")


debug_log = stdout
logger = ConsoleLogger(stdout, CoreLogging.Info)
global_logger(logger)

# export list

######################################################################

