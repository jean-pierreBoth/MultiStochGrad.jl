using Printf

using Logging
using Base.CoreLogging

include("evaluator.jl")
include("scsg.jl")

debug_log = stdout
logger = ConsoleLogger(stdout, CoreLogging.Info)
global_logger(logger)

# export list

######################################################################

