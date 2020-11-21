using MultiStochGrad, Test


using Logging
using Base.CoreLogging

logger = ConsoleLogger(stdout, CoreLogging.Debug)
global_logger(logger)


include("TestLinearRegression.jl")

@testset "linreg" begin
@test test_linear_regression_scsg()
@test test_linear_regression_svrg()
end

# possibly  after download of mnist files
include("TestLogisticRegression.jl")

@testset "logistic_reg" begin
@test mnist_logistic_regression_svrg()
@test mnist_logistic_regression_scsg()
end