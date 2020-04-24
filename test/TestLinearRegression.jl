# logistic regression on mnist data

using MultiStochGrad

using  LinearAlgebra, Test
using Random, Distributions
using Printf




using LinearRegressionMod


function test_linear_regression_scsg()
    true_coefficients = [13.37, -4.2, 3.14]
    # sample noise according to a N(0,1)
    datas = Vector{Vector{Float64}}()
    values = Vector{Float64}()
    d = Normal()
    for i in 1:100
        v = Vector{Float64}()
        push!(v,1.)
        push!(v, rand())
        push!(v, rand())
        y = dot(true_coefficients, v) + rand(d)
        push!(datas, v)
        push!(values, y)
    end
    observations= LinearRegressionMod.Observations(datas,values)
    # define our problem
    linreg = LinearRegressionMod.LinearRegression(observations)
    # define our evaluations 
    term_function =  TermFunction(LinearRegressionMod.LinearRegression.term_value, observations, Dims{1}(3))
    term_gradient = TermGradient(LinearRegression.term_gradient ,observations, Dims{1}(3))
    evaluator = Evaluator(term_function, term_gradient)
    # define parameters for scsg
    scsg_pb = SCSG(0.1, 0.1, 5 , 0.95)
    # solve.
    nb_iter = 50
    position = fill(1., 3)
    position, value = minimize(scsg_pb, evaluator, nb_iter, position)
    @printf(stdout, "value = %f, position = %f %f %f ", value , position)
end