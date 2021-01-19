# logistic regression on mnist data

using MultiStochGrad

using  LinearAlgebra, Test
using Random, Distributions


include("../src/applis/LinearRegression.jl")


# the . means we use the module LinearRegressionMod in the current module, other wise refer toa package!


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
    observations= Observations(datas,values)
    nbterms = length(observations.datas)
    # define our evaluations
    linear_reg_term_value = make_linear_reg_term_value(observations)
    term_function =  TermFunction{typeof(linear_reg_term_value)}(linear_reg_term_value, nbterms, Dims{1}(3))
    #
    linear_reg_term_gradient = make_linear_reg_term_gradient(observations)
    term_gradient = TermGradient{typeof(linear_reg_term_gradient)}(linear_reg_term_gradient ,nbterms, Dims{1}(3))
    evaluator = Evaluator{typeof(linear_reg_term_value),typeof(linear_reg_term_gradient)}(term_function,term_gradient) 
    #
    scsg_pb = SCSG(0.1, 0.3, 1 , 0.7)
    # solve.
    nb_iter = 50
    position = fill(1., 3)
    initial_error = compute_value(evaluator, position)
    @info "\n initial error \n\n" initial_error
    position, value = minimize(scsg_pb, evaluator, nb_iter, position)
    @info "value , position " value  position
    @info "\n\n test_linear_regression_scsg done \n \n"
    value < 0.65 ? true : false
end



function test_linear_regression_svrg()
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
    observations= Observations(datas,values)
    nbterms = length(observations.datas)
    # define our evaluations 
    linear_reg_term_value = make_linear_reg_term_value(observations)
    term_function =  TermFunction{typeof(linear_reg_term_value)}(linear_reg_term_value, nbterms, Dims{1}(3))
    #
    linear_reg_term_gradient = make_linear_reg_term_gradient(observations)
    term_gradient = TermGradient{typeof(linear_reg_term_gradient)}(linear_reg_term_gradient ,nbterms, Dims{1}(3))
    evaluator = Evaluator{typeof(linear_reg_term_value),typeof(linear_reg_term_gradient)}(term_function,term_gradient)
    #
    svrg_pb = SVRG(100, 0.2)
    # solve.
    nb_iter = 60
    position = fill(1., 3)
    initial_error = compute_value(evaluator, position)
    @info "\n initial error \n\n" initial_error
    position, value = minimize(svrg_pb, evaluator, nb_iter, position)
    @info "value , position ", value ,  position
    @info "\n\n test_linear_regression_svrg done \n \n"
    value < 0.65 ? true : false
end