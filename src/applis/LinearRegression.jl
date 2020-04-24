# linear regression use case





include("/home.1/jpboth/Julia/MultiStochGrad.jl/src/evaluator.jl")


    using LinearAlgebra




    # We have to implement 
    # (observations: Observations, position: Array{Float64}, term : Int64) -> Float64
    """
    # function term_value(observations:: Observations, position:: Array{Float64}, term :: Int64)

    ```math
    term value = 1/2 * (y - \\dot(obervation[term],position))^2
    ```
    """

    function term_value(observations:: Observations, position:: Array{Float64,1}, term :: Int64)
        x, y = observations.datas[term], observations.value_at_data[term]
        @assert length(position) == length(x) "inequal vector length"
        0.5 * (y - dot(position, x))^2
    end


    function term_gradient(observations::Observations, position::Array{Float64}, term::Int64, gradient::Array{Float64})
        #
        x, y = observations.datas[term], observations.value_at_data[term]
        e = y - dot(x, position)
        gradient = -e * x
    end




    """
    # function LinearRegression


    1. vectors in data have been augmented by one (with valuse of 1.) for interception term so that the length of
        vectors in datas is 1 + length of data in Observations

    2. The coefficients of the last class have been assumed to be 0 to take into account
        for the identifiability constraint (Cf  Machine Learning Murphy par 9.2.2.1-2)

    Our unknown is a 2 dimensional array  (length(Observations.datas[1]) + 1, nbclass -1)

    """

    mutable struct LinearRegression
        # length of observation + 1. Values 1. in first slot  of arrays.
        datas :: Vector{Tuple{Vector{Float64}, Float64}}
        #
        term_value :: Function
        #
        term_gradient ::Function
        # Must reformat data to take care of constraints.
        function LinearRegression(observations::Observations)
            nbobs = length(observations.datas)
            datas = Vector{Tuple{Vector{Float64}, Float64}}(undef, nbobs)
            # add interception term
            obs_dim = length(observations.datas[1]) + 1
            # we search coefficients as 2 dimensional arrays , one column by class
            # and each column corresponds to observations type
            our_dim = Dims{1}(obs_dim)
            for i in 1:nbobs
                obs = zeros(Float64, our_dim)
                obs[1] = 1.
                obs[2:end] = observations.datas[i][1:end]
                datas[i] = (obs, observations.value_at_data[i])
            end
            new(datas, term_value, term_gradient)
        end
    end

