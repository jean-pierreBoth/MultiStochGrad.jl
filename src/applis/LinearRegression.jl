# linear regression use case





include("../evaluator.jl")


    using LinearAlgebra




    # We have to implement 
    # (observations: Observations, position: Array{Float64}, term : Int64) -> Float64
    """
    # function term_value(observations:: Observations, position:: Array{Float64}, term :: Int64)

    ```math
    term value = 1/2 * (y - \\dot(obervation[term],position))^2
    ```
    """

    function linear_reg_term_value(observations:: Observations, position:: Array{Float64,1}, term :: Int64)
        x, y = observations.datas[term], observations.value_at_data[term]
        @assert length(position) == length(x) "inequal vector length"
        0.5 * (y - dot(position, x))^2
    end


    function linear_reg_term_gradient(observations::Observations, position::Array{Float64,1}, term::Int64, gradient::Array{Float64,1})
        #
        x, y = observations.datas[term], observations.value_at_data[term]
        e = y - dot(x, position)
#        @debug "term_gradient e , x term"  e x term
        # it is the .= that forces affectation! Just = does not write into gradient. copy! is clear
        copy!(gradient, -e * x)
    end




    """
    # function LinearRegression


    ## Fields

    - datas

    1. vectors in data have been augmented by one (with valuse of 1.) for interception term so that the length of
        vectors in datas is 1 + length of data in Observations

    - term_value

    - term_gradient
    """

    mutable struct LinearRegression{F<:Function, G<:Function}
        # length of observation + 1. Values 1. in first slot  of arrays.
        datas :: Vector{Tuple{Vector{Float64}, Float64}}
        #
        term_value :: F
        #
        term_gradient ::G
        # Must reformat data to take care of constraints.
        function LinearRegression{F,G}(observations::Observations) where {F <: Function,G <: Function}
            nbobs = length(observations.datas)
            datas = Vector{Tuple{Vector{Float64}, Float64}}(undef, nbobs)
            # add interception term
            obs_dim = length(observations.datas[1])
            our_dim = Dims{1}(obs_dim)
            for i in 1:nbobs
                obs = zeros(Float64, our_dim)
                obs[1:end] = observations.datas[i][1:end]
                datas[i] = (obs, observations.value_at_data[i])
            end
            new(datas, linear_reg_term_value, linear_reg_term_gradient)
        end
    end

