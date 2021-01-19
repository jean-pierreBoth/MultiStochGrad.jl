# linear regression use case







using LinearAlgebra


include("Observations.jl")



# We have to implement 
# (observations: Observations, position: Array{Float64}, term : Int64) -> Float64
"""
# function make_linear_reg_term_value(observations:: Observations)

return a closure function linear\\_reg\\_term\\_value(position:: Array{Float64}, term :: Int64)

```math
term value = 1/2 * (y -  observation[term] \\cdot position)^2
```
Our position data is encoded in an array of dimension (1 + length of an observation)

"""
function make_linear_reg_term_value(observations:: Observations)
    function linear_reg_term_value(position:: Array{Float64,1}, term :: Int64)
        x, y = observations.datas[term], observations.value_at_data[term]
        @assert length(position) == length(x) "inequal vector length"
        0.5 * (y - dot(position, x))^2
    end
end





"""
# function make_linear_reg_term_gradient(observations::Observations)
 
returns a closure: function linear\\_reg\\_term\\_gradient(position::Array{Float64,1}, term::Int64, gradient::Array{Float64,1})

just take the gradient of the expression for linear\\_reg\\_term\\_value
"""
function make_linear_reg_term_gradient(observations::Observations)
    function linear_reg_term_gradient(position::Array{Float64,1}, term::Int64, gradient::Array{Float64,1})
        x, y = observations.datas[term], observations.value_at_data[term]
        e = y - dot(x, position)
    #        @debug "term_gradient e , x term"  e x term
        # it is the .= that forces affectation! Just = does not write into gradient. copy! is clear
        copy!(gradient, -e * x)        
    end
end





