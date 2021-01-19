
"""

# struct Observations


## Fields

- datas : list of data vector one for each observations
- value\\_at\\_datas
For regressions problems for example length(datas) is number of observations.
    and length(datas[1]) is 1+dimension of observations data beccause of interception terms.
"""
mutable struct Observations 
    datas :: Vector{Vector{Float64}}
    value_at_data :: Vector{Float64}
    #
    function Observations(datas::Vector{Vector{Float64}}, values::Vector{Float64})
        new(datas, values)
    end
end

