using Printf

include("multistochgradrs.jl")


"""

# SCSG

This is the structure for Stochastic Controlled Stochastic Gradient

"""

mutable struct SCSG
    # initial step size
    eta_zero :: Float64
    # fraction of nbterms to consider in initialization of mⱼ governing evolution of nb small iterations
    m_zero : Float64
    # m_0 in the paper
    mini_batch_size_init :: UInt64
    # related to B_0 in Paper. Fraction of terms to consider in initialisation of B_0
    large_batch_size_init :: Float64
    #
    nb_iter:: UInt64
end



"""

# Observations

datas list of data vector
a vector of value at each data
"""

mutable struct Observations 
    # 
    datas :: Vector{Vector{Float64}}
    value_at_data :: Vector{Float64}
end


######################################################################


"""

# TermFunction

A structure grouping observations and an evaluation function


"""
mutable struct TermFunction{F<:Function}
    eval :: F
    observations :: Observations
    function TermFunction(eval : F, observations : Observations)
        # check signature
        dist = eval(observations, observations.datas[1], 1)
        @assert abs(dist - observations.value_at_data[1]) < 1.E-10 "incoherent function"
        new(eval, observations)
    end
end


"""

"""

mutable struct TermGradient{F<:Function}
    eval :: F
    observations :: Observations
    function TermFunction(eval : F, observations : Observations)
        # check signature
        direction = rand(length(observations.datas[1]))
        dist = eval(observations, observations.datas[1], direction)
        # find an assertion
        new(eval, observations)
    end
end

#####################################################################

"""
#  SgdPb problem

   This struct contains the function that computes the value of the term of the sum
    and the function that computes the gradient of a term of the sum 


    partial_value must be a function of signature (Observations, Vector{Float64}, UInt) -> Float64

"""

mutable struct SgdPb
    #  A vector of observation , associated value
    partial_value :: TermFunction
    partial_gradient :: TermGradient
end