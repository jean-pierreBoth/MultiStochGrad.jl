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

    We will make this structure a function lile object

"""
mutable struct TermFunction{F<:Function}
    eval :: F
    observations :: Observations
end




"""
A functor transforming a structure in a function 


so that something declared as : 
        ``f = TermFunction(myfonction, observations)``  
 becomes a function we can call as f(position, term) and execute
 myfonction
We now are sure the aval function passed as field eval in a TermFunction
struct has the right signature  

see Function like objects in Julia manual v1.4

"""
function (term_f::TermFunction)(position :: Vector{Float64}, term :: UInt)
    return term_f.eval(term_f.observations, position, term)
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
    observations :: Observations
    partial_value :: Function
    partial_gradient :: Function
end