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

# SGD problem

This struct contains the function that computes the value of the term of the sum
    and the function that computes the gradient of a term of the sum 

"""

mutable struct SgdPb
end