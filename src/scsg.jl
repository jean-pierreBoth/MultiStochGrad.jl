# SCSG implementation




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
    mini_batch_size_init :: Int64
    # related to B_0 in Paper. Fraction of terms to consider in initialisation of B_0
    large_batch_size_init :: Float64
end


function minimize(scsg_pb::SCSG, evaluation::Evaluator, initial_position::Vector{Float64})
end