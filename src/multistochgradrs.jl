# This file provides glue from julia to our multistochgrad rust crate
#
#

using Printf

using Logging
using Base.CoreLogging

# all julia function calling rust functions are suffixed by _rs, the rust name being without the _rs suffix

const libmultistochgradso = "libmultistochgrad_rs"




# to be called before anything
"""
# initialization of Julia DL_LOAD_PATH

 `function setRustlibPath(path::String)`

This function tells julia where is installed the rust dynamic library implementing the
MultiStochgrad  algorithms.
It must be called after `using MultiStochGrad` and before any function call

The argument is the path to the rust library.
"""
function setRustlibPath(path::String)
    push!(Base.DL_LOAD_PATH, path)
end


# a pointer to ObservationsPtr  will be in fact a pointer to FFiObservations
mutable struct ObservationsPtr
end


function initialize_observation_rs(obs::Observations)
    nbobs = length(obs)
    dim = length(obs[1])
    @assert nbobs == length(obs.value_at_data) "incoherent length in Observations"
    # 
    vec_data_ref = map(x-> pointer(x[1]), obs.datas)
    obs_ptr = ccall(
        (Symbol("initialize_observation"),libmultistochgradso),
        Ptr{ObservationsPtr},
        (UInt64, UInt64, Ref{Ptr{Float64}}, Ref{Float64},),
        UInt64(nbobs), UInt64(dim), vec_data_ref, obs.value_at_data
    )
    obs_ptr
end




function minimize_rs(sgdpb::SgdPb, initialpos::Vector{Float64})
    # transfer observation 
    # make C pointer fonction form TermFunction and Gradient function

end