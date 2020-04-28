push!(LOAD_PATH, "../src/", "../test/src/")

DOCUMENTER_DEBUG=true

using Documenter, MultiStochGrad


makedocs(
    format = Documenter.HTML(prettyurls = false),
    sitename = "MultiStochGrad",
    pages = Any[
        "Introduction" => "INTRO.md",
        "MultiStochGrad.jl " => "index.md",
        "scsg.jl" => "scsg.md",
    ]
)

