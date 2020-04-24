push!(LOAD_PATH, "../src/", "../test/src/")

DOCUMENTER_DEBUG=true

using Documenter, MultiGradStoch


makedocs(
    format = Documenter.HTML(prettyurls = false),
    sitename = "MultiGradStoch",
    pages = Any[
        "Introduction" => "INTRO.md",
        "MultiStochGrad.jl " => "index.md",
        "evaluator.jl" => "interface.md",
        "scsg.jl"   => "scsg.md",
        "Tests" => "Test.md",
        "Applications" => "Applications"
    ]
)

