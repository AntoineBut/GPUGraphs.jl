using GPUGraphs
using Documenter

DocMeta.setdocmeta!(GPUGraphs, :DocTestSetup, :(using GPUGraphs); recursive = true)

makedocs(;
    modules = [GPUGraphs],
    authors = "AntoineBut",
    sitename = "GPUGraphs.jl",
    format = Documenter.HTML(;
        canonical = "https://AntoineBut.github.io/GPUGraphs.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/AntoineBut/GPUGraphs.jl", devbranch = "main")
