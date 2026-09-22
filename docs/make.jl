using FractionalNeuralSampling
using Distributions
using CairoMakie

using Documenter
using Documenter.Remotes: GitHub
using DocumenterVitepress

format = DocumenterVitepress.MarkdownVitepress(;
    repo = "github.com/brendanjohnharris/FractionalNeuralSampling.jl",
    devbranch = "main",
    devurl = "dev"
)

pages = [
    "Home" => "index.md",
    "Quick start" => "quickstart.md",
    "Densities" => "densities.md",
    "Samplers" => "samplers.md",
    "Adaptive samplers" => "adaptive.md",
    "Noise processes" => "noise.md",
    "Solvers" => "solvers.md",
    "Boundaries" => "boundaries.md",
    "Reference" => "reference.md",
]

makedocs(;
    authors = "brendanjohnharris <bhar9988@uni.sydney.edu.au> and contributors",
    sitename = "FractionalNeuralSampling",
    format,
    # Explicit: the `origin` remote is `www.github.com/...`, which Documenter can't parse.
    repo = GitHub("brendanjohnharris", "FractionalNeuralSampling.jl"),
    doctest = false,
    # `:missing_docs` and `:docs_block` keep a `@docs` entry for an undocumented symbol from
    # failing the build while the docstrings are still being written.
    warnonly = [:missing_docs, :docs_block, :cross_references, :example_block],
    modules = [FractionalNeuralSampling],
    pages
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/brendanjohnharris/FractionalNeuralSampling.jl",
    target = "build",
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true
)
