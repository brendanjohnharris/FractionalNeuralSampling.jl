using FractionalNeuralSampling
using Documenter

DocMeta.setdocmeta!(FractionalNeuralSampling, :DocTestSetup, :(using FractionalNeuralSampling); recursive=true)

makedocs(;
    modules=[FractionalNeuralSampling],
    authors="brendanjohnharris <bhar9988@uni.sydney.edu.au> and contributors",
    sitename="FractionalNeuralSampling.jl",
    format=Documenter.HTML(;
        canonical="https://brendanjohnharris.github.io/FractionalNeuralSampling.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/brendanjohnharris/FractionalNeuralSampling.jl",
    devbranch="main",
)
