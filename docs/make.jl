using Documenter, GaussianRandomFields

DocMeta.setdocmeta!(GaussianRandomFields, :DocTestSetup, :(using GaussianRandomFields, Plots, Printf); recursive=true)

makedocs(
    modules=[GaussianRandomFields],
    authors="PieterjanR",
    repo="https://github.com/PieterjanRobbe/GaussianRandomFields.jl/blob/{commit}{path}#{line}",
    sitename="GaussianRandomFields.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical="https://PieterjanRobbe.github.io/GaussianRandomFields.jl",
        edit_link="main",
        assets=String[],
    ),
    pages = [
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "API" => "API.md"
    ],
    checkdocs=:exports
)

deploydocs(
    repo = "github.com/PieterjanRobbe/GaussianRandomFields.jl",
    devbranch="main",
)
