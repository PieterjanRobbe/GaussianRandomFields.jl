using Documenter, GaussianRandomFields

DocMeta.setdocmeta!(GaussianRandomFields, :DocTestSetup, :(using GaussianRandomFields); recursive=true)

makedocs(
    sitename="GaussianRandomFields.jl",
    modules = [GaussianRandomFields],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
             "Home" => "index.md",
             "Tutorial" => "tutorial.md",
             "API" => "API.md"
    ],
    doctest=false,
    checkdocs=:exports
)

deploydocs(
    repo = "github.com/PieterjanRobbe/GaussianRandomFields.jl.git",
)
