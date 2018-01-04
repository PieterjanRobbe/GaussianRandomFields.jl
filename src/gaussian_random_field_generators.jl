# TODO show methods for generators
## gaussian_random_field_samplers.jl ##

"""
`abstract type GaussianRandomFieldGenerator`

Astract type for a Gaussian random field generator, such as `KarhunenLoeveExapnsion`, `CirculantEmbedding`, `CholeskyFactorization`, etc.
"""
abstract type GaussianRandomFieldGenerator end

