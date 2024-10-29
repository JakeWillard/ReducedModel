using LinearAlgebra
using SparseArrays
using FFTW
using Interpolations
using FiniteDifferences
using Optimization
using OptimizationOptimJL
using ForwardDiff
using RecipesBase
using ProgressMeter
using Plots

include("./relativity.jl")
include("./precomputation.jl")
include("./evolution.jl")
include("./plotting.jl")


ht = HCoordTransform([1.0, 1.0, 1.0, 1.0]) do x
    diagm([-1.0, 1.0, 1.0, 1.0])
end


G = christoffel_symbols_in_plane(LinRange(0, 1, 3), LinRange(0, 1, 3), ht)

G(0.5, 0.5)