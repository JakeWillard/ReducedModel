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

include("./spacetime.jl")
include("./harmonic.jl")
include("./precomputation.jl")
include("./evolution.jl")
include("./plotting.jl")
