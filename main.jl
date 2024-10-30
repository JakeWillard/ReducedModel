using LinearAlgebra
using SparseArrays
using FFTW
using HDF5
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


st = FlatCartesian()
output = spacetime_to_grid(st, LinRange(0, 1, 100), LinRange(0, 1, 100), savefile="./test.h5")




sol = basic_simulation(1000; pre=Precomputables(UPSTREAM_B=1.0, HYPERVISCOSITY=5.0))

plotsimulation(sol, size=(1000, 1000))