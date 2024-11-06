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



pre = SchwarzchildMidplane(4.0, 1.0, 1.0, pi/3.0, pi/3.0; savefile="./test.h5")


sol = basic_simulation(100, pre=Precomputables("./test.h5"; UPSTREAM_B=0.2, HYPERVISCOSITY=5.0))


# st = Spacetime("./test.h5")
# st.METRIC([1.0, 1.0, ])

plotsimulation(sol, size=(1000, 1000))