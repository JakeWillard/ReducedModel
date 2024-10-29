# using LinearAlgebra
# using SparseArrays

struct Evolver

    Ny :: Int64
    dy :: Float64
    dt :: Float64
    d_L :: Float64
    sig0 :: Float64
    L :: SparseMatrixCSC

end


function Evolver(sig0, d_L; Ny=200, hy=1.0, cfl=0.2)

    dy = hy / Ny 
    dt = cfl * dy

    Dyy = spdiagm(1 => ones(Ny-1)) / (2*dy)^2
    Dyy += spdiagm(-1 => ones(Ny-1)) / (2*dy)^2
    Dyy += spdiagm(0 => -2*ones(Ny)) / (2*dy)^2

    Dy = spdiagm(1 => ones(Ny-1)) / (2*dy)
    Dy += spdiagm(-1 => -ones(Ny-1)) / (2*dy)
    ys = LinRange(0, hy, Ny)
    k = zeros(Ny)
    k[2:end] =  d_L^2 ./ ys[2:end]
    K = spdiagm(0 => k)
    Dxx = K * Dy

    L = Dyy + Dxx

    return Evolver(Ny, dy, dt, d_L, sig0, L)
end




function set_inner_boundary_point!(u, ev::Evolver)

    Bm0 = (u[2,2] - u[1,2]) / ev.dy
    theta0 = asinh(ev.d_L * Bm0 * sqrt(1 - ev.d_L^2))
    f0 = tanh(theta0)*Bm0

    Bm_guess = (u[3,3] - u[2,3]) / ev.dy
    theta_guess = asinh(ev.d_L * Bm_guess * sqrt(1 - ev.d_L^2))
    f1_guess = tanh(theta_guess)*Bm_guess

    # evolve boundary point using f1_guess
    u[1,3] = u[1,2] + ev.dt * (f0 + f1_guess) / 2.0

    # make a better guess
    Bm_guess = (u[2,3] - u[1,3]) / ev.dy
    theta_guess = asinh(ev.d_L * Bm_guess * sqrt(1 - ev.d_L^2))
    f1_guess = tanh(theta_guess)*Bm_guess

    # evolve again using better guess
    u[1,3] = u[1,2] + ev.dt * (f0 + f1_guess) / 2.0
end


function set_outer_boundary_point!(u, ev::Evolver)

    # approximate V at the boundary
    dpsi_dt = (u[end-1,3] - u[end-1,2]) / ev.dt
    dpsi_dy = (u[end,2] - u[end-1,2]) / ev.dy
    V = dpsi_dt / dpsi_dy

    # compute B0 
    B0 = sqrt(ev.sig0) / sqrt(1 - V^2)
    
    u[end,3] = u[end-1,3] + ev.dy * B0
end


function evolve_non_boundary_points!(u, ev::Evolver)

    u[2:end-1,3] = 2*u[2:end-1,2] - u[2:end-1,1] + ev.dt^2 * (ev.L * u[:,2])[2:end-1]
end


