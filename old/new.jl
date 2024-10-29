using LinearAlgebra
using SparseArrays

struct SymCase

    Ny :: Int64 
    Nt :: Int64 
    hy :: Float64 
    ht :: Float64
    L :: SparseMatrixCSC
    Dy :: SparseMatrixCSC
    Dt :: SparseMatrixCSC
    b :: Vector{Float64}

end



function SymCase(sig0, curv::Function; Ny=200, Nt=200, hy=1.0, ht=1.0)

    # 1D Identity operators
    It = sparse(I, Nt, Nt)
    Iy = sparse(I, Ny, Ny)

    # distances
    dy = hy / Ny
    dt = ht / Nt

    # 2nd derivative in y
    Dyy_1d = spdiagm(1 => ones(Ny-1)) / (2*dy)^2
    Dyy_1d += spdiagm(-1 => ones(Ny-1)) / (2*dy)^2
    Dyy_1d += spdiagm(0 => -2*ones(Ny)) / (2*dy)^2
    Dyy = kron(It, Dyy_1d)

    # 2nd derivative in t 
    Dtt_1d = spdiagm(1 => ones(Nt-1)) / (2*dt)^2
    Dtt_1d += spdiagm(-1 => ones(Nt-1)) / (2*dt)^2
    Dtt_1d += spdiagm(0 => -2*ones(Nt)) / (2*dt)^2
    Dtt_1d[Nt,:] = zeros(Nt)
    Dtt_1d[Nt,Nt-2:Nt] = Dtt_1d[Nt-1,Nt-2:Nt] # correct at final time
    Dtt = kron(Dtt_1d, Iy)
    dropzeros!(Dtt)

    # derivative in y
    Dy_1d = spdiagm(1 => ones(Ny-1)) / (2*dy)
    Dy_1d += spdiagm(-1 => -ones(Ny-1)) / (2*dy)
    Dy_1d[1,:] = zeros(Ny)
    Dy_1d[1,1:2] = [-1, 1] / dy # correct at left boundary
    Dy_1d[Ny,:] = zeros(Ny)
    Dy_1d[Ny,Ny-1:Ny] = [-1,1] / dy # correct at right boundary
    Dy = kron(It, Dy_1d)
    dropzeros!(Dy)

    # derivative in t
    Dt_1d = spdiagm(1 => ones(Nt-1)) / (2*dt)
    Dt_1d += spdiagm(-1 => -ones(Nt-1)) / (2*dt)
    Dt_1d[1,:] = zeros(Nt)
    Dt_1d[1,1] = 1 # make identity at intial time boundary
    Dt_1d[2,:] = zeros(Nt)
    Dt_1d[2,1:2] = [-1, 1] / dt # couple only first and second times for Neumann condition
    Dt_1d[Nt,:] = zeros(Nt)
    Dt_1d[Nt,Nt-1:Nt] = [-1,1] / dt # correct at final time boundary
    Dt = kron(Dt_1d, Iy)
    dropzeros!(Dt)

    # 2nd derivative in x using curvature input
    ys = LinRange(0, hy, Ny)
    K_1d = spdiagm(0 => curv.(ys))
    Dxx = kron(It, K_1d) * Dy

    # compute spacetime Laplacian 
    L = -Dtt + Dxx + Dyy 

    # compute rhs vector 
    b_mat = zeros(Ny, Nt)
    b_mat[end,:] .= sqrt(sig0)
    b = reshape(b_mat, Ny*Nt)

    return SymCase(Ny, Nt, hy, ht, L, Dy, Dt, b)
end


function setup_iteration(u, case::SymCase)

    Ny = case.Ny 
    Nt = case.Nt

    # compute proper acceleration
    dpsi_dt = case.Dt * u 
    dpsi_dy = case.Dy * u
    V = reshape(dpsi_dt./dpsi_dy, Ny, Nt)
    V[:,end] .= 0 # since we defined the Dt operator to be the identity at this boundary
    theta = reshape(tanh.(V), Ny*Nt)
    dtheta_tau = (cosh.(theta) .* case.Dt*theta + sinh.(theta) .* case.Dy*theta)

    # compute acceleration vector
    at = reshape(sinh.(theta) .* dtheta_tau, Ny, Nt)
    ay = reshape(cosh.(theta) .* dtheta_tau, Ny, Nt)

    # make it so we get the right BCs
    at[1,:] .= 0
    at[:,1] .= 1
    ay[:,1] .= 0
    ay[:,2] .= 0

    










end
