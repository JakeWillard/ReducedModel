
using LinearAlgebra
using SparseArrays

struct Grid

    Ny :: Int64 
    Nt :: Int64
    dy :: Float64 
    dt :: Float64
    d_L :: Float64
    Dy :: SparseMatrixCSC 
    Dyy :: SparseMatrixCSC 
    Dxx :: SparseMatrixCSC
    Dt :: SparseMatrixCSC 
    Dtt :: SparseMatrixCSC
    Y0 :: SparseMatrixCSC
    YF :: SparseMatrixCSC
    T0 :: SparseMatrixCSC
    T1 :: SparseMatrixCSC
    #TF :: SparseMatrixCSC   
    ON_BOUNDARY :: SparseMatrixCSC
    OFF_BOUNDARY :: SparseMatrixCSC

end

function Grid(Ny, Nt, Deltay, Deltat, d_L)

    It = sparse(I, Nt, Nt)
    Iy = sparse(I, Ny, Ny)

    dy = Deltay / Ny
    dt = Deltat / Nt

    Dy_1d = spdiagm(1 => ones(Ny-1)) / (2*dy)
    Dy_1d += spdiagm(-1 => -ones(Ny-1)) / (2*dy)
    Dy_1d[1,:] = zeros(Ny)
    Dy_1d[1,1:2] = [-1, 1] / dy 
    Dy_1d[Ny,:] = zeros(Ny)
    Dy_1d[Ny,Ny-1:Ny] = [-1,1] / dy 
    Dy = kron(It, Dy_1d)
    dropzeros!(Dy)

    Dt_1d = spdiagm(1 => ones(Nt-1)) / (2*dt)
    Dt_1d += spdiagm(-1 => -ones(Nt-1)) / (2*dt)
    Dt_1d[1,:] = zeros(Nt)
    Dt_1d[1,1:2] = [-1, 1] / dt 
    Dt_1d[Nt,:] = zeros(Nt)
    Dt_1d[Nt,Nt-1:Nt] = [-1,1] / dt 
    Dt = kron(Dt_1d, Iy)
    dropzeros!(Dt)

    Dyy_1d = spdiagm(1 => ones(Ny-1)) / (2*dy)^2
    Dyy_1d += spdiagm(-1 => ones(Ny-1)) / (2*dy)^2
    Dyy_1d += spdiagm(0 => -2*ones(Ny)) / (2*dy)^2
    Dyy_1d[1,:] = zeros(Ny)
    Dyy_1d[1,1:3] = Dyy_1d[2,1:3]
    Dyy_1d[Ny,:] = zeros(Ny)
    Dyy_1d[Ny,Ny-2:Ny] = Dyy_1d[Ny-1,Ny-2:Ny]
    Dyy = kron(It, Dyy_1d)
    dropzeros!(Dyy)

    L_1d = spzeros(Ny,Ny)
    L_1d[1,1] = 1
    LEFT_VALUE = kron(It, L_1d)
    one_y2 = zeros(Ny)
    one_y2[2:end] = 1 ./ (LinRange(0, Deltay, Ny)[2:end].^2)
    Dxx = 2*d_L^2 * kron(It, spdiagm(0 => one_y2)) * (LEFT_VALUE - I)

    Dtt_1d = spdiagm(1 => ones(Nt-1)) / (2*dt)^2
    Dtt_1d += spdiagm(-1 => ones(Nt-1)) / (2*dt)^2
    Dtt_1d += spdiagm(0 => -2*ones(Nt)) / (2*dt)^2
    Dtt_1d[1,:] = zeros(Nt)
    Dtt_1d[1,1:3] = Dtt_1d[2,1:3]
    Dtt_1d[Nt,:] = zeros(Nt)
    Dtt_1d[Nt,Nt-2:Nt] = Dtt_1d[Nt-1,Nt-2:Nt]
    Dtt = kron(Dtt_1d, Iy)
    dropzeros!(Dtt)

    It_mod = sparse(I, Nt, Nt)
    It_mod[1,1] = 0.0
    It_mod[2,2] = 0.0
    dropzeros!(It_mod)

    Y0_1d = spzeros(Ny, Ny)
    Y0_1d[1,1] = 1
    Y0 = kron(It_mod, Y0_1d)

    YF_1d = spzeros(Ny, Ny)
    YF_1d[Ny,Ny] = 1
    YF = kron(It_mod, YF_1d)

    T0_1d = spzeros(Nt, Nt)
    T0_1d[1,1] = 1
    T0 = kron(T0_1d, Iy)

    T1_1d = spzeros(Nt, Nt)
    T1_1d[2,2] = 1
    T1 = kron(T1_1d, Iy)

    # Tf_1d = spzeros(Nt, Nt)
    # Tf_1d[Nt,Nt] = 1
    # Tf = kron(Tf_1d, Iy)

    ON_BOUNDARY = Y0 + YF + T0 + T1
    OFF_BOUNDARY = I - ON_BOUNDARY

    # By = spzeros(Ny,Ny)
    # By[1,1] = 1
    # By[Ny,Ny] = 1
    # Bt = spzeros(Nt, Nt)
    # Bt[1,1] = 1
    # ON_BOUNDARY = kron(Bt, Iy) + kron(It, By)
    # OFF_BOUNDARY = I - ON_BOUNDARY

    return Grid(Ny, Nt, dy, dt, d_L, Dy, Dyy, Dxx, Dt, Dtt, Y0, YF, T0, T1, ON_BOUNDARY, OFF_BOUNDARY)
end