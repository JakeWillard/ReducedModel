

struct FlatSolver

    grd :: Grid
    A :: SparseMatrixCSC
    bmat_const :: Matrix{Float64}
    u0 :: Vector{Float64}
    ERROR_THRESHOLD :: Float64
    MAX_ITERATIONS :: Int64

end


function FlatSolver(Ny, Nt, Deltay, Deltat, d_L, B0; thresh=1e-6, MaxIt=100)

    grd = Grid(Ny, Nt, Deltay, Deltat, d_L)
    
    A = grd.OFF_BOUNDARY * (grd.Dxx + grd.Dyy - grd.Dtt)
    A += grd.T0
    A += grd.T1 
    A += grd.Y0
    A += grd.YF * grd.Dy

    psi0 = [(i-1)*grd.dy*B0 for i=1:grd.Ny]

    bmat_const = zeros(Ny, Nt)
    bmat_const[:,1] = psi0[:]
    bmat_const[:,2]= psi0[:]
    bmat_const[Ny,3:end] .= B0

    u0mat = zeros(Ny, Nt)
    for j=1:Nt
        u0mat[:,j] = psi0[:]
    end
    u0 = reshape(u0mat, Ny*Nt)

    return FlatSolver(grd, A, bmat_const, u0, thresh, MaxIt)
end


function rhs(u, solver::FlatSolver)

    psi = reshape(u, solver.grd.Ny, solver.grd.Nt)
    Bm = (psi[2,:] - psi[1,:]) / solver.grd.dy
    Uout = sqrt(1 - solver.grd.d_L^2) * Bm
    Vin = abs.(tanh.(asinh.(Uout)))
    dpsi = Vin .* Bm
    dpsi[:] .= 0.2

    bmat = solver.bmat_const[:,:]
    for j=3:solver.grd.Nt 
        bmat[1,j] = bmat[1,j-1] + solver.grd.dt * (dpsi[j-1] + dpsi[j])/2
    end

    return reshape(bmat, solver.grd.Ny*solver.grd.Nt)
end


function solve(solver::FlatSolver)

    u = solver.u0[:]
    b = rhs(u, solver)

    for dummy=1:solver.MAX_ITERATIONS

        u_next = solver.A \ b
        err = norm(u_next - u)
        u[:] = u_next[:]
        b[:] = rhs(u, solver)

        if err < solver.ERROR_THRESHOLD
            return reshape(u, solver.grd.Ny, solver.grd.Nt)
        end

        println(err)
    end

    println("Did not converge.")
    return reshape(u, solver.grd.Ny, solver.grd.Nt)
end