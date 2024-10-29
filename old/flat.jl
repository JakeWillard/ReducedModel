
function Vin(u, grd::Grid)

    psi = reshape(u, grd.Ny, grd.Nt)
    Bm = (psi[2,:] - psi[1,:]) / grd.dy
    Uout = sqrt(1 - grd.d_L^2) * Bm

    return tanh.(asinh.(Uout))
end

function Avar(u, grd::Grid)

    psi = reshape(u, grd.Ny, grd.Nt)
    Bm = (psi[2,:] - psi[1,:]) / grd.dy
    Uout = sqrt(1 - grd.d_L^2) * Bm
    Vin = abs.(tanh.(asinh.(Uout)))
    Vin_diag = kron(spdiagm(0 => Vin), sparse(I, grd.Ny, grd.Ny))

    Avar = grd.Y0 * (grd.Dt - Vin_diag * grd.Dy)
    return Avar
end


function flat_solve(B0, grd::Grid; Nits=100, thresh=1e-6)

    Aconst = grd.OFF_BOUNDARY * (grd.Dyy - grd.Dtt)
    Aconst += grd.YF * grd.Dy
    Aconst += grd.T0
    Aconst += grd.T1
    
    psi0 = [(i-1)*grd.dy*B0 for i=1:grd.Ny]

    bmat = zeros(grd.Ny, grd.Nt)
    bmat[grd.Ny,:] .= B0
    bmat[:,1] = psi0
    bmat[:,2] = psi0
    b = reshape(bmat, grd.Ny*grd.Nt)

    u0mat = zeros(grd.Ny, grd.Nt)
    for j=1:grd.Nt
        u0mat[:,j] = psi0
    end 
    u = reshape(u0mat, grd.Ny*grd.Nt)

    for dummy=1:Nits 
        u_next = (Aconst + Avar(u, grd)) \ b
        err = norm(u - u_next)
        u[:] = u_next[:]
        if err < thresh
            return reshape(u, grd.Ny, grd.Nt)
        end
        println(err)
    end
    
    println("Did not converge.")
    return reshape(u, grd.Ny, grd.Nt)
end