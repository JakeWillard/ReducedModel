
struct Parameters

    Ny :: Int64
    dy :: Float64
    dt :: Float64
    d_L :: Float64
    kappa :: Vector{Float64}
    sig0 :: Float64

end



function Parameters(sig0, d_L; Ny=200, hy=1.0, cfl=0.2)

    dy = hy / Ny
    dt = cfl * dy

    kappa = ones(Ny)*2*d_L^2/hy

    # kappa = zeros(Ny)
    # ys = LinRange(0, hy, Ny)
    # kappa = [4*d_L^2 * (1 - y/hy) / hy for y in ys]

    return Parameters(Ny, dy, dt, d_L, kappa, sig0)
end


function simple_initial_values(par::Parameters)

    psi = zeros(par.Ny, 4)
    zeta = zeros(par.Ny, 3)

    psi0 = [(i-1)*par.dy*par.sig0 for i=1:par.Ny]
    for i=1:3
        psi[:,i] = psi0[:]
    end

    return psi, zeta
end

function improved_initial_values(buff, par::Parameters)

    c = par.kappa[2]*sqrt(par.sig0) / (par.kappa[2]*par.Ny*par.dy - 2) - buff
    a =  sqrt(par.sig0) - 2*c*par.Ny*par.dy
    ys = [(i-1)*par.dy for i=1:par.Ny]
    psi0 = [a*y + c*y^2 for y in ys]

    psi = zeros(par.Ny, 4)
    zeta = zeros(par.Ny, 3)
    for i=1:3
        psi[:,i] = psi0[:]
    end

    return psi, zeta
end

function initialize_approximate_solution(par::Parameters)

    Bx0 = sqrt(par.sig0)
    chi = (1 - par.d_L^2) / (1 + par.d_L^2)
    a = Bx0*chi
    b = 0.5*(1 - chi)/(par.Ny * par.dy)

    ys = [(i-1)*par.dy for i=1:par.Ny]
    psi0 = [a*y + b*y^2 for y in ys]

    psi = zeros(par.Ny, 4)
    zeta = zeros(par.Ny, 3)
    for i=1:3
        psi[:,i] = psi0[:]
    end

    return psi, zeta
end


function evolve!(psi, zeta, par::Parameters)

    # compute psi first derivatives (at all y values)
    psi_t = psi_t = (psi[:,3] - psi[:,2]) / par.dt
    psi_y = zeros(par.Ny)
    psi_y[2:end-1] = (psi[3:end,3] - psi[1:end-2,3]) / (2*par.dy)
    psi_y[1] = (psi[2,3] - psi[1,3]) / par.dy
    psi_y[end] = (psi[end,3] - psi[end-1,3]) / par.dy

    # compute norm of psi gradient (only at non-boundary y values)
    grad_psi = (psi_y[2:end-1].^2 - psi_t[2:end-1].^2) .* sign.(psi_y[2:end-1])

    # compute spacetime laplacian of psi (only at non-boundary y values)
    psi_yy = (psi[3:end,3] + psi[1:end-2,3] - 2*psi[2:end-1,3]) / par.dy^2
    psi_xx = -par.kappa[2:end-1] .* psi_y[2:end-1]
    psi_tt = (psi[2:end-1,3] + psi[2:end-1,1] - 2*psi[2:end-1,2]) / par.dt^2
    Lpsi = psi_xx + psi_yy - psi_tt

    # compute zeta y derivative (only at non-boundary y values)
    zeta_y = (zeta[3:end,2] - zeta[1:end-2,2]) / (2*par.dy)

    # evolve zeta at non-boundary points using a predictor-corrector sheme
    zeta_t = -grad_psi .* Lpsi - tanh.(zeta[2:end-1,2]) .* zeta_y
    zeta_p = zeta[2:end-1,2] + par.dt * zeta_t
    zeta_t_c = (zeta_t - grad_psi .* Lpsi - tanh.(zeta_p) .* zeta_y) / 2.0
    zeta[2:end-1,3] = zeta[2:end-1,2] + par.dt * zeta_t_c

    # extrapolate boundaries 
    zeta[1,3] = 2*zeta[2,3] - zeta[3,3]
    zeta[end,3]= 2*zeta[end-1,3] - zeta[end-2,3]
    
    # set correct boundary conditions for zeta
    # zeta[end,3] = 0# -acosh(psi_y[end]/sqrt(par.sig0))      # NOTE: likely point of failure if the code becomes unstable.
    # zeta[1,3] = -0.1#-asinh(par.d_L * sqrt(1 - par.d_L^2)*psi_y[1])

    # advect psi
    psi[:,4] = psi[:,3] - par.dt * tanh.(zeta[:,3]) .* psi_y
end


function fill_output_arrays!(psi_out, zeta_out, psi_in, zeta_in, par::Parameters)

    Nt = size(psi_out)[2]
    psi_out[:,1:4] = psi_in[:,:]
    zeta_out[:,2:4] = zeta_in[:,:]

    psi_mut = psi_in[:,:]
    zeta_mut = zeta_in[:,:]

    for i=4:Nt
        evolve!(psi_mut, zeta_mut, par)
        psi_out[:,i] = psi_mut[:,end]
        zeta_out[:,i] = zeta_mut[:,end]
        psi_mut[:,1:3] = psi_mut[:,2:4]
        zeta_mut[:,1:2] = zeta_mut[:,2:3]
    end

    return psi_out, zeta_out
end