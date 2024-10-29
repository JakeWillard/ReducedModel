
struct Parameters

    Ny :: Int64
    dy :: Float64
    dt :: Float64
    d_L :: Float64
    sig0 :: Float64
    Nsmooth :: Int64
    SMOOTHER :: SparseMatrixCSC

end


function Parameters(;Ny=200, hy=1.0, cfl=0.2, d_L=0.2, sig0=10.0, Nsmooth=20)
    
    # NOTE: Nsmooth must be an odd number
    dy = hy / Ny
    dt = cfl*dy

    S = spdiagm(0 => ones(Ny)) / 3
    S += spdiagm(1 => ones(Ny-1)) / 3
    S += spdiagm(-1 => ones(Ny-1)) / 3
    S[1,:] .= 0
    S[end:end,:] .=0 
    S[1,1] = 1
    S[end,end-1] = 1

    SMOOTHER = S ^ Nsmooth

    return Parameters(Ny, dy, dt, d_L, sig0, Nsmooth, SMOOTHER)
end


function cost_function(x, p)

    n_args = 3  # number of parameters at the front of p
    Ny, dy, dt = p[1:n_args]
    Ny = Int64(Ny)
    l1, l2 = x[1:2]

    # unpack
    zeta_0 = p[n_args+1:Ny+n_args]
    zeta_p = x[3:end]
    rhs = p[Ny+n_args+1:end]

    # y derivative at last value of t
    zeta_0_y = zeros(Ny)
    zeta_0_y[2:end-1] = (zeta_0[3:end] - zeta_0[1:end-2]) / (2*dy)
    zeta_0_y[1] = (zeta_0[2] - zeta_0[1]) / dy
    zeta_0_y[end] = (zeta_0[end] - zeta_0[end-1]) / dy

    # y derivative at next value of t
    zeta_p_y = zeros(Ny)
    zeta_p_y[2:end-1] = (zeta_p[3:end] - zeta_p[1:end-2]) / (2*dy)
    zeta_p_y[1] = (zeta_p[2] - zeta_p[1]) / dy
    zeta_p_y[end] = (zeta_p[end] - zeta_p[end-1]) / dy

    # averaged quantities
    zeta_y = (zeta_0_y + zeta_p_y) / 2.0
    zeta_t = (zeta_p - zeta_0) / dt
    zeta_avg = (zeta_0 + zeta_p) / 2.0

    # inertial term of force law
    lhs = cosh.(zeta_avg) .* zeta_t + sinh.(zeta_avg) .* zeta_y

    # boundary values TODO: figure out the correct thing
    zeta_upstream = -0.1
    zeta_downstream = -0.1

    return norm(lhs[2:end-1] - rhs) + l1*(x[end] - zeta_upstream) + l2*(x[1] - zeta_downstream)
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
    kappa = ones(par.Ny-2) * 2 * par.d_L^2 / (par.Ny*par.dy)
    psi_yy = (psi[3:end,3] + psi[1:end-2,3] - 2*psi[2:end-1,3]) / par.dy^2
    psi_xx = -kappa .* psi_y[2:end-1]
    psi_tt = (psi[2:end-1,3] + psi[2:end-1,1] - 2*psi[2:end-1,2]) / par.dt^2
    Lpsi = psi_xx + psi_yy - psi_tt

    # compute zeta y derivative (only at non-boundary y values)
    zeta_y = (zeta[3:end,2] - zeta[1:end-2,2]) / (2*par.dy)

    # evolve zeta at non-boundary points using a predictor-corrector sheme
    zeta_t = -grad_psi .* Lpsi - tanh.(zeta[2:end-1,2]) .* zeta_y
    zeta_p = zeta[2:end-1,2] + par.dt * zeta_t
    zeta_t_c = (zeta_t - grad_psi .* Lpsi - tanh.(zeta_p) .* zeta_y) / 2.0
    zeta[2:end-1,3] = zeta[2:end-1,2] + par.dt * zeta_t_c

    # boundary conditions
    zeta[1,3] = -0.1
    zeta[end,3]= zeta[end-1,3]

    # use this value as initial guess for implicit scheme
    # p = zeros(1 + 2*par.Ny)
    # x0 = zeros(par.Ny + 2)
    # p[1:3] = [par.Ny, par.dy, par.dt]
    # p[4:par.Ny+3] = zeta[:,2]
    # p[par.Ny+4:end] = -(grad_psi .* Lpsi)[:]
    # x0[3:end] = zeta[:,3]
    # prob = OptimizationProblem(OptimizationFunction(cost_function), x0, p)
    # zeta[:,3] = solve(prob, Optim.NelderMead())[3:end]

    # smooth out the result
    zeta[:,3] = par.SMOOTHER * zeta[:,3]

    # advect psi
    psi[:,4] = psi[:,3] - par.dt * tanh.(zeta[:,3]) .* psi_y


    # apply time averaging
    zeta[:,3] = (zeta[:,3] + zeta[:,2] + zeta[:,1]) / 3.0
    psi[:,4] = (psi[:,4] + psi[:,3] + psi[:,2] + psi[:,1]) / 4.0


    # smooth psi
    # psi[:,4] = par.SMOOTHER * psi[:,4]
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



function simple_simulation(Nt; par=Parameters())

    psi_in, zeta_in = simple_initial_values(par)

    psi_out = zeros(par.Ny, Nt)
    zeta_out = zeros(par.Ny, Nt)
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

    return transpose(psi_out), transpose(zeta_out)
end