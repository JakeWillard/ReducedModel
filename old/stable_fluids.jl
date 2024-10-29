
# Method inspired by https://pages.cs.wisc.edu/~chaol/data/cs777/stam-stable_fluids.pdf

struct Precomputables

    # Dimensional parameters
    Y_NUMBER :: Int64
    Y_SPACING :: Float64
    Y_POINTS :: Vector{Float64}
    T_SPACING :: Float64
    N_TRACE :: Int64

    # Derivative operators
    Dy :: SparseMatrixCSC
    Dyy :: SparseMatrixCSC

    # Physical parameters
    ASPECT_RATIO :: Float64
    UPSTREAM_B :: Float64
    DIFFUSION_VECTOR :: Vector{Float64}

    Precomputables(; Y_NUMBER=200, Y_SIZE=1.0, CFL=0.2, N_TRACE=10, ASPECT_RATIO=0.2, UPSTREAM_B=10.0, EFFECTIVE_VISCOSITY=0.3) = begin
        
        # compute grid spacing
        Y_SPACING = Y_SIZE / Y_NUMBER
        T_SPACING = CFL * Y_SPACING
        Y_POINTS = LinRange(0, Y_SIZE, Y_NUMBER)

        # compute first derivative matrix
        Dy = spdiagm(1 => ones(Y_NUMBER-1)) / (2 * Y_SPACING)
        Dy += spdiagm(-1 => -ones(Y_NUMBER-1)) / (2 * Y_SPACING)
        Dy[1,1:2] = [-1, 1] / Y_SPACING
        Dy[end, end-1:end] = [-1, 1] / Y_SPACING

        # compute second derivative matrix
        Dyy = spdiagm(0 => -2*ones(Y_NUMBER)) / Y_SPACING^2
        Dyy += spdiagm(1 => ones(Y_NUMBER-1)) / Y_SPACING^2
        Dyy += spdiagm(-1 => ones(Y_NUMBER-1)) / Y_SPACING^2
        Dyy[1,1:3] = Dyy[2,1:3]
        Dyy[end,end-2:end] = Dyy[end-1,end-2:end]

        # compute diffusion vector
        ks = fftfreq(4*Y_NUMBER) * 4 * Y_NUMBER
        DIFFUSION_VECTOR = exp.(-ks.^2 * T_SPACING * EFFECTIVE_VISCOSITY)

        new(Y_NUMBER, Y_SPACING, Y_POINTS, T_SPACING, N_TRACE, Dy, Dyy, ASPECT_RATIO, UPSTREAM_B, DIFFUSION_VECTOR)
    end
end


struct Solution

    PLASMA_RAPIDITY :: Matrix{Float64}
    FLUX_FUNCTION :: Matrix{Float64}
    Y_VALUES :: Vector{Float64}
    T_VALUE :: Vector{Float64}

end



function backtrace(zeta, pre::Precomputables)

    # interpolation function for zeta
    y = pre.Y_POINTS[:]
    zeta_interp = linear_interpolation(y, zeta, extrapolation_bc=Line()) 

    # trace backwards with RK4
    dt = pre.T_SPACING / pre.N_TRACE
    for dummy=1:pre.N_TRACE
        k1 = -tanh.(zeta_interp.(y))
        k2 = -tanh.(zeta_interp.(y + k1*dt/2))
        k3 = -tanh.(zeta_interp.(y + k2*dt/2))
        k4 = -tanh.(zeta_interp.(y + k3*dt/2))
        y += dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    end

    return y, zeta_interp.(y)
end


function compute_w1(zeta, psi, pre::Precomputables)

    # compute derivatives of psi
    psi_y = pre.Dy * psi[:,3]
    psi_t = (psi[:,3] - psi[:,2]) / pre.T_SPACING
    grad_psi = sqrt.(abs.(psi_y.^2 - psi_t.^2))
    psi_xx = -2*pre.ASPECT_RATIO^2 / (pre.Y_NUMBER*pre.Y_SPACING) .* psi_y
    psi_yy = pre.Dyy * psi[:,3]
    psi_tt = (psi[:,3] + psi[:,1] - 2*psi[:,2]) / pre.T_SPACING^2

    # apply force with Euler
    return zeta + pre.T_SPACING * grad_psi .* (psi_xx + psi_yy - psi_tt) 
end


function compute_w2(zeta, pre::Precomputables)

    _, w2 = backtrace(zeta, pre)

    return w2
end



function compute_w3(zeta, psi, pre::Precomputables)

    # compute inner boundary value
    Uin = pre.ASPECT_RATIO * sqrt(1 - pre.ASPECT_RATIO^2) * (pre.Dy * psi)[1]
    zeta_b = -asinh(Uin)

    # extrapolate a periodic function with the right boundary conditions
    zeta_extrap = zeros(4*pre.Y_NUMBER)
    zeta_extrap[1:pre.Y_NUMBER] = -reverse(zeta .- zeta_b)
    zeta_extrap[2*pre.Y_NUMBER+1:end] = reverse(zeta_extrap[1:2*pre.Y_NUMBER])
    zeta_extrap[:] = zeta_extrap .+ zeta_b

    # perform FFT and apply diffusion
    zeta_fft = fft(zeta_extrap)
    zeta_fft[:] = pre.DIFFUSION_VECTOR .* zeta_fft

    # invert fft and return original domain
    w3 = real.(ifft(zeta_fft))[pre.Y_NUMBER+1:2*pre.Y_NUMBER]
    return w3
end


function advect_psi(psi, zeta, pre::Precomputables)

    # extend domain of psi to ensure UPSTREAM_B is the upstream flux density
    psi_ext = zeros(pre.Y_NUMBER + 10)
    psi_ext[1:pre.Y_NUMBER] = psi[:]
    psi_ext[pre.Y_NUMBER+1:end] = [psi[end] + i*pre.Y_SPACING*pre.UPSTREAM_B for i=0:9]
    y_ext = zeros(pre.Y_NUMBER + 10)
    y_ext[1:pre.Y_NUMBER] = pre.Y_POINTS
    y_ext[pre.Y_NUMBER+1:end] = [pre.Y_POINTS[end] + i*pre.Y_SPACING for i=0:9]

    # use this to make psi interpolation function, then backtrace to compute advected psi
    psi_interp = linear_interpolation(y_ext, psi_ext, extrapolation_bc=Line()) 
    new_ys, _ = backtrace(zeta, pre)
    psi_advected = psi_interp.(new_ys)

    return psi_advected
end
    


function evolve_variables(zeta, psi, pre::Precomputables)

    w1 = compute_w1(zeta, psi, pre)
    w2 = compute_w2(w1, pre)
    zeta_new = compute_w3(w2, psi[:,3], pre)
    psi_new = advect_psi(psi[:,3], zeta_new, pre)

    return zeta_new, psi_new
end



function basic_simulation(Nt; pre=Precomputables())

    # these initial conditions assume the aspect ratio isn't zero
    @assert pre.ASPECT_RATIO != 0

    hy = pre.Y_POINTS[end]
    Uin = pre.ASPECT_RATIO * sqrt(1 - pre.ASPECT_RATIO^2) * pre.UPSTREAM_B * exp(-2*pre.ASPECT_RATIO^2)
    psi0 = -(pre.UPSTREAM_B*hy/2) * exp.(2*pre.ASPECT_RATIO^2 * (1 .- pre.Y_POINTS/hy)) / pre.ASPECT_RATIO^2
    zeta0 = -asinh(Uin) * ones(pre.Y_NUMBER)

    psi_out = zeros(pre.Y_NUMBER, Nt)
    zeta_out = zeros(pre.Y_NUMBER, Nt)
    for i=1:3
        psi_out[:,i] = psi0[:]
        zeta_out[:,i] = zeta0[:]
    end

    for i=4:Nt
        zeta_i, psi_i = evolve_variables(zeta_out[:,i-1], psi_out[:,i-3:i-1], pre)
        zeta_out[:,i] = zeta_i[:]
        psi_out[:,i] = psi_i[:]
    end

    return transpose(zeta_out), transpose(psi_out), pre
end









