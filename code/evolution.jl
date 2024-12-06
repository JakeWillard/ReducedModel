
# Method drawn from https://pages.cs.wisc.edu/~chaol/data/cs777/stam-stable_fluids.pdf


struct Solution

    PLASMA_RAPIDITY :: Matrix{Float64}
    FLUX_FUNCTION :: Matrix{Float64}
    F_SQUARED :: Matrix{Float64}
    RECONNECTION_RATE :: Vector{Float64}
    Y_VALUES :: Vector{Float64}
    T_VALUES :: Vector{Float64}

    Solution(zeta::Matrix{Float64}, psi::Matrix{Float64}, pre::Precomputables) = begin
        
        Nt = size(zeta)[2]
        T_VALUES = [i*pre.T_SPACING for i=0:Nt-1]

        # compute grad(psi)^2
        F_SQUARED = zeros(pre.Y_NUMBER, Nt)
        F_SQUARED[:,1] = (pre.Dy * psi[:,1]).^2 - (psi[:,2] - psi[:,1]).^2 / pre.T_SPACING
        F_SQUARED[:,end] = (pre.Dy * psi[:,end]).^2 - (psi[:,end] - psi[:,end-1]).^2 / pre.T_SPACING
        for i=2:Nt-1
            F_SQUARED[:,i] = (pre.Dy * psi[:,i]).^2 - (psi[:,i+1] - psi[:,i-1]).^2 / (2*pre.T_SPACING)
        end

        # compute reconnection rate
        # RECONNECTION_RATE = zeros(Nt)
        # RECONNECTION_RATE[2:end-1] = (psi[1,3:end] - psi[1,1:end-2]) / 2*pre.T_SPACING
        # RECONNECTION_RATE[1] = RECONNECTION_RATE[2]
        # RECONNECTION_RATE[end] = RECONNECTION_RATE[end-1]
        # RECONNECTION_RATE = RECONNECTION_RATE[:] / pre.UPSTREAM_B

        RECONNECTION_RATE = abs.(tanh.(zeta[1,:]))

        new(zeta, psi, F_SQUARED, RECONNECTION_RATE, pre.Y_POINTS, T_VALUES)
    end
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
        k4 = -tanh.(zeta_interp.(y + k3*dt))
        y += dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    end

    return y, zeta_interp.(y)
end


function electromagnetic_acceleration(t, zeta, psi, pre::Precomputables)

    # compute derivatives of psi
    psi_y = pre.Dy * psi[:,3]
    psi_t = (psi[:,3] - psi[:,2]) / pre.T_SPACING
    grad_psi_sqrd = psi_y.^2 - psi_t.^2
    grad_psi = sqrt.(abs.(grad_psi_sqrd)) .* sign.(grad_psi_sqrd)
    psi_xx = -2*pre.ASPECT_RATIO^2 / pre.Y_POINTS[end] .* psi_y
    psi_yy = pre.Dyy * psi[:,3]
    psi_tt = (psi[:,3] + psi[:,1] - 2*psi[:,2]) / pre.T_SPACING^2

    # y component of gradient of psi
    grad_psi_y = [(pre.ST.METRIC_INVERSE([t, 0.0, pre.Y_POINTS[i], 0.0]) * [psi_t[i], psi_y[i]])[2] for i=1:pre.Y_NUMBER]

    return grad_psi_y .* (psi_tt - psi_xx - psi_yy) ./ (cosh.(zeta).^2)
end


function gravitational_acceleration(t, zeta, pre::Precomputables)

    u = hcat(cosh.(zeta), sinh.(zeta))
    a = [dot(u[i,:], pre.ST.GAMMA([t, 0.0, pre.Y_POINTS[i], 0.0])*u[i,:]) / u[i,1]^2 for i=1:pre.Y_NUMBER]

    return a
end



function compute_w1(t, zeta, psi, pre::Precomputables)

    a_em = electromagnetic_acceleration(t, zeta, psi, pre)
    a_g = gravitational_acceleration(t, zeta, pre)

    # apply force with Euler
    return zeta + pre.T_SPACING * (a_em + a_g)
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
    zeta_extrap[pre.Y_NUMBER+1:2*pre.Y_NUMBER] = zeta[:] .- zeta_b
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
    psi_ext[pre.Y_NUMBER+1:end] = [psi[end] + i*pre.Y_SPACING*pre.UPSTREAM_B for i=1:10]
    y_ext = zeros(pre.Y_NUMBER + 10)
    y_ext[1:pre.Y_NUMBER] = pre.Y_POINTS
    y_ext[pre.Y_NUMBER+1:end] = [pre.Y_POINTS[end] + i*pre.Y_SPACING for i=1:10]

    # use this to make psi interpolation function, then backtrace to compute advected psi
    psi_interp = linear_interpolation(y_ext, psi_ext, extrapolation_bc=Line()) 
    new_ys, _ = backtrace(zeta, pre)
    psi_advected = psi_interp.(new_ys)

    return psi_advected
end
    


function evolve_variables(t, zeta, psi, pre::Precomputables)

    w1 = compute_w1(t, zeta, psi, pre)
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

    T_POINTS = LinRange(0, pre.T_SPACING*Nt, Nt)
    for i=4:Nt
        zeta_i, psi_i = evolve_variables(T_POINTS[i-1], zeta_out[:,i-1], psi_out[:,i-3:i-1], pre)
        zeta_out[:,i] = zeta_i[:]
        psi_out[:,i] = psi_i[:]
    end

    return Solution(zeta_out, psi_out, pre)
end


# function out_of_equilibrium(Nt; pre=Precomputables())

#     psi0 = pre.UPSTREAM_B * pre.Y_POINTS .^5
#     zeta0 = zeros(pre.Y_NUMBER)

#     psi_out = zeros(pre.Y_NUMBER, Nt)
#     zeta_out = zeros(pre.Y_NUMBER, Nt)
#     for i=1:3
#         psi_out[:,i] = psi0[:]
#         zeta_out[:,i] = zeta0[:]
#     end

#     for i=4:Nt
#         zeta_i, psi_i = evolve_variables(zeta_out[:,i-1], psi_out[:,i-3:i-1], pre)
#         zeta_out[:,i] = zeta_i[:]
#         psi_out[:,i] = psi_i[:]
#     end

#     return Solution(zeta_out, psi_out, pre)

# end