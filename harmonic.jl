

function first_derivative_matrix(N, h)

    dr = h / N

    D = spdiagm(1 => ones(N-1)) / (2*dr)
    D += spdiagm(-1 => ones(N-1)) / (2*dr)
    D[1,1:2] = [-1, 1] / dr
    D[end,end-1:end] = [-1, 1] / dr

    return D
end


function second_derivative_matrix(N, h)

    dr = h / N

    D = spdiagm(1 => ones(N-1)) / dr^2
    D += spdiagm(-1 => ones(N-1)) / dr^2
    D += spdiagm(0 => -2*ones(N)) / dr^2
    D[1,1:3] = D[2,1:3]
    D[end,end-2:end] = D[end-1,end-2:end]

    return D
end


function rhs_values(A0)

    Ap = zeros(size(A0))
    Ap[1,:,:,:] = A0[1,:,:,:]
    Ap[2,:,:,:] = A0[2,:,:,:]
    Ap[:,1,:,:] = A0[:,1,:,:]
    Ap[:,end,:,:] = A0[:,end,:,:]
    Ap[:,:,1,:] = A0[:,:,1,:]
    Ap[:,:,end,:] = A0[:,:,end,:]
    Ap[:,:,:,1] = A0[:,:,:,1]
    Ap[:,:,:,end] = A0[:,:,:,end]

    return Ap
end



function harmonic_coordinate_transformation(DOMAIN_LB::Vector{Float64}, DOMAIN_UB::Vector{Float64}, ST::Spacetime; GRID_DIMENSIONS=Int64[50, 5, 50, 5], PRECOMPILE_INVERSE=false)

    # define gradient vector of volume element per volume density
    grad_vol(x) = ST.METRIC_INVERSE(x) * ForwardDiff.gradient(ST.VOLUME_ELEMENT, x) / ST.VOLUME_ELEMENT(x)

    # label grid dimensions and total number of gridpoints
    Nt, Nx, Ny, Nz = GRID_DIMENSIONS
    Ng = Nt*Nx*Ny*Nz

    # label the original domain bounds
    t0, x0, y0, z0 = DOMAIN_LB
    tf, xf, yf, zf = DOMAIN_UB

    # construct a boundary operator which is identity on the boundary and sets to zero elsewhere
    Bvals = zeros(Nt, Nx, Ny, Nz)
    Bvals[1,:,:,:] .= 1.0
    Bvals[2,:,:,:] .= 1.0
    Bvals[:,1,:,:] .= 1.0
    Bvals[:,end,:,:] .= 1.0
    Bvals[:,:,1,:] .= 1.0
    Bvals[:,:,end,:] .= 1.0
    Bvals[:,:,:,1] .= 1.0
    Bvals[:,:,:,end] .= 1.0
    B = spdiagm(0 => reshape(Bvals, Ng))

    # construct vectors holding the values of the original coordinates at each gridpoint
    tvec = vcat([LinRange(t0, tf, Nt) for _=1:Nx*Ny*Nz]...)
    xvec = vcat([vcat([ones(Nt)*xi for xi in LinRange(x0, xf, Nx)]...) for _=1:Ny*Nz]...)
    yvec = vcat([vcat([ones(Nt*Nx)*yi for yi in LinRange(y0, yf, Ny)]...) for _=1:Nz]...)
    zvec = vcat([ones(Nt*Nx*Ny)*zi for zi in LinRange(z0, zf, Nz)]...)

    # initialize vectors for the diagonals of operators Gt, Gx, Gy, and Gz
    Gt_vals = zeros(Ng)
    Gx_vals = zeros(Ng)
    Gy_vals = zeros(Ng)
    Gz_vals = zeros(Ng)

    # these operators represent multiplication by a component of grad_vol at each point on the grid
    for i=1:Ng

        del_vol = grad_vol([tvec[i], xvec[i], yvec[i], zvec[i]])
        Gt_vals[i] = del_vol[1]
        Gx_vals[i] = del_vol[2]
        Gy_vals[i] = del_vol[3]
        Gz_vals[i] = del_vol[4]

    end
    Gt = spdiagm(0 => Gt_vals)
    Gx = spdiagm(0 => Gx_vals)
    Gy = spdiagm(0 => Gy_vals)
    Gz = spdiagm(0 => Gz_vals)

    # 1D identity operations
    Ix = sparse(I, Nx, Nx)
    Iy = sparse(I, Ny, Ny)
    Iz = sparse(I, Nz, Nz)
    It = sparse(I, Nt, Nt)

    # derivative operations
    Dt = kron(Iz, kron(Iy, kron(Ix, first_derivative_matrix(Nt, ht))))
    Dx = kron(Iz, kron(Iy, kron(first_derivative_matrix(Nx, hx), It)))
    Dy = kron(Iz, kron(first_derivative_matrix(Ny, hy), kron(Ix, It)))
    Dz = kron(first_derivative_matrix(Nz, hz), kron(Iy, kron(Ix, It)))
    Dtt = kron(Iz, kron(Iy, kron(Ix, second_derivative_matrix(Nt, ht))))
    Dxx = kron(Iz, kron(Iy, kron(second_derivative_matrix(Nx, hx), It)))
    Dyy = kron(Iz, kron(second_derivative_matrix(Ny, hy), kron(Ix, It)))
    Dzz = kron(second_derivative_matrix(Nz, hz), kron(Iy, kron(Ix, It)))

    # compute the linear differential operator on the LHS of the system
    L = -Dtt + Dxx + Dyy + Dzz + Gt*Dt + Gx*Dx + Gy*Dy + Gz*Dz

    # compute the LHS matrix including boundary conditions
    LHS = B + (sparse(I, Ng, Ng) - B)*L

    # compute RHS vectors
    RHS_T = reshape(rhs_values(reshape(tvec, (Nt, Nx, Ny, Nz))), Ng)
    RHS_X = reshape(rhs_values(reshape(xvec, (Nt, Nx, Ny, Nz))), Ng)
    RHS_Y = reshape(rhs_values(reshape(yvec, (Nt, Nx, Ny, Nz))), Ng)
    RHS_Z = reshape(rhs_values(reshape(zvec, (Nt, Nx, Ny, Nz))), Ng)

    # compute new coordinates on the grid
    Tvec = LHS \ RHS_T
    Xvec = LHS \ RHS_X
    Yvec = LHS \ RHS_Y
    Zvec = LHS \ RHS_Z

    # compute bounds for inverse function
    INVERSE_LB = Float64[minimum(Tvec), minimum(Xvec), minimum(Yvec), minimum(Zvec)]
    INVERSE_UB = Float64[maximum(Tvec), maximum(Xvec), maximum(Yvec), maximum(Zvec)]

    # construct interpolation functions for each coordinate
    interp_ranges = (LinRange(t0, tf, Nt), LinRange(x0, xf, Nx), LinRange(y0, yf, Ny), LinRange(z0, zf, Nz))
    Tf = cubic_spline_interpolation(interp_ranges, reshape(Tvec, (Nt, Nx, Ny, Nz)), extrapolation_bc = Line())
    Xf = cubic_spline_interpolation(interp_ranges, reshape(Xvec, (Nt, Nx, Ny, Nz)), extrapolation_bc = Line())
    Yf = cubic_spline_interpolation(interp_ranges, reshape(Yvec, (Nt, Nx, Ny, Nz)), extrapolation_bc = Line())
    Zf = cubic_spline_interpolation(interp_ranges, reshape(Zvec, (Nt, Nx, Ny, Nz)), extrapolation_bc = Line())

    # construct interpolation function for the coordinate transformation itself
    ORIGINAL_TO_HARMONIC(x) = [Tf(x...), Xf(x...), Yf(x...), Zf(x...)]

    # define const function for the inverse problem 
    cost_function = OptimizationFunction(AutoForwardDiff()) do x_original, x_harmonic
        norm(ORIGINAL_TO_HARMONIC(x_original) - x_harmonic)
    end

    # define inverse transformation
    HARMONIC_TO_ORIGINAL(x) = begin
        x0 = (DOMAIN_LB + DOMAIN_UB) / 2.0
        prob = OptimizationProblem(cost_function, x0, x, lb=DOMAIN_LB, ub=DOMAIN_UB)
        return solve(prob, OptimizationOptimJL.NelderMead()).u
    end

    # define metric in new coordinate system
    METRIC_HARMONIC(x) = begin
            
        # perform inverse transform
        x_original = HARMONIC_TO_ORIGINAL(x)

        # compute jacobian matrix
        J = ForwardDiff.jacobian(ORIGINAL_TO_HARMONIC, x_original)

        return transpose(J) * ST.METRIC(x_original) * J
    end

    GAMMA(r) = begin

        # construct finite difference operator 
        der = central_fdm(3,1, factor=1e6)

        # compute gradient of the metric components
        dg = zeros(4, 4, 4)
        dg[1,:,:] = der(t -> METRIC_HARMONIC([t; r[2:end]]), x[1])
        dg[2,:,:] = der(x -> METRIC_HARMONIC([r[1]; x; r[3:4]]), r[2])
        dg[3,:,:] = der(y -> METRIC_HARMONIC([r[1:2]; y; r[4]]), r[3])
        dg[4,:,:] = der(x -> METRIC_HARMONIC([r[1:3]; z]), r[4])

        # compute christoffel symbols of the first kind 
        G1 = zeros(4, 4, 4)
        for i=1:4
            G1[i,:,:] = 0.5*(dg[:,i,:] + transpose(dg[:,i,:]) - dg[i,:,:])
        end

        # raise the first index
        ginv = inv(METRIC_HARMONIC(r))
        G2 = zeros(4, 4, 4)
        for i=1:4
            for j=1:4
                G2[:,i,j] = ginv * G1[:,i,j]
            end
        end

        return G2
    end

    # force functions involving the inverse map to precompile (optional)
    if PRECOMPILE_INVERSE
        x0 = (DOMAIN_LB + DOMAIN_UB) / 2.0
        HARMONIC_TO_ORIGINAL(x0)
        METRIC_HARMONIC(x0)
        GAMMA(x0)
    end


    return ORIGINAL_TO_HARMONIC, HARMONIC_TO_ORIGINAL, Spacetime(METRIC_HARMONIC, GAMMA, DOMAIN_LB=INVERSE_LB, DOMAIN_UB=INVERSE_UB)
end