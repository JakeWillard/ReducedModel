

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


struct HCoordTransform

    SOLUTION_BOUNDS :: Vector{Float64}
    INVERSE_LB :: Vector{Float64}
    INVERSE_UB :: Vector{Float64}

    ORIGINAL_TO_HARMONIC :: Function
    HARMONIC_TO_ORIGINAL :: Function
    G_ORIGINAL :: Function
    G_HARMONIC :: Function

    HCoordTransform(METRIC::Function, SOLUTION_BOUNDS; GRID_DIMENSIONS=Int64[50, 5, 50, 5], PRECOMPILE_INVERSE=false) = begin
        
        # define function which computes the gradient of the volume element per volume element
        vol(x) = sqrt(-det(METRIC(x)))
        grad_vol(x) = ForwardDiff.gradient(vol, x) / vol(x)

        # label grid dimensions and total number of gridpoints
        Nt, Nx, Ny, Nz = GRID_DIMENSIONS
        Ng = Nt*Nx*Ny*Nz

        # label the solution bounds
        ht, hx, hy, hz = SOLUTION_BOUNDS

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
        tvec = vcat([LinRange(0, ht, Nt) for _=1:Nx*Ny*Nz]...)
        xvec = vcat([vcat([ones(Nt)*xi for xi in LinRange(0, hx, Nx)]...) for _=1:Ny*Nz]...)
        yvec = vcat([vcat([ones(Nt*Nx)*yi for yi in LinRange(0, hy, Ny)]...) for _=1:Nz]...)
        zvec = vcat([ones(Nt*Nx*Ny)*zi for zi in LinRange(0, hz, Nz)]...)

        # initialize vectors for the diagonals of operators Gt, Gx, Gy, and Gz
        Gt_vals = zeros(Ng)
        Gx_vals = zeros(Ng)
        Gy_vals = zeros(Ng)
        Gz_vals = zeros(Ng)

        # these operators represent multiplication by a component of grad_vol at each point on the grid
        for i=1:Ng

            position = [tvec[i], xvec[i], yvec[i], zvec[i]]
            dvol = grad_vol(position)
            del_vol = inv(METRIC(position)) * dvol
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
        interp_ranges = (LinRange(0, ht, Nt), LinRange(0, hx, Nx), LinRange(0, hy, Ny), LinRange(0, hz, Nz))
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
            prob = OptimizationProblem(cost_function, SOLUTION_BOUNDS/2.0, x, lb=zeros(4), ub=SOLUTION_BOUNDS)
            return solve(prob, OptimizationOptimJL.NelderMead()).u
        end

        # define metric tensor in new coordinate system as function of new coordinates 
        G_HARMONIC(x) = begin
            
            # perform inverse transform
            x_original = HARMONIC_TO_ORIGINAL(x)

            # compute jacobian matrix
            J = ForwardDiff.jacobian(ORIGINAL_TO_HARMONIC, x_original)

            return transpose(J) * METRIC(x_original) * J
        end

        # force functions involving the inverse map to precompile (optional)
        if PRECOMPILE_INVERSE
            HARMONIC_TO_ORIGINAL(SOLUTION_BOUNDS/2.0)
            G_HARMONIC(SOLUTION_BOUNDS/2.0)
        end

        new(SOLUTION_BOUNDS, INVERSE_LB, INVERSE_UB, ORIGINAL_TO_HARMONIC, HARMONIC_TO_ORIGINAL, METRIC, G_HARMONIC)
    end
end



function christoffel_symbols_in_plane(Y_POINTS, T_POINTS, HT::HCoordTransform)

    Nt = length(T_POINTS)
    Ny = length(Y_POINTS)
    GAMMA = zeros(Ny, Nt, 2, 2)

    prog = Progress(Nt*Ny)
    der = central_fdm(3,1, factor=1e6)
    for j in eachindex(T_POINTS)
        for i in eachindex(Y_POINTS)

            # compute gradient of the metric components
            dg = zeros(4, 4, 4)
            dg[1,:,:] = der(t -> HT.G_HARMONIC([t, 0.0, Y_POINTS[i], 0.0]), T_POINTS[j])
            dg[2,:,:] = der(x -> HT.G_HARMONIC([T_POINTS[j], x, Y_POINTS[i], 0.0]), 0.0)
            dg[3,:,:] = der(y -> HT.G_HARMONIC([T_POINTS[j], 0.0, y, 0.0]), Y_POINTS[i])
            dg[4,:,:] = der(z -> HT.G_HARMONIC([T_POINTS[j], 0.0, Y_POINTS[i], z]), 0.0)

            # compute christoffel symbols of the first kind 
            gamma_first_kind = zeros(4, 4, 4)
            for k=1:4
                gamma_first_kind[k,:,:] = 0.5*(dg[:,k,:] + transpose(dg[:,k,:]) - dg[k,:,:])
            end

            # compute only the relevant christoffel symbols of the second kind 
            ginv = inv(HT.G_HARMONIC([T_POINTS[j], 0.0, Y_POINTS[i], 0.0]))
            GAMMA[i,j,1,1] = dot(ginv[2,:], gamma_first_kind[:,1,1])
            GAMMA[i,j,1,2] = dot(ginv[2,:], gamma_first_kind[:,1,2])
            GAMMA[i,j,2,2] = dot(ginv[2,:], gamma_first_kind[:,2,2])
            GAMMA[i,j,2,1] = GAMMA[i,j,1,2]

            next!(prog)
        end
    end

    # construct interpolation functions for each component
    Gtt = cubic_spline_interpolation((Y_POINTS, T_POINTS), GAMMA[:,:,1,1], extrapolation_bc = Line())
    Gty = cubic_spline_interpolation((Y_POINTS,T_POINTS), GAMMA[:,:,1,2], extrapolation_bc = Line())
    Gyy = cubic_spline_interpolation((Y_POINTS,T_POINTS), GAMMA[:,:,2,2], extrapolation_bc = Line())

    # define interpolation function for the matrix itself
    G(y, t) = [Gtt(y, t) Gty(y, t); Gty(y, t) Gyy(y, t)]
    return G
end




