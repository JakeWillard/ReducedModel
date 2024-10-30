struct Spacetime

    METRIC :: Function
    METRIC_INVERSE :: Function 
    VOLUME_ELEMENT :: Function 
    GAMMA :: Function
    DOMAIN_LB :: Vector{Float64}
    DOMAIN_UB :: Vector{Float64}

    Spacetime(args...; DOMAIN_LB=-[Inf, Inf, Inf, Inf], DOMAIN_UB=[Inf, Inf, Inf, Inf]) = begin

        new(args..., DOMAIN_LB, DOMAIN_UB)
    end
end


function Spacetime(METRIC::Function, GAMMA::Function; DOMAIN_LB=-[Inf, Inf, Inf, Inf], DOMAIN_UB=[Inf, Inf, Inf, Inf])

    METRIC_INVERSE(x) = inv(METRIC(x))
    VOLUME_ELEMENT(x) = sqrt(det(METRIC(x)))

    return Spacetime(METRIC, METRIC_INVERSE, VOLUME_ELEMENT, GAMMA, DOMAIN_LB=DOMAIN_LB, DOMAIN_UB=DOMAIN_UB)
end


function FlatCartesian()

    METRIC(x) = diagm([-1.0, 1.0, 1.0, 1.0])
    GAMMA(x) = zeros(2, 2)

    return Spacetime(METRIC, GAMMA)
end


function spacetime_to_grid(ST::Spacetime, Y_POINTS, T_POINTS; xval=0.0, yval=0.0)

    DOMAIN_LB = ST.DOMAIN_LB[[1, 3]]
    DOMAIN_UB = ST.DOMAIN_UB[[1, 3]]

    Ny = length(Y_POINTS)
    Nt = length(T_POINTS)
    G_MATRIX = zeros(Ny, Nt, 2, 2)
    Ginv_MATRIX = zeros(Ny, Nt, 2, 2)
    GAMMA_MATRIX = zeros(Ny, Nt, 2, 2)

    for i=1:Ny
        for j=1:Nt
            position = [t[j], xval, y[i], yval]
            g = ST.METRIC(position)
            ginv = inv(g)
            GAMMA_MATRIX[i,j,:,:] = ST.GAMMA(position)[3, [1,3], [1,3]]
            G_MATRIX[i,j,:,:] = g[[1,3], [1,3]]
            Ginv_MATRIX[i,j,:,:] = ginv[[1,3], [1,3]]
        end
    end

    # interpolation functions for components 
    Gtt = cubic_spline_interpolation((Y_POINTS, T_POINTS), G_MATRIX[:,:,1,1])
    Gty = cubic_spline_interpolation((Y_POINTS, T_POINTS), G_MATRIX[:,:,1,2])
    Gyy = cubic_spline_interpolation((Y_POINTS, T_POINTS), G_MATRIX[:,:,2,2])
    G_func(x) = [Gtt(x[3], x[1]) Gty(x[3], x[1]); Gty(x[3], x[1]) Gyy(x[3], x[1])]
    vol(x) = sqrt(det(G_func(x)))

    Ginvtt = cubic_spline_interpolation((Y_POINTS, T_POINTS), Ginv_MATRIX[:,:,1,1])
    Ginvty = cubic_spline_interpolation((Y_POINTS, T_POINTS), Ginv_MATRIX[:,:,1,2])
    Ginvyy = cubic_spline_interpolation((Y_POINTS, T_POINTS), Ginv_MATRIX[:,:,2,2])
    Ginv_func(x) = [Ginvtt(x[3], x[1]) Ginvty(x[3], x[1]); Ginvty(x[3], x[1]) Ginvyy(x[3], x[1])]

    GAMtt = cubic_spline_interpolation((Y_POINTS, T_POINTS), GAMMA_MATRIX[:,:,1,1])
    GAMty = cubic_spline_interpolation((Y_POINTS, T_POINTS), GAMMA_MATRIX[:,:,1,2])
    GAMyy = cubic_spline_interpolation((Y_POINTS, T_POINTS), GAMMA_MATRIX[:,:,2,2])
    GAM_func(x) = [GAMtt(x[3], x[1]) GAMty(x[3], x[1]); GAMty(x[3], x[1]) GAMyy(x[3], x[1])]

    return Spacetime(G_func, Ginv_func, vol, GAM_func, DOMAIN_LB=DOMAIN_LB, DOMAIN_UB=DOMAIN_UB)
end


