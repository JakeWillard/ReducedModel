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
    VOLUME_ELEMENT(x) = sqrt(-det(METRIC(x)))

    return Spacetime(METRIC, METRIC_INVERSE, VOLUME_ELEMENT, GAMMA, DOMAIN_LB=DOMAIN_LB, DOMAIN_UB=DOMAIN_UB)
end


function Spacetime(savefile)

    # read in the data
    fid = h5open(savefile, "r")
    Y_POINTS = fid["Y_POINTS"][:]
    T_POINTS = fid["T_POINTS"][:]
    DOMAIN_LB = fid["DOMAIN_LB"][:]
    DOMAIN_UB = fid["DOMAIN_UB"][:]
    G_MATRIX = fid["METRIC"][:,:,:,:]
    Ginv_MATRIX = fid["METRIC_INVERSE"][:,:,:,:]
    GAMMA_MATRIX = fid["GAMMA"][:,:,:,:]
    close(fid)

    # ranges for interpolation
    ys = LinRange(Y_POINTS[1], Y_POINTS[end], length(Y_POINTS))
    ts = LinRange(T_POINTS[1], T_POINTS[end], length(T_POINTS))
    ranges = (ys, ts)

    # make the interpolation functions 
    Gtt = cubic_spline_interpolation(ranges, G_MATRIX[:,:,1,1])
    Gty = cubic_spline_interpolation(ranges, G_MATRIX[:,:,1,2])
    Gyy = cubic_spline_interpolation(ranges, G_MATRIX[:,:,2,2])
    G_func(x) = [Gtt(x[3], x[1]) Gty(x[3], x[1]); Gty(x[3], x[1]) Gyy(x[3], x[1])]
    vol(x) = sqrt(det(G_func(x)))

    Ginvtt = cubic_spline_interpolation(ranges, Ginv_MATRIX[:,:,1,1])
    Ginvty = cubic_spline_interpolation(ranges, Ginv_MATRIX[:,:,1,2])
    Ginvyy = cubic_spline_interpolation(ranges, Ginv_MATRIX[:,:,2,2])
    Ginv_func(x) = [Ginvtt(x[3], x[1]) Ginvty(x[3], x[1]); Ginvty(x[3], x[1]) Ginvyy(x[3], x[1])]

    GAMtt = cubic_spline_interpolation(ranges, GAMMA_MATRIX[:,:,1,1])
    GAMty = cubic_spline_interpolation(ranges, GAMMA_MATRIX[:,:,1,2])
    GAMyy = cubic_spline_interpolation(ranges, GAMMA_MATRIX[:,:,2,2])
    GAM_func(x) = [GAMtt(x[3], x[1]) GAMty(x[3], x[1]); GAMty(x[3], x[1]) GAMyy(x[3], x[1])]

    return Spacetime(G_func, Ginv_func, vol, GAM_func, DOMAIN_LB=DOMAIN_LB, DOMAIN_UB=DOMAIN_UB)
end


function FlatCartesian()

    METRIC(x) = diagm([-1.0, 1.0, 1.0, 1.0])
    GAMMA(x) = zeros(2, 2)

    return Spacetime(METRIC, GAMMA)
end


function spacetime_to_grid(ST::Spacetime, Y_POINTS, T_POINTS; xval=0.0, yval=0.0, savefile=nothing)

    DOMAIN_LB = [T_POINTS[1], Y_POINTS[1]]
    DOMAIN_UB = [T_POINTS[end], Y_POINTS[end]]

    Ny = length(Y_POINTS)
    Nt = length(T_POINTS)
    G_MATRIX = zeros(Ny, Nt, 2, 2)
    Ginv_MATRIX = zeros(Ny, Nt, 2, 2)
    GAMMA_MATRIX = zeros(Ny, Nt, 2, 2)

    for i=1:Ny
        for j=1:Nt
            position = [T_POINTS[j], xval, Y_POINTS[i], yval]
            g = ST.METRIC(position)
            ginv = inv(g)
            GAMMA_MATRIX[i,j,:,:] = ST.GAMMA(position)[3, [1,3],[1,3]]
            G_MATRIX[i,j,:,:] = g[[1,3], [1,3]]
            Ginv_MATRIX[i,j,:,:] = ginv[[1,3], [1,3]]
        end
    end

    # save this data (optional)
    if !isnothing(savefile)
        fid = h5open(savefile, "w")
        fid["METRIC"] = G_MATRIX[:,:,:,:]
        fid["METRIC_INVERSE"] = Ginv_MATRIX[:,:,:,:]
        fid["GAMMA"] = GAMMA_MATRIX[:,:,:,:]
        fid["Y_POINTS"] = Vector(Y_POINTS)[:]
        fid["T_POINTS"] = Vector(T_POINTS)[:]
        fid["DOMAIN_LB"] = DOMAIN_LB[:]
        fid["DOMAIN_UB"] = DOMAIN_UB[:]
        close(fid)
    end

    # ranges for interpolation (for if Y_POINTS and T_POINTS aren't LinRange objects)
    ys = LinRange(Y_POINTS[1], Y_POINTS[end], length(Y_POINTS))
    ts = LinRange(T_POINTS[1], T_POINTS[end], length(T_POINTS))
    ranges = (ys, ts)

    # interpolation functions for components 
    Gtt = cubic_spline_interpolation(ranges, G_MATRIX[:,:,1,1])
    Gty = cubic_spline_interpolation(ranges, G_MATRIX[:,:,1,2])
    Gyy = cubic_spline_interpolation(ranges, G_MATRIX[:,:,2,2])
    G_func(x) = [Gtt(x[3], x[1]) Gty(x[3], x[1]); Gty(x[3], x[1]) Gyy(x[3], x[1])]
    vol(x) = sqrt(det(G_func(x)))

    Ginvtt = cubic_spline_interpolation(ranges, Ginv_MATRIX[:,:,1,1])
    Ginvty = cubic_spline_interpolation(ranges, Ginv_MATRIX[:,:,1,2])
    Ginvyy = cubic_spline_interpolation(ranges, Ginv_MATRIX[:,:,2,2])
    Ginv_func(x) = [Ginvtt(x[3], x[1]) Ginvty(x[3], x[1]); Ginvty(x[3], x[1]) Ginvyy(x[3], x[1])]

    GAMtt = cubic_spline_interpolation(ranges, GAMMA_MATRIX[:,:,1,1])
    GAMty = cubic_spline_interpolation(ranges, GAMMA_MATRIX[:,:,1,2])
    GAMyy = cubic_spline_interpolation(ranges, GAMMA_MATRIX[:,:,2,2])
    GAM_func(x) = [GAMtt(x[3], x[1]) GAMty(x[3], x[1]); GAMty(x[3], x[1]) GAMyy(x[3], x[1])]

    return Spacetime(G_func, Ginv_func, vol, GAM_func, DOMAIN_LB=DOMAIN_LB, DOMAIN_UB=DOMAIN_UB)
end