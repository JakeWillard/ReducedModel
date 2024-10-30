struct Precomputables

    # Grid parameters
    Y_NUMBER :: Int64
    Y_SPACING :: Float64
    Y_POINTS :: Vector{Float64}
    T_SPACING :: Float64
    N_TRACE :: Int64

    # Derivative operators
    Dy :: SparseMatrixCSC
    Dyy :: SparseMatrixCSC

    # Spacetime 
    ST :: Spacetime

    # Physical parameters
    ASPECT_RATIO :: Float64
    UPSTREAM_B :: Float64
    DIFFUSION_VECTOR :: Vector{Float64}

    Precomputables(; ST=FlatCartesian(), Y_NUMBER=200, Y_SIZE=1.0, CFL=0.2, N_TRACE=10, ASPECT_RATIO=0.2, UPSTREAM_B=10.0, HYPERVISCOSITY=1.0, HYPERVISCOSITY_EXPONENT=1) = begin
        
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
        DIFFUSION_VECTOR = exp.(-ks.^(2*HYPERVISCOSITY_EXPONENT) * T_SPACING * HYPERVISCOSITY)

        new(Y_NUMBER, Y_SPACING, Y_POINTS, T_SPACING, N_TRACE, Dy, Dyy, ST, ASPECT_RATIO, UPSTREAM_B, DIFFUSION_VECTOR)
    end
end



function Precomputables(savefile; kwargs...)

    Precomputables(ST=Spacetime(savefile), kwargs...)
end



function SchwarzchildMidplane(r0, dt, dr, dtheta, dphi; savefile=nothing, GRID_DIMENSIONS=Int64[50, 5, 50, 5], PRECOMPILE_INVERSE=false, kwargs...)

    original_spacetime = Spacetime(x -> zeros(2, 2)) do x
        t, r, theta, phi = x
        alpha2 = 1 - 1/r
        return diagm([-alpha2, 1/alpha2, r^2, r^2*sin(theta)^2])
    end

    



end