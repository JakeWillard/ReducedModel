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
    GAMMA :: Function
    METRIC :: Function

    # Physical parameters
    ASPECT_RATIO :: Float64
    UPSTREAM_B :: Float64
    DIFFUSION_VECTOR :: Vector{Float64}

    Precomputables(; GAMMA=(t, y) -> zeros(2, 2), METRIC=(y, t) -> diagm([-1.0, 1.0]), Y_NUMBER=200, Y_SIZE=1.0, CFL=0.2, N_TRACE=10, ASPECT_RATIO=0.2, UPSTREAM_B=10.0, HYPERVISCOSITY=1.0, HYPERVISCOSITY_EXPONENT=1) = begin
        
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

        new(Y_NUMBER, Y_SPACING, Y_POINTS, T_SPACING, N_TRACE, Dy, Dyy, ASPECT_RATIO, UPSTREAM_B, DIFFUSION_VECTOR)
    end
end