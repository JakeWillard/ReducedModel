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

    # Physical parameters
    ASPECT_RATIO :: Float64
    UPSTREAM_B :: Float64
    DIFFUSION_VECTOR :: Vector{Float64}

    Precomputables(; Y_NUMBER=200, Y_SIZE=1.0, CFL=0.2, N_TRACE=10, ASPECT_RATIO=0.2, UPSTREAM_B=10.0, HYPERVISCOSITY=1.0, HYPERVISCOSITY_EXPONENT=1) = begin
        
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


@userplot PlotSimulation

@recipe function f(ps::PlotSimulation)

    # the argument is a Solution
    sol = ps.args[1]

    # plot x and y values are Y_VALUES and T_VALUES 
    x = sol.Y_VALUES 
    y = sol.T_VALUES

    # compute contours that are reference null surfaces
    null_surfaces = zeros(length(y), length(x))
    for i in eachindex(x)
        for j in eachindex(y)
            null_surfaces[j,i] = x[i] + y[j]
        end
    end

    # setup the subplots
    grid := false
    link := :both
    layout := @layout [a b; c d; e]

    # plot F_SQUARED as heatmap
    @series begin
        title := "Proper Magnetic Energy"
        seriestype := :heatmap
        subplot := 1
        xaxis := :false
        x, y, transpose(sol.F_SQUARED)
    end

    # plot zeta as heatmap 
    @series begin
        title := "Plasma Rapidity"
        seriestype := :heatmap
        subplot := 2
        xaxis := :false
        yaxis := :false
        x, y, transpose(sol.PLASMA_RAPIDITY)
    end

    # plot psi contours
    @series begin
        title := "Flux Contours"
        seriestype := :contour
        legend := :false
        subplot := 3
        color := :black
        x, y, transpose(sol.FLUX_FUNCTION)
    end

    # plot reference null surfaces
    @series begin
        title := "Null Surfaces"
        seriestype := :contour 
        legend := :false
        subplot := 4
        color := :black 
        linestype := :dashed
        yaxis := :false
        x, y, null_surfaces
    end

    # plot reconnection rate vs time
    @series begin
        title := "Reconnection Rate"
        seriestype := :line
        legend := :false
        subplot := 5
        y, sol.RECONNECTION_RATE
    end


end





