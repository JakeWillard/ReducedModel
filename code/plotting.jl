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