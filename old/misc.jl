function simulate(psi_0, psi_1, Nt, ev::Evolver)

    u = zeros(ev.Ny, 3)
    u[:,1] = psi_0
    u[:,2] = psi_1
    psi_output = zeros(ev.Ny, Nt+2)
    psi_output[:,1:2] = u[:,1:2]

    for t=1:Nt
        evolve_non_boundary_points!(u, ev)
        set_inner_boundary_point!(u, ev)
        set_outer_boundary_point!(u, ev)
        
        psi_output[:,t+2] = u[:,3]
        u[:,1:2] = u[:,2:3]
    end

    return psi_output[:,3:end]
end



function simulate_from_static(psi0, Nt, ev::Evolver)

    return simulate(psi0, psi0, Nt, ev)
end


function simulate_from_end(psi_output, Nt, ev::Evolver)

    Ny = ev.Ny
    return simulate(psi_output[:,end-1], psi_output[:,end], Nt, ev)

end


function simulate_more(psi_output, Nt, ev::Evolver)

    Ny, Nold = size(psi_output)
    Nnew = Nt + Nold

    psi_new = zeros(Ny, Nnew)
    psi_new[:,1:Nold] = psi_output[:,:]
    psi_new[:,Nold+1:end] = simulate_from_end(psi_output, Nt, ev)

    return psi_new
end




