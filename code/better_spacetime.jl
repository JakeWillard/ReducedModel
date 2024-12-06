

struct Spacetime

    METRIC :: Function
    METRIC_INVERSE :: Function 
    VOLUME_ELEMENT :: Function 
    GAMMA :: Function
    DOMAIN_LB :: Vector{Float64}
    DOMAIN_UB :: Vector{Float64}

    Spacetime4D(args...; DOMAIN_LB=-[Inf, Inf, Inf, Inf], DOMAIN_UB=[Inf, Inf, Inf, Inf]) = begin

        new(args..., DOMAIN_LB, DOMAIN_UB)
    end
end


function Spacetime(METRIC::Function, GAMMA::Function, x, z; kwargs...)

    # 

end


