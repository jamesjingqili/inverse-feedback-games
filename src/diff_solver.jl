module Differentiable_Solvers

"----------------------- differentiable forward game solver -----------------------"

import StaticArrays
import ForwardDiff
using NamedTupleTools: @namedtuple
using iLQGames:
    LQGame,
    LTVSystem,
    AffineStrategy,
    QuadraticPlayerCost,
    SystemTrajectory,
    uindex,
    dynamics,
    player_costs,
    control_input,
    next_x,
    n_players,
    n_states,
    n_controls,
    time_disc2cont,
    linearize_discrete,
    regularize,
    LinearizationStyle
using Infiltrator
using LinearAlgebra
using StaticArrays

"A type relaxed version of solve_lq_game! without side effects"
function solve_lq_game_FBNE(g::LQGame)
    costs = player_costs(g)
    h = length(costs)
    uids = uindex(g)

    # initializting the optimal cost to go representation for DP
    # quadratic cost to go
    cost_to_go = map(c -> (ζ = c.l, Z = c.Q), last(costs))
    nx = first(size(dynamics(g)[1].A))
    full_xrange = StaticArrays.SVector{nx}(1:nx)
    
    # working backwards in time to solve the dynamic program
    map(h:-1:1) do kk 
        dyn = dynamics(g)[kk]
        cost = costs[kk]
        # convenience shorthands for the relevant quantities
        A = dyn.A
        B = dyn.B

        S, Y = mapreduce((a, b) -> vcat.(a, b), uids, cost, cost_to_go) do uidᵢ, cᵢ, Cᵢ
            BᵢZᵢ = B[:, uidᵢ]' * Cᵢ.Z
            (
                cᵢ.R[uidᵢ, :] + BᵢZᵢ * B,                     # rows of S
                [(BᵢZᵢ * A) (B[:, uidᵢ]' * Cᵢ.ζ + cᵢ.r[uidᵢ])],
            ) # rows of Y
        end

        # solve for the gains `P` and feed forward terms `α` simulatiously
        P_and_α = S \ Y
        P = P_and_α[:, full_xrange]
        α = P_and_α[:, end]

        # compute F and β as intermediate resu,lt for estimating the cost to go
        F = A - B * P
        β = -B * α

        # update Z and ζ (cost to go representation for the next step backwards
        # in time)
        cost_to_go = map(cost, cost_to_go) do cᵢ, Cᵢ
            PRᵢ = P' * cᵢ.R
            (
                ζ = (F' * (Cᵢ.ζ + Cᵢ.Z * β) + cᵢ.l + PRᵢ * α - P' * cᵢ.r),
                Z = (F' * Cᵢ.Z * F + cᵢ.Q + PRᵢ * P),
            )
        end

        AffineStrategy(P, α)
    end |> reverse
end




"A type relaxed version of trajectory! without side effects"
function trajectory(x0, g, γ, op = zero(SystemTrajectory, g))
    xs, us = StaticArrays.SVector{n_states(g)}[], StaticArrays.SVector{n_controls(g)}[]
    reduce(zip(γ, op.x, op.u); init = x0) do xₖ, (γₖ, x̃ₖ, ũₖ)
        Δxₖ = xₖ - x̃ₖ
        uₖ = control_input(γₖ, Δxₖ, ũₖ)
        push!(xs, xₖ)
        push!(us, uₖ)
        next_x(dynamics(g), xₖ, uₖ, 0.0)
    end
    vectype = StaticArrays.SizedVector{length(γ)}
    SystemTrajectory{0.1}(vectype(xs), vectype(us), 0.0) # loses information
end



"A type relaxed version of lq_approximation! without side effects and gradient
optimzation"
function lq_approximation(g, op, solver)
    lqs = map(eachindex(op.x), op.x, op.u) do k, x, u
        # discrete linearization along the operating point
        t = time_disc2cont(op, k)
        # @infiltrate
        
        ldyn = linearize_discrete(dynamics(g), x, u, t)

        # quadratiation of the cost along the operating point
        qcost =
            map(player_costs(g)) do pc
                x_cost = x -> pc(g, x, u, t)
                u_cost = u -> pc(g, x, u, t)
                l = ForwardDiff.gradient(x_cost, x)
                Q = ForwardDiff.hessian(x_cost, x)
                r = ForwardDiff.gradient(u_cost, u)
                R = ForwardDiff.hessian(u_cost, u)
                c = QuadraticPlayerCost(l, Q, r, R)
                regularize(solver, c)
            end |> StaticArrays.SizedVector{n_players(g),QuadraticPlayerCost}
        @namedtuple(ldyn, qcost)
    end
    dyn = LTVSystem([lq.ldyn for lq in lqs])
    pcost = [lq.qcost for lq in lqs]
    LQGame(uindex(g), dyn, pcost)
end

end # module