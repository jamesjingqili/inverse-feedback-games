function solve_lq_game_OLNE_with_costate!(strategies, g::LQGame, x0)
    # extract control and input dimensions
    nx, nu, T, N = n_states(g), n_controls(g), horizon(g), n_players(g)
    # N is the number of players
    Mₜ = [player_costs(g)[T][ii].Q for ii in 1:N]
    mₜ = [player_costs(g)[T][ii].l for ii in 1:N]

    λ = zeros((T+1)*nx*N)
    x = [zeros(nx) for t in 1:T+1]
    x[1] = x0
    for kk in T:-1:1
        tmp_M = [zeros(nx, nx) for ii in 1:N]
        tmp_m = [zeros(nx) for ii in 1:N]
        Mₜ, mₜ = [tmp_M Mₜ], [tmp_m mₜ]
        dyn = dynamics(g)[kk]
        cost = player_costs(g)[kk]
        # convenience shorthands for the relevant quantities
        A, B = dyn.A, dyn.B
        Λₜ, ηₜ = I(nx), zeros(nx)
        
        P = zeros(nu, nx)
        α = zeros(nu)

        for (ii, udxᵢ) in enumerate(uindex(g))
            inv_Rₜ = inv(cost[ii].R[udxᵢ,udxᵢ])
            Λₜ +=  B[:, udxᵢ]*inv_Rₜ*B[:, udxᵢ]'*Mₜ[ii,2]
            ηₜ -=  B[:, udxᵢ]*inv_Rₜ*(B[:, udxᵢ]'*mₜ[ii,2] + cost[ii].r[udxᵢ])
        end
        for (ii, udxᵢ) in enumerate(uindex(g))
            inv_Λₜ = inv(Λₜ)
            inv_Rₜ = inv(cost[ii].R[udxᵢ,udxᵢ])
            P[udxᵢ,:] = - inv_Rₜ*B[:,udxᵢ]'*(Mₜ[ii,2]*inv_Λₜ*A)
            α[udxᵢ] = - inv_Rₜ*B[:,udxᵢ]'*(Mₜ[ii,2]*inv_Λₜ*ηₜ+mₜ[ii,2]) - inv_Rₜ*cost[ii].r[udxᵢ]
            mₜ[ii,1] = cost[ii].l + A'*(mₜ[ii,2] + Mₜ[ii,2]*inv_Λₜ*ηₜ)
            Mₜ[ii,1] = cost[ii].Q + A'*Mₜ[ii,2]*inv_Λₜ*A
        end
        strategies[kk] = AffineStrategy(SMatrix{nu,nx}(-P), SVector{nu}(-α)) 
    end
    for t in 1:1:T
        for ii in 1:1:N
            dyn = dynamics(g)[t]
            x[t+1] = dyn.A*x[t] + dyn.B*(-strategies[t].P*x[t]-strategies[t].α)
            # λ[(t-1)*nx*N+(ii-1)*nx+1 : (t-1)*nx*N+ii*nx] = dyn.A'*(Mₜ[ii,t+1]*x[t+1]+mₜ[ii,t+1])
        end
    end
    for t in horizon(g):-1:1
        for ii in 1:N
            # @infiltrate
            λ[(t-1)*nx*N+(ii-1)*nx+1:(t-1)*nx*N+ii*nx] = transpose(dynamics(g)[t].A)*(λ[t*nx*N+(ii-1)*nx+1:t*nx*N+ii*nx] + (player_costs(g)[t][ii].Q*x[t+1] + player_costs(g)[t][ii].l))
        end
    end
    # @infiltrate
    return λ[1:T*nx*n_players(g)]
end