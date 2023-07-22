using Infiltrator
function solve_lq_game_OLNE_B!(strategies, g::LQGame)
    # extract control and input dimensions
    nx, nu, T, N = n_states(g), n_controls(g), horizon(g), n_players(g)
    # m is the input size of agent i, and T is the horizon.
    # N is the number of players
    Mₜ = [player_costs(g)[T][ii].Q for ii in 1:N]
    mₜ = [player_costs(g)[T][ii].l for ii in 1:N]
    for kk in T:-1:1
        dyn = dynamics(g)[kk]
        cost = player_costs(g)[kk]
        # convenience shorthands for the relevant quantities
        A, B = dyn.A, dyn. B
        Λₜ, ηₜ = I(nx), zeros(nx)
        M_next, m_next = Mₜ, mₜ
        
        for (ii, udxᵢ) in enumerate(uindex(g))
            inv_Rₜ = inv(cost[ii].(R[udxᵢ,:]))
            Λₜ = Λₜ + B[:, udxᵢ]*inv_Rₜ*B[:, udxᵢ]'*M_next[ii]
            ηₜ = ηₜ - B[:, udxᵢ]*inv_Rₜ*(B[:, udxᵢ]'*m_next[ii] + cost[ii].r[udxᵢ])
        end
        for (ii, udxᵢ) in enumerate(uindex(g))
            pinv_Λₜ = pinv(Λₜ)
            mₜ[ii] = cost[ii].l + A'*(m_next[ii] + M_next[ii]*pinv_Λₜ*ηₖ)
            Mₜ[ii] = cost[ii].Q + A'*M_next[ii]*pinv_Λₜ*A
            
            inv_Rₜ = inv(cost[ii].(R[udxᵢ,:]))
            P = - inv_Rₜ*B[:,udxᵢ]'*(M_next[ii]*pinv_Λₜ*A)
            α = - inv_Rₜ*B[:,udxᵢ]'*(M_next[ii]*pinv_Λₜ*ηₜ+m_next[ii]) - inv_Rₜ*cost[ii].r[udxᵢ]
            strategies[kk] = AffineStrategy(P, α)    
        end        
    end
end