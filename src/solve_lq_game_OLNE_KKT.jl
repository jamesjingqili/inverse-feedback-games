using Infiltrator
function solve_lq_game_OLNE_KKT!(strategies, g::LQGame, x0)
    # extract control and input dimensions
    nx, nu, m, T = n_states(g), n_controls(g), length(uindex(g)[1]), horizon(g)
    # m is the input size of agent i, and T is the horizon.
    num_player = n_players(g) # number of player
    M_size = nu+nx+nx*num_player # size of the M matrix for each time instant, will be used to define KKT matrix
    # initialize some intermidiate variables in KKT conditions
    M_next, N_next, n_next = zeros(M_size, M_size), zeros(M_size, nx), zeros(M_size)
    Mₜ,     Nₜ,      nₜ     = zeros(M_size, M_size), zeros(M_size, nx), zeros(M_size)
    λ = zeros(T*nx*num_player)
    K, k = zeros(M_size, nx), zeros(M_size)
    for t in T:-1:1 # work in backwards to construct the KKT constraint matrix
        dyn, cost = dynamics(g)[t], player_costs(g)[t]
        # convenience shorthands for the relevant quantities
        A, B = dyn.A, dyn.B
        Âₜ, B̂ₜ = zeros(nx*num_player, nx*num_player), zeros(nx*num_player, nu)
        Rₜ, Qₜ = zeros(nu, nu), zeros(nx*num_player, nx)
        rₜ, qₜ = zeros(nu), zeros(nx*num_player)
        
        if t == T
            for (ii, udxᵢ) in enumerate(uindex(g))
                B̂ₜ[(ii-1)*nx+1:ii*nx, (ii-1)*m+1:ii*m] = B[:,udxᵢ]
                Qₜ[(ii-1)*nx+1:ii*nx,:], Rₜ[udxᵢ,:] = cost[ii].Q, cost[ii].R[udxᵢ,:]
                qₜ[(ii-1)*nx+1:ii*nx], rₜ[udxᵢ] = cost[ii].l, cost[ii].r[udxᵢ]
            end
            Nₜ[nu+1:nu+nx,:] = -A
            Mₜ[1:nu, 1:nu] = Rₜ
            Mₜ[1:nu, nu+1:nu+nx*num_player] = transpose(B̂ₜ)
            Mₜ[nu+1:nu+nx, 1:nu] = -B
            Mₜ[nu+1:nu+nx, M_size-nx+1:M_size] = I(nx)
            Mₜ[nu+nx+1:M_size, nu+1:nu+nx*num_player] = -I(nx*num_player)
            Mₜ[nu+nx+1:M_size, M_size-nx+1:M_size] = Qₜ
            nₜ[1:nu], nₜ[nu+nx+1:nu+nx+nx*num_player] = rₜ, qₜ
            
            M_next, N_next, n_next = Mₜ, Nₜ, nₜ
            # @infiltrate
            K, k = -Mₜ\Nₜ, -Mₜ\nₜ
            # inv_Mₜ = inv(Mₜ)
            # K, k = -inv_Mₜ*Nₜ, -inv_Mₜ*nₜ
            # QR_Mₜ = qr(Mₜ)
            # K, k = QR_Mₜ\(-Nₜ), QR_Mₜ\(-nₜ)
            strategies[t] = AffineStrategy(SMatrix{nu, nx}(-K[1:nu,:]), SVector{nu}(-k[1:nu]))
            # @infiltrate
        else
            for (ii, udxᵢ) in enumerate(uindex(g))
                Âₜ[(ii-1)*nx+1:ii*nx, (ii-1)*nx+1:ii*nx] = A
                B̂ₜ[(ii-1)*nx+1:ii*nx, (ii-1)*m+1:ii*m] = B[:,udxᵢ]
                Qₜ[(ii-1)*nx+1:ii*nx,:], Rₜ[udxᵢ,:] = cost[ii].Q, cost[ii].R[udxᵢ,:]
                qₜ[(ii-1)*nx+1:ii*nx], rₜ[udxᵢ] = cost[ii].l, cost[ii].r[udxᵢ]
            end
            Mₜ, Nₜ, nₜ = zeros((T-t+1)*M_size, (T-t+1)*M_size), zeros((T-t+1)*M_size, nx), zeros((T-t+1)M_size)
            # @infiltrate
            Mₜ[end-(T-t)*M_size+1:end, end-(T-t)*M_size+1:end] = M_next
            Mₜ[end-(T-t)*M_size+1:end, M_size-nx+1:M_size] = N_next
            nₜ[end-(T-t)*M_size+1:end] = n_next

            Nₜ[nu+1:nu+nx,:] = -A
            Mₜ[M_size-nx*num_player+1:M_size, M_size+nu+1:M_size+nu+nx*num_player] = transpose(Âₜ)
            Mₜ[1:nu, 1:nu] = Rₜ
            Mₜ[1:nu, nu+1:nu+nx*num_player] = transpose(B̂ₜ)
            Mₜ[nu+1:nu+nx, 1:nu] = -B
            Mₜ[nu+1:nu+nx, M_size-nx+1:M_size] = I(nx)
            Mₜ[nu+nx+1:M_size, nu+1:nu+nx*num_player] = -I(nx*num_player)
            Mₜ[nu+nx+1:M_size, M_size-nx+1:M_size] = Qₜ
            nₜ[1:nu], nₜ[nu+nx+1:nu+nx+nx*num_player] = rₜ, qₜ
            
            M_next, N_next, n_next = Mₜ, Nₜ, nₜ
            K, k = -Mₜ\Nₜ, -Mₜ\nₜ
            # inv_Mₜ = inv(Mₜ)
            # K, k = -inv_Mₜ*Nₜ, -inv_Mₜ*nₜ
            # QR_Mₜ = qr(Mₜ)
            # K, k = QR_Mₜ\(-Nₜ), QR_Mₜ\(-nₜ)
            # @infiltrate
            strategies[t] = AffineStrategy(SMatrix{nu, nx}(-K[1:nu,:]), SVector{nu}(-k[1:nu]))
        end
    end   
    λ_solution = K*x0+k
    
    for t in 1:1:T

        λ[(t-1)*nx*num_player+1:t*nx*num_player] = λ_solution[(t-1)*M_size+nu+1:(t-1)*M_size+nu+nx*num_player]
    end
    # @infiltrate
    return λ
end