# using Infiltrator
function solve_lq_game_FBNE_KKT!(strategies, g::LQGame, x0)
    # extract control and input dimensions
    nx, nu, m, T = n_states(g), n_controls(g), length(uindex(g)[1]), horizon(g)
    # m is the input size of agent i, and T is the horizon.
    num_player = n_players(g) # number of player
    M_size = nu+nx+nx*num_player# + (num_player-1)*nu # size of the M matrix for each time instant, will be used to define KKT matrix
    new_M_size = nu+nx+nx*num_player + (num_player-1)*nu
    # initialize some intermidiate variables in KKT conditions
    M_next, N_next, n_next = zeros(M_size, M_size), zeros(M_size, nx), zeros(M_size)
    Mₜ,     Nₜ,      nₜ     = zeros(M_size, M_size), zeros(M_size, nx), zeros(M_size)
    λ = zeros(T*nx*num_player)
    record_old_Mₜ_size = M_size
    K, k = zeros(M_size, nx), zeros(M_size)
    for t in T:-1:1 # work in backwards to construct the KKT constraint matrix
        dyn, cost = dynamics(g)[t], player_costs(g)[t]
        # convenience shorthands for the relevant quantities
        A, B = dyn.A, dyn.B
        Âₜ, B̂ₜ = zeros(nx*num_player, nx*num_player), zeros(nx*num_player, nu)
        Rₜ, Qₜ = zeros(nu, nu), zeros(nx*num_player, nx)
        rₜ, qₜ = zeros(nu), zeros(nx*num_player)
        B̃ₜ, R̃ₜ = zeros(num_player*nx, (num_player-1)*nu), zeros((num_player-1)*nu, nu)
        Πₜ = zeros((num_player-1)*nu, num_player*nx)
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
            nₜ[1:nu], nₜ[M_size-nx*num_player+1:M_size] = rₜ, qₜ
            
            M_next, N_next, n_next = Mₜ, Nₜ, nₜ
            K, k = -Mₜ\Nₜ, -Mₜ\nₜ
            # inv_Mₜ = inv(Mₜ)
            # K, k = -inv_Mₜ*Nₜ, -inv_Mₜ*nₜ
            strategies[t] = AffineStrategy(SMatrix{nu, nx}(-K[1:nu,:]), SVector{nu}(-k[1:nu]))     
        else
            for (ii, udxᵢ) in enumerate(uindex(g))
                Âₜ[(ii-1)*nx+1:ii*nx, (ii-1)*nx+1:ii*nx] = A
                B̂ₜ[(ii-1)*nx+1:ii*nx, (ii-1)*m+1:ii*m] = B[:,udxᵢ]
                Qₜ[(ii-1)*nx+1:ii*nx,:], Rₜ[udxᵢ,:] = cost[ii].Q, cost[ii].R[udxᵢ,:]
                qₜ[(ii-1)*nx+1:ii*nx], rₜ[udxᵢ] = cost[ii].l, cost[ii].r[udxᵢ]
                udxᵢ_complement = setdiff(1:1:nu, udxᵢ)
                
                B̃ₜ[(ii-1)*nx+1:ii*nx, (ii-1)*(num_player-1)*m+1:ii*(num_player-1)*m] = B[:,udxᵢ_complement] # 
                R̃ₜ[(ii-1)*(num_player-1)*m+1:ii*(num_player-1)*m, :] = cost[ii].R[udxᵢ_complement,:] # 
                Πₜ[(ii-1)*(num_player-1)*m+1:ii*(num_player-1)*m, (ii-1)*nx+1:ii*nx] = K[udxᵢ_complement, :] #
            end
            Mₜ, Nₜ, nₜ = zeros(new_M_size+record_old_Mₜ_size, new_M_size+record_old_Mₜ_size), zeros(new_M_size+record_old_Mₜ_size, nx), zeros(new_M_size+record_old_Mₜ_size)

            Mₜ[end-record_old_Mₜ_size+1:end, end-record_old_Mₜ_size+1:end] = M_next
            Mₜ[end-record_old_Mₜ_size+1:end, nu+nx*num_player+(num_player-1)*nu+1:nu+nx*num_player+(num_player-1)*nu+nx] = N_next
            nₜ[end-record_old_Mₜ_size+1:end] = n_next

            Nₜ[nu+1:nu+nx,:] = -A
            Mₜ[new_M_size-(num_player-1)*nu-nx*num_player+1:new_M_size-(num_player-1)*nu, new_M_size+nu+1:new_M_size+nu+nx*num_player] = transpose(Âₜ) #
            Mₜ[new_M_size-(num_player-1)*nu+1:new_M_size, new_M_size+nu+1:new_M_size+nu+nx*num_player] = transpose(B̃ₜ) #
            Mₜ[new_M_size-(num_player-1)*nu+1:new_M_size, nu+num_player*nx+1:nu+num_player*nx+(num_player-1)*nu] = -I((num_player-1)*nu) #
            Mₜ[new_M_size-(num_player-1)*nu+1:new_M_size, new_M_size+1:new_M_size+nu] = R̃ₜ #
            Mₜ[1:nu, 1:nu] = Rₜ
            Mₜ[1:nu, nu+1:nu+nx*num_player] = transpose(B̂ₜ)
            Mₜ[nu+1:nu+nx, 1:nu] = -B
            Mₜ[nu+1:nu+nx, new_M_size-nx+1:new_M_size] = I(nx)
            Mₜ[nu+nx+1:nu+nx+num_player*nx, nu+1:nu+nx*num_player] = -I(nx*num_player)
            Mₜ[nu+nx+1:nu+nx+num_player*nx, nu+nx*num_player+1:nu+nx*num_player+(num_player-1)*nu] = transpose(Πₜ) #
            Mₜ[nu+nx+1:nu+nx+num_player*nx, new_M_size-nx+1:new_M_size] = Qₜ
            nₜ[1:nu], nₜ[nu+nx+1:nu+nx+num_player*nx] = rₜ, qₜ
            
            M_next, N_next, n_next = Mₜ, Nₜ, nₜ
            K, k = -Mₜ\Nₜ, -Mₜ\nₜ
            # inv_Mₜ = inv(Mₜ)
            # K, k = -inv_Mₜ*Nₜ, -inv_Mₜ*nₜ
            record_old_Mₜ_size += new_M_size
            strategies[t] = AffineStrategy(SMatrix{nu, nx}(-K[1:nu,:]), SVector{nu}(-k[1:nu]))
            
        end
    end
    λ_solution = K*x0+k
    for t in 1:1:T
        λ[(t-1)*nx*num_player+1:t*nx*num_player] = λ_solution[ (t-1)*new_M_size+nu+1:(t-1)*new_M_size+nu+nx*num_player ]
    end
    return λ
end