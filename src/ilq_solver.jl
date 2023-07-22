# using Infiltrator
@with_kw struct iLQSolver{TLM, TOM, TQM}
    equilibrium_type::String = "FBNE"
    "The regularization term for the state cost quadraticization."
    state_regularization::Float64 = 0.0
    "The regularization term for the control cost quadraticization."
    control_regularization::Float64 = 0.0
    "The initial scaling of the feed-forward term."
    α_scale_init::Float64 = 0.5
    "The geometric scaling of the feed-forward term per scaling step in
    backtrack scaling."
    α_scale_step::Float64 = 0.5
    "Iteration is aborted if this number is exceeded."
    max_n_iter::Int = 200
    "The maximum number of backtrackings per scaling step"
    max_scale_backtrack::Int = 20
    "The maximum elementwise difference bewteen operating points for
    convergence."
    max_elwise_diff_converged::Float64 = α_scale_init/10
    "The maximum elementwise difference bewteen operating points for per
    iteration step."
    max_elwise_diff_step::Float64 = 20 * max_elwise_diff_converged
    "Preallocated memory for lq approximations."
    _lq_mem::TLM
    "Preallocated memory for quadraticization results."
    _qcache_mem::TQM
    "Preallocated memory for operting points."
    _op_mem::TOM
end

qcache(solver::iLQSolver) = solver._qcache_mem

function regularize(solver::iLQSolver, c::QuadraticPlayerCost)
    return QuadraticPlayerCost(c.l, c.Q + I * solver.state_regularization,
                               c.r, c.R + I * solver.control_regularization)
end

function iLQSolver(g, args...; kwargs...)
    return iLQSolver(args...; kwargs...,
                     _lq_mem=lqgame_preprocess_alloc(g),
                     _qcache_mem=zero(QuadCache{n_states(g),n_controls(g)}),
                     _op_mem=zero(SystemTrajectory, g))
end

function has_converged(solver::iLQSolver, last_op::SystemTrajectory{h},
                       current_op::SystemTrajectory{h}) where {h}
    return are_close(current_op, last_op, solver.max_elwise_diff_converged)
end

function are_close(op1::SystemTrajectory, op2::SystemTrajectory,
                   max_elwise_diff::Float64)
    @assert horizon(op1) == horizon(op2)
    return all(infnorm(op1.x[k] - op2.x[k]) < max_elwise_diff for k in
               eachindex(op2.x))
end

# modifies the current strategy to stabilize the update
function scale!(current_strategy::SizedVector, current_op::SystemTrajectory,
                α_scale::Float64)
    map!(current_strategy, current_strategy) do el
        return AffineStrategy(el.P, el.α * α_scale) # for each element in the strategy, scale the affine term by 0.5
    end
end

function backtrack_scale!(current_strategy::SizedVector,
                          current_op::SystemTrajectory, last_op::SystemTrajectory,
                          cs::ControlSystem, solver::iLQSolver)
    for i in 1:solver.max_scale_backtrack
        # initially we do a large scaling. Afterwards, always half feed forward
        # term.
		sf = i == 1 ? solver.α_scale_init : solver.α_scale_step
        scale!(current_strategy, current_op, sf)
        # we compute the new trajectory but abort integration once we have
        # diverged more than solver.max_elwise_diff_step
        
        if trajectory!(current_op, cs, current_strategy, last_op,
                       first(last_op.x), solver.max_elwise_diff_step)
            return true
        end
    end
    # in this case, the result in next_op is not really meaningful because the
    # integration has not been finished, thus the `success` state needs to be
    # evaluated and handled by the caller.
    return false
end


function OL_KKT_residual(λ::Vector, current_op::SystemTrajectory, g::LQGame, x0::SVector) #1    
    # current_op is the trajectory under the strategy evaluated now
    # g is the linear quadratic approximation for the trajectory under the strategy evaluated now
    # extract control and input dimensions
    nx, nu, m, T = n_states(g), n_controls(g), length(uindex(g)[1]), horizon(g)
    # m is the input size of agent i, and T is the horizon.
    num_player = n_players(g) # number of player
    M_size = nu+nx+nx*num_player # size of the M matrix for each time instant, will be used to define KKT matrix
    # initialize some intermidiate variables in KKT conditions
    M_next, N_next, n_next = zeros(M_size, M_size), zeros(M_size, nx), zeros(M_size)
    Mₜ,     Nₜ,      nₜ     = zeros(M_size, M_size), zeros(M_size, nx), zeros(M_size)
    solution_vector = zeros(T*M_size)

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
            solution_vector[(t-1)*M_size+1:(t-1)*M_size+nu] = current_op.u[t]
            solution_vector[(t-1)*M_size+nu+1:(t-1)*M_size+nu+nx*num_player] = λ[(t-1)*nx*num_player+1:t*nx*num_player]
            solution_vector[(t-1)*M_size+nu+nx*num_player+1:t*M_size] = current_op.x[t]
        else
            for (ii, udxᵢ) in enumerate(uindex(g))
                Âₜ[(ii-1)*nx+1:ii*nx, (ii-1)*nx+1:ii*nx] = A
                B̂ₜ[(ii-1)*nx+1:ii*nx, (ii-1)*m+1:ii*m] = B[:,udxᵢ]
                Qₜ[(ii-1)*nx+1:ii*nx,:], Rₜ[udxᵢ,:] = cost[ii].Q, cost[ii].R[udxᵢ,:]
                qₜ[(ii-1)*nx+1:ii*nx], rₜ[udxᵢ] = cost[ii].l, cost[ii].r[udxᵢ]
            end
            Mₜ, Nₜ, nₜ = zeros((T-t+1)*M_size, (T-t+1)*M_size), zeros((T-t+1)*M_size, nx), zeros((T-t+1)M_size)
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
            solution_vector[(t-1)*M_size+1:(t-1)*M_size+nu] = current_op.u[t]
            solution_vector[(t-1)*M_size+nu+1:(t-1)*M_size+nu+nx*num_player] = λ[(t-1)*nx*num_player+1:t*nx*num_player]
            solution_vector[(t-1)*M_size+nu+nx*num_player+1:t*M_size] = current_op.x[t]    
        end
    end
    
    loss = norm(Mₜ*solution_vector + Nₜ*x0+nₜ, 2)
    return loss   
end

function OL_KKT_line_search!(last_KKT_residual, λ::Vector, current_strategy::SizedVector, last_strategy::SizedVector,
                          current_op::SystemTrajectory, last_op::SystemTrajectory,
                          cs::ControlSystem, solver::iLQSolver, g::GeneralGame, current_lqg_approx::LQGame, x0::SVector) #2
    # return the current_op as the new trajectory, the current_strategy as the γₖ + α(γ_next - γₖ), and the last_KKT_residual
    # question: given a line-searched operating point, how can we retrive the policy achieving that trajectory?
    α = 1.0
    Δ_strategy = current_strategy-last_strategy
    for iter in 1:solver.max_scale_backtrack
        trajectory!(current_op, cs, last_strategy + α*Δ_strategy, last_op, x0, solver.max_elwise_diff_step)
        lq_approximation!(current_lqg_approx, solver, g, current_op)
        current_loss = OL_KKT_residual(λ, current_op, current_lqg_approx, x0)
        # @infiltrate
        if current_loss < last_KKT_residual
            current_strategy = last_strategy + α*Δ_strategy
            last_KKT_residual = copy(current_loss)
            # println("KKT residual is ",last_KKT_residual)
            # println("Line Search finished!")
            return true, current_strategy, current_op, last_KKT_residual
            # println("α is ", α)
            break
        end
        α = α * 0.5
    end
    # println("Current α is ",α)
    # @warn "Line Search failed."
    return true, current_strategy, current_op, last_KKT_residual
end

function trajectory_KKT_line_search!(last_KKT_residual, λ::Vector, current_strategy::SizedVector, last_strategy::SizedVector,
                                    current_op::SystemTrajectory, last_op::SystemTrajectory,
                                    cs::ControlSystem, solver::iLQSolver, g::GeneralGame, current_lqg_approx::LQGame, x0::SVector)
    α = 1.0
    Δ_op = current_op - last_op
    for iter in 1:solver.max_scale_backtrack
        tmp_op = last_op+α*Δ_op
        lq_approximation!(current_lqg_approx, solver, g, tmp_op)
        current_loss = OL_KKT_residual(λ, tmp_op, current_lqg_approx, x0)
        if current_loss < last_KKT_residual
            last_KKT_residual = current_loss
            # println(last_KKT_residual)
            println("Line Search succeed!")
            current_op = last_op + α*Δ_op
            return true, current_strategy, current_op, last_KKT_residual
            # println("α is ", α)
            break
        end
        α = α * 0.5
    end
    # println("Current α is ",α)
    # @warn "Line Search failed."
    return true, current_strategy, current_op, last_KKT_residual
end


function solve(g::AbstractGame, solver::iLQSolver, args...)
    op0 = zero(SystemTrajectory, g)
    γ0 = zero(strategytype(g))
    return solve!(op0, γ0, g, solver, args...)
end

function solve(initial_strategy::StaticVector, g, args...)
    return solve!(zero(SystemTrajectory, g), copy(initial_strategy), g, args...)
end

function solve(initial_op::SystemTrajectory, initial_strategy::StaticVector, g,
               args...)
    return solve!(copy(initial_op), copy(initial_strategy), g, args...)
end

"Copy the `initial_op` to a new operting point instance using preallocated memory."
function copyop_prealloc!(solver::iLQSolver, initial_op::SystemTrajectory)
    om = solver._op_mem
    new_op = SystemTrajectory{samplingtime(om)}(om.x, om.u, initialtime(initial_op))
    return copyto!(new_op, initial_op)
end


"""

    $(TYPEDSIGNATURES)


Computes a solution solution to a (potentially non-linear and non-quadratic)
finite horizon game g.
"""
function solve!(initial_op::SystemTrajectory, initial_strategy::StaticVector,
                g::GeneralGame, solver::iLQSolver, x0::SVector,
                verbose::Bool=false)

    converged = false
    i_iter = 0
    last_op = copyop_prealloc!(solver, initial_op)

    current_op = initial_op
    current_strategy = initial_strategy
    lqg_approx = solver._lq_mem
    last_λ = 100*ones(horizon(g)*n_states(g)*length(uindex(g)))
    last_strategy = copy(initial_strategy)
    
    # 0. compute the operating point for the first run.
    # TODO -- we could probably allow to skip this in some warm-starting scenarios
    # @infiltrate
    trajectory!(current_op, dynamics(g), current_strategy, last_op, x0) # repeat the initial state, for the first run.
    
    # things going to be updated: current_strategy, current_op, lqg_approx, last_λ
    while !(converged || i_iter >= solver.max_n_iter)
        # sanity chech to make sure that we don't manipulate the wrong
        # object...
        @assert !(last_op === current_op) "current and last operating point
        refer to the *same* object."

        # 1. linearize dynamics and quadratisize costs to obtain an lq game
        lq_approximation!(lqg_approx, solver, g, current_op)
        last_KKT_residual = OL_KKT_residual(last_λ, current_op, lqg_approx, x0)
        # 2. solve the current lq version of the game
        if solver.equilibrium_type == "FBNE"
            solve_lq_game_FBNE!(current_strategy, lqg_approx)
        elseif solver.equilibrium_type == "FBNE_KKT"
            λ=solve_lq_game_FBNE_KKT!(current_strategy, lqg_approx, x0)
        elseif solver.equilibrium_type == "OLNE"
            solve_lq_game_OLNE!(current_strategy, lqg_approx)
        elseif solver.equilibrium_type == "OLNE_KKT"
            λ=solve_lq_game_OLNE_KKT!(current_strategy, lqg_approx, x0)
        elseif solver.equilibrium_type == "OLNE_costate"
            λ=solve_lq_game_OLNE_with_costate!(current_strategy, lqg_approx, x0)
        elseif solver.equilibrium_type == "FBNE_costate"
            λ=solve_lq_game_FBNE_with_costate!(current_strategy, lqg_approx, x0)
        elseif solver.equilibrium_type == "Stackelberg"
            solve_lq_game_Stackelberg!(current_strategy, lqg_approx)
        else
            @error "solver.equilibrium_type is wrong. Please check."
        end
        # @infiltrate
        # 3. do line search to stabilize the strategy selection and extract the
        # next operating point
        copyto!(last_op, current_op)
        
        # copyto!(current_strategy, last_strategy)
        # copyto!(lqg_approx, last_lqg_approx)
        # copyto!(last_strategy, current_strategy)
        # @infiltrate
        if solver.equilibrium_type=="FBNE_KKT"||solver.equilibrium_type=="OLNE_KKT"||solver.equilibrium_type=="OLNE_costate"||solver.equilibrium_type=="FBNE_costate"
            # success = backtrack_scale!(current_strategy, current_op, last_op, dynamics(g), solver)
            # @infiltrate
            success, current_strategy, current_op, last_KKT_residual = OL_KKT_line_search!(last_KKT_residual, λ, current_strategy, last_strategy, current_op, last_op,
                                        dynamics(g), solver, g, lqg_approx, x0)
            # success = OL_KKT_line_search!(last_KKT_residual, λ, current_strategy, last_strategy, current_op, last_op,
            #                             dynamics(g), solver, g, lqg_approx, x0)
            # println("KKT value:",last_KKT_residual)
            # @infiltrate
            trajectory!(current_op, dynamics(g), current_strategy, last_op, x0, solver.max_elwise_diff_step)
            last_λ=copy(λ)
            last_strategy = copy(current_strategy)
        else
            success = backtrack_scale!(current_strategy, current_op, last_op, dynamics(g), solver)# take the last_op and current_strategy to current_op. 
        end
        
        if(!success)
            # @infiltrate
            verbose && @warn "Could not stabilize solution. Or, line searched failed."
            # we immetiately return and state that the solution has not been
            # stabilized
            return false, current_op, current_strategy
        end
        # @infiltrate i_iter == solver.max_n_iter-1
        i_iter += 1
        converged = has_converged(solver, last_op, current_op)

    end

    # NOTE: for `converged == false` the result may not be meaningful. `converged`
    # has to be handled outside this function
    return converged, current_op, current_strategy
end
