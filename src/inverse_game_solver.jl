using Infiltrator
using LinearAlgebra
using iLQGames:
	SystemTrajectory
	GeneralGame
using iLQGames
using Distributions
include("diff_solver.jl")

"evaluate trajectory prediction loss: "
function inverse_game_loss(θ::Vector, g::GeneralGame, expert_traj::SystemTrajectory, x0::SVector, parameterized_cost, equilibrium_type)
	current_cost = parameterized_cost(θ) # modify the cost vector
	current_game = GeneralGame(g.h, g.uids, g.dyn, current_cost)
	solver = iLQSolver(current_game, max_scale_backtrack=10, max_elwise_diff_step=Inf,equilibrium_type=equilibrium_type)
	converged, trajectory, strategies = solve(current_game, solver, x0)
	
	if converged==true
		println("converged!")
	else
		println("not converge T_T")
	end

	return norm(trajectory.x - expert_traj.x)^2 + norm(trajectory.u - expert_traj.u)^2, trajectory, strategies
end

"get gradient: "
function inverse_game_gradient(current_loss::Float64, θ::Vector, g::GeneralGame, expert_traj::SystemTrajectory, 
								x0::SVector, parameterized_cost, equilibrium_type)
	num_parameters = length(θ)
	gradient = zeros(num_parameters)
	Δ = 0.001
	for ii in 1:num_parameters
		θ_new = copy(θ)
		θ_new[ii] += Δ
		new_loss, tmp_traj, tmp_strategy = inverse_game_loss(θ_new, g, expert_traj, x0, parameterized_cost, equilibrium_type)
		# @infiltrate
		gradient[ii] = (new_loss-current_loss)/Δ
	end
	return gradient
end
function inverse_game_gradient_for_debug(current_loss::Float64, θ::Vector, g::GeneralGame, expert_traj::SystemTrajectory, 
								x0::SVector, parameterized_cost, equilibrium_type)
	num_parameters = length(θ)
	gradient = zeros(num_parameters)
	Δ = 0.001
	for ii in 1:num_parameters
		θ_new = copy(θ)
		θ_new[ii] += Δ
		new_loss, tmp_traj, tmp_strategy = inverse_game_loss_for_debug(θ, θ_new, g, expert_traj, x0, parameterized_cost, equilibrium_type)
		@infiltrate
		gradient[ii] = (new_loss-current_loss)/Δ
	end
	return gradient
end


function inverse_game_loss_for_debug(old_θ,θ::Vector, g::GeneralGame, expert_traj::SystemTrajectory, x0::SVector, parameterized_cost, equilibrium_type)
	current_cost = parameterized_cost(old_θ) # modify the cost vector
	current_game = GeneralGame(g.h, g.uids, g.dyn, current_cost)
	solver = iLQSolver(current_game, max_scale_backtrack=10, max_elwise_diff_step=Inf,equilibrium_type=equilibrium_type)
	converged, trajectory, strategies = solve(current_game, solver, x0)

	new_game = GeneralGame(g.h, g.uids, g.dyn, parameterized_cost(θ))
	new_solver = iLQSolver(new_game, max_scale_backtrack=10, max_elwise_diff_step=Inf,equilibrium_type=equilibrium_type)
	lqg = Differentiable_Solvers.lq_approximation(new_game, trajectory, new_solver) # 
	# lqg = iLQGames.lq_approximation(new_solver, new_game, trajectory)
	new_trajectory = zero(SystemTrajectory, g)
	if converged==true
		println("converged!")
	else
		println("not converge T_T")
	end

    if equilibrium_type=="OLNE_KKT" || equilibrium_type=="OLNE_costate" || equilibrium_type=="OLNE"
        iLQGames.trajectory!(new_trajectory, g.dyn, Differentiable_Solvers.solve_lq_game_OLNE(lqg), trajectory, x0 )
    elseif equilibrium_type=="FBNE_KKT" || equilibrium_type=="FBNE_costate" || equilibrium_type=="FBNE"
        # new_trajectory = iLQGames.trajectory!(new_trajectory, g.dyn, Differentiable_Solvers.solve_lq_game_FBNE(lqg), trajectory, x0 )
        strategies = Differentiable_Solvers.solve_lq_game_FBNE(lqg)        
        # iLQGames.solve_lq_game_FBNE_with_costate!(strategies, lqg, x0)
        new_trajectory = Differentiable_Solvers.trajectory(x0, g, reverse(strategies), trajectory)
        # @infiltrate
        # iLQGames.trajectory!(new_trajectory, g.dyn, strategies, trajectory, x0)
        println("good too!")
    else
        @warn "equilibrium_type is wrong!"
    end
    loss_value = norm(new_trajectory.x - expert_traj.x)^2 + norm(new_trajectory.u - expert_traj.u)^2
    return ForwardDiff.value.(loss_value), new_trajectory, strategies
end





