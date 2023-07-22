# Main codes start here:

using iLQGames
import iLQGames: dx

# parametes: number of states, number of inputs, sampling time, horizon
nx, nu, ΔT, game_horizon = 4, 2, 0.1, 20

# setup the dynamics
struct Unicycle <: ControlSystem{ΔT,nx,nu} end
# state: (px, py, phi, v)
dx(cs::Unicycle, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2])
dynamics = Unicycle()

# player-1 wants the unicycle to stay close to the origin,
# player-2 wants to keep close to 1 m/s
costs = (FunctionPlayerCost((g, x, u, t) -> (1*x[1]^2 + x[2]^2 + u[1]^2)),
         FunctionPlayerCost((g, x, u, t) -> ((x[4] - 1)^2 + u[2]^2)))

# indices of inputs that each player controls
player_inputs = (SVector(1), SVector(2))
# the horizon of the game
g = GeneralGame(game_horizon, player_inputs, dynamics, costs)

# get a solver, choose initial conditions and solve (in about 9 ms with AD)
solver = iLQSolver(g, max_scale_backtrack=10, max_elwise_diff_step=Inf, equilibrium_type = "OLNE_costate")
x0 = SVector(1, 1, 0, 0.5)
converged, expert_traj, expert_strategies = solve(g, solver, x0)
position_indices = tuple(SVector(1,2))
plot_traj(expert_traj, position_indices, [:red, :green], player_inputs)


function parameterized_cost(θ::Vector)
	costs = (FunctionPlayerCost((g, x, u, t) -> (θ[1]*x[1]^2 + x[2]^2 + u[1]^2)),
         FunctionPlayerCost((g, x, u, t) -> ((x[4] - 1)^2 + u[2]^2)))
	return costs
end


max_GD_iteration_num = 6

θ = [2.0]
sol = [zeros(1) for iter in 1:max_GD_iteration_num+1]
sol[1] = θ
loss = zeros(max_GD_iteration_num)
gradient = [zeros(1) for iter in 1:max_GD_iteration_num]
for iter in 1:max_GD_iteration_num
	sol[iter+1], loss[iter], gradient[iter] = inverse_game_gradient_descent(sol[iter], g, expert_traj, x0, 20, parameterized_cost)
	if loss[iter]<0.1
		break
	end
end

