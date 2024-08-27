using iLQGames
import iLQGames: dx
using Plots
using ForwardDiff
using iLQGames:
    SystemTrajectory
using iLQGames:
    LinearSystem
using Infiltrator
using Optim
using LinearAlgebra
using Distributed
using Dates
using Statistics
include("../src/diff_solver.jl")
include("../src/experiment_utils.jl") # NOTICE!! Many functions are defined there.

"Forward Game Problem: generate expert demo"
# parametes: number of states, number of inputs, sampling time, horizon
nx, nu, ΔT, game_horizon = 5, 2, 0.1, 30

# setup the dynamics
struct DoubleUnicycle <: ControlSystem{ΔT,nx,nu} end
# state: (px, py, phi, v)
dx(cs::DoubleUnicycle, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2], 0 )
dynamics = DoubleUnicycle()
costs = (FunctionPlayerCost((g, x, u, t) -> (  20*(x[1]-x[5])^2  +  2*(u[1]^2 + u[2]^2) )), ) # NOTE: the cost is a tuple of costs for each player

# indices of inputs that each player controls
player_inputs = (SVector(1,2),) # NOTE: the input is a tuple of inputs for each player
# the horizon of the game
g = GeneralGame(game_horizon, player_inputs, dynamics, costs)



# get a solver, choose initial conditions and solve (in about 9 ms with AD)
x0 = SVector(1, 0., pi/2, 1, 0)


# if we want to solve an LQ game, then consider running the following 2 lines of codes:
solver = iLQSolver(g, max_scale_backtrack=5, max_elwise_diff_step=Inf, equilibrium_type="FBNE")
c, expert_traj, strategies = solve(g, solver, x0)





θ_true = [0]



# define the loss function that measuring the difference between the predicted trajectory under θ and the expert trajectory

function loss(
    θ, 
    nominal_game = g,
    nominal_solver = solver,
    expert_traj = expert_traj, 
    obs_time_list = 1:game_horizon, 
    obs_state_list = 1:nx, 
    obs_control_list = 1:nu,  
    x0=x0, # not include the extra cost params 
    debug_mode=false
    ) 
    extended_x0 = SVector{5}(vcat(x0, θ)) # where we concatenate the cost parameters (with dual) to the initial state
    purified_x0 = SVector{5}(vcat(x0, ForwardDiff.value.(θ))) # where we remove the dual number associated with the cost parameters

    # solve the nominal game under the current θ:
    nominal_converged, nominal_traj, nominal_strategies = solve(nominal_game, nominal_solver, purified_x0)
    # derive the LQ approximation along the nominal trajectory under the current θ:
    lqg = Differentiable_Solvers.lq_approximation(nominal_game, nominal_traj, nominal_solver)
    # what is the local LQ policy?
    local_strategies = Differentiable_Solvers.solve_lq_game_FBNE(lqg)
    # evaluate the predicted trajectory under the current θ:
    traj = Differentiable_Solvers.trajectory(
        extended_x0, 
        nominal_game, 
        local_strategies, 
        nominal_traj
    )
    expert_data = transpose(mapreduce(permutedims, vcat, Vector([Vector(expert_traj.u[t][obs_control_list]) for t in obs_time_list])))
    prediction = transpose(mapreduce(permutedims, vcat, Vector([Vector(traj.u[t][obs_control_list]) for t in obs_time_list])))
    loss_value = norm(expert_data - prediction)^2
    if debug_mode
        println("expert_data: ", expert_data)
        println("prediction: ", prediction)
        println("loss_value: ", loss_value)
        return loss_value, expert_data, prediction, nominal_converged, nominal_traj, traj, nominal_strategies, local_strategies
    else
        return loss_value
    end
end

# ground truth: 0, 2, 1, 2
loss_x0 = loss([0.013], g, solver, expert_traj, 1:10, [], 1:nu, x0[1:4])

loss_x0, tmp1, pred, cc, nominal_traj, traj, nominal_strategies,local_strategies  = loss([0], g, solver, expert_traj, 1:10, [], 1:nu, x0[1:4], true)

gradient_x0 = ForwardDiff.gradient(x -> loss(x, g, solver, expert_traj, 1:10, [], 1:nu, x0[1:4]), [0.013] )



# ------------------------------------------------------------------------------------------------------------------------------------------
"Now, we write down the gradient descent to infer the θ_true"

initial_θ_guess = [2.0]
step_size = 0.01
max_iter = 100
tolerance = 1e-3

θ = initial_θ_guess
loss_values = []
for i in 1:max_iter
    global θ
    loss_value = loss(θ, g, solver, expert_traj, 1:10, [], 1:nu, x0[1:4])
    push!(loss_values, loss_value)
    if loss_value < tolerance
        break
    end
    gradient = ForwardDiff.gradient(x -> loss(x, g, solver, expert_traj, 1:10, [], 1:nu, x0[1:4]), θ)
    θ = θ - step_size * gradient
    println("iter: ", i, " loss: ", loss_value, " θ: ", θ)
end







