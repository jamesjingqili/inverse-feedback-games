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
nx, nu, ΔT, game_horizon = 4, 2, 0.1, 30

# setup the dynamics
struct DoubleUnicycle <: ControlSystem{ΔT,nx,nu} end
# state: (px, py, phi, v)
dx(cs::DoubleUnicycle, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2] )
dynamics = DoubleUnicycle()
costs = (FunctionPlayerCost((g, x, u, t) -> (  20*(x[1]-0)^2  +  2*(u[1]^2 + u[2]^2) )), ) # NOTE: the cost is a tuple of costs for each player

# indices of inputs that each player controls
player_inputs = (SVector(1,2),) # NOTE: the input is a tuple of inputs for each player
# the horizon of the game
g = GeneralGame(game_horizon, player_inputs, dynamics, costs)



# get a solver, choose initial conditions and solve (in about 9 ms with AD)
x0 = SVector(1, 0., pi/2, 1)


# if we want to solve an LQ game, then consider running the following 2 lines of codes:
solver = iLQSolver(g, max_scale_backtrack=5, max_elwise_diff_step=Inf, equilibrium_type="FBNE")
c, expert_traj, strategies = solve(g, solver, x0)

# define the cost function parameterized by θ ∈ R⁴:
function parameterized_cost(θ::Vector)
    costs = (FunctionPlayerCost((g, x, u, t) -> (  θ[1]*(x[1]-0)^2  +  2*(u[1]^2 + u[2]^2) )),)
    return costs
end

θ_true = [20]


# ------------------------------------------------------------------------------------------------------------------------------------------
"Inverse Game Porblem: infer cost from noisy incomplete, partially observed expert demo"
GD_iter_num = 20
num_clean_traj = 1 # number of clean expert demonstrations, without any noise
noise_level_list = [0.05] # the noise level of the expert demonstration, which is a list of floats, if we have two noise levels, then it is [0.02, 0.05]
num_noise_level = length(noise_level_list) # the number of noise levels
num_obs = 1 # the number of expert demonstrations
games = []
x0_set = [x0 for ii in 1:num_clean_traj]

"define initial solution of θ:"
θ₀ = 2*ones(1);










"All the following code does not need to be changed:"




"generate noisy expert demo:"
c_expert,expert_traj_list,expert_equi_list=generate_traj(g,x0_set,parameterized_cost,["FBNE","FBNE"])
noisy_expert_traj_list = [[[zero(SystemTrajectory, g) for kk in 1:num_obs] for jj in 1:num_noise_level] for ii in 1:num_clean_traj]
for ii in 1:num_clean_traj
    for jj in 1:num_noise_level
        tmp = generate_noisy_observation(nx, nu, g, expert_traj_list[ii], noise_level_list[jj], num_obs);
        for kk in 1:num_obs
            for t in 1:g.h
                noisy_expert_traj_list[ii][jj][kk].x[t] = tmp[kk].x[t];
                noisy_expert_traj_list[ii][jj][kk].u[t] = tmp[kk].u[t];
            end
        end
    end
end

"define the data structure to store the results:"
conv_table_list = [[[] for jj in 1:num_noise_level] for ii in 1:num_clean_traj];
sol_table_list = deepcopy(conv_table_list);
x0_table_list = deepcopy(conv_table_list);
loss_table_list = deepcopy(conv_table_list);
grad_table_list = deepcopy(conv_table_list);
equi_table_list = deepcopy(conv_table_list);
iter_table_list = deepcopy(conv_table_list);
comp_time_table_list = deepcopy(conv_table_list);

θ_list_list = deepcopy(conv_table_list);
index_list_list = deepcopy(conv_table_list);
optim_loss_list_list = deepcopy(conv_table_list);
state_prediction_error_list_list = deepcopy(conv_table_list);
generalization_error_list = deepcopy(conv_table_list);
ground_truth_loss_list = deepcopy(conv_table_list);
init_x0_list = deepcopy(conv_table_list);




"define the partial observation and incomplete expert demonstrations:"
obs_time_list= [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30] # we observe trajectories at these time steps
obs_state_list = [1,2,3,4] # we observe all states, but it is possible to observe only a subset of states
obs_control_list = 1:nu




"we now solve the inverse game problem:"
random_init_x0=false
if obs_state_list != 1:nx
    random_init_x0 = true
end
for ii in 1:num_clean_traj
    for jj in 1:num_noise_level
        if noise_level_list[jj] == 0.0
            tmp_num_obs = num_obs
        else
            tmp_num_obs = num_obs
        end
        init_x0 = [noisy_expert_traj_list[ii][jj][kk].x[1]  for kk in 1:tmp_num_obs]
        println("Now the $(jj)-th noise level")
        conv_table,x0_table,sol_table,loss_table,grad_table,equi_table,iter_table,ground_truth_loss = run_experiment_x0(g,θ₀,init_x0, 
                                                                                                noisy_expert_traj_list[ii][jj], parameterized_cost, GD_iter_num, 20, 1e-4, 
                                                                                                obs_time_list,obs_state_list, obs_control_list, "FBNE", 0.000000001, 
                                                                                                true, 10.0,expert_traj_list[ii],false,false,[],true,
                                                                                                10, 0.1, 0.1)
        θ_list, index_list, optim_loss_list = get_the_best_possible_reward_estimate_single(init_x0, ["FBNE","FBNE"], sol_table, loss_table, equi_table)

        push!(conv_table_list[ii][jj], conv_table)
        push!(x0_table_list[ii][jj], x0_table)
        push!(init_x0_list[ii][jj], init_x0)
        push!(sol_table_list[ii][jj], sol_table)
        push!(loss_table_list[ii][jj], loss_table)
        push!(grad_table_list[ii][jj], grad_table)
        push!(equi_table_list[ii][jj], equi_table)
        push!(iter_table_list[ii][jj], iter_table)
        push!(θ_list_list[ii][jj], θ_list)
        push!(index_list_list[ii][jj], index_list)
        push!(optim_loss_list_list[ii][jj], optim_loss_list)
        push!(ground_truth_loss_list[ii][jj], ground_truth_loss)
        "The first run may take a long time due to precompilation!"
    end
end




















# ------------------------------------------------------------------------------------------------------------------------------------------
"inferred θ:"
inferred_θ_at_each_iteration = sol_table_list[1][1][1][1]
println("inferred θ at each iteration: ", inferred_θ_at_each_iteration)


"ground truth θ:"
print("ground truth θ: ", θ_true)






"Plot the prediction loss:"
# plot prediction loss:
plot(1:GD_iter_num, [loss_table_list[1][1][1][1][ii] for ii in 1:GD_iter_num], label="prediction loss", xlabel="iteration", ylabel="loss", title="prediction loss")


"plot the noisy expert trajectory data, inferred trajectory and the clean expert trajectory:"
# plot the noisy trajectory:
scatter([noisy_expert_traj_list[1][1][1].x[t][1] for t in 1:game_horizon], [noisy_expert_traj_list[1][1][1].x[t][2] for t in 1:game_horizon], 
    label="noisy expert data", xlabel="x", ylabel="y", title="inferred trajectories",
    aspect_ratio = :equal)
# compute the inferred trajectory:
inferred_parameter = sol_table_list[1][1][1][1][end]
inferred_cost = parameterized_cost(inferred_parameter)
inferred_solver = iLQSolver(g, max_scale_backtrack=5, max_elwise_diff_step=Inf, equilibrium_type="FBNE")
inferred_c, inferred_traj, inferred_strategies = solve(g, inferred_solver, x0)
# plot the inferred trajectory and the clearn expert trajectory:
plot!([inferred_traj.x[t][1] for t in 1:game_horizon], [inferred_traj.x[t][2] for t in 1:game_horizon],linestyle=:dash,lw=6, label="inferred trajectory")
plot!([expert_traj.x[t][1] for t in 1:game_horizon], [expert_traj.x[t][2] for t in 1:game_horizon], lw=3, label="clean expert trajectory")


