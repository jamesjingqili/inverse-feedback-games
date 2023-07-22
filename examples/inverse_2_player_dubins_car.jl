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
nx, nu, ΔT, game_horizon = 9, 4, 0.1, 40

# setup the dynamics
struct DoubleUnicycle <: ControlSystem{ΔT,nx,nu} end
# state: (px, py, phi, v)
dx(cs::DoubleUnicycle, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2], 
                                    x[8]cos(x[7]), x[8]sin(x[7]), u[3], u[4],0)
dynamics = DoubleUnicycle()
costs = (FunctionPlayerCost((g, x, u, t) -> (  8*(x[5]-x[9])^2  +  2*(u[1]^2 + u[2]^2) )),
        FunctionPlayerCost((g, x, u, t) -> (  4*(x[5]-x[1])^2  +  4*(x[8]-1)^2 + 2*(u[3]^2 + u[4]^2) ))   )

# indices of inputs that each player controls
player_inputs = (SVector(1,2), SVector(3,4))
# the horizon of the game
g = GeneralGame(game_horizon, player_inputs, dynamics, costs)

# get a solver, choose initial conditions and solve (in about 9 ms with AD)
solver1 = iLQSolver(g, max_scale_backtrack=10, max_elwise_diff_step=Inf, equilibrium_type="OLNE_costate")
x0 = SVector(0, 0.5, pi/2, 1,       1, 0, pi/2, 1,0.0)

c1, expert_traj1, strategies1 = solve(g, solver1, x0)

solver2 = iLQSolver(g, max_scale_backtrack=5, max_elwise_diff_step=Inf, equilibrium_type="FBNE")
c2, expert_traj2, strategies2 = solve(g, solver2, x0)

function parameterized_cost(θ::Vector)
    costs = (FunctionPlayerCost((g, x, u, t) -> (  θ[1]*(x[5]-x[9])^2  +  θ[2]*x[1]^2 +  2*(u[1]^2 + u[2]^2) )),
            FunctionPlayerCost((g, x, u, t) -> (  θ[3]*(x[5]-x[1])^2  +  θ[4]*(x[8]-1)^2 + 2*(u[3]^2 + u[4]^2) ))   )
    return costs
end

θ_true = [8, 0, 4, 4]


# ------------------------------------------------------------------------------------------------------------------------------------------
"Inverse Game Porblem: infer cost from noisy incomplete, partially observed expert demo"
GD_iter_num = 60
num_clean_traj = 1
noise_level_list = [0.0]
num_noise_level = length(noise_level_list)
num_obs = 1
games = []
x0_set = [x0 for ii in 1:num_clean_traj]

"modify the clean expert demo to noisy expert demo:"
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

"define initial solution of θ:"
θ₀ = 4*ones(4);





"define the partial observation and incomplete expert demonstrations:"
obs_time_list= [1,2,3,4,5,6,7,8,9,10,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
obs_state_list = [1,2,3,5,6,7]
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
        if random_init_x0 == true
            init_x0 = [noisy_expert_traj_list[ii][jj][kk].x[1]-[0,0,0,noisy_expert_traj_list[ii][jj][kk].x[1][4],0,0,0,noisy_expert_traj_list[ii][jj][kk].x[1][8],0] + (0.8*ones(9)+0.4*rand(9)).*[0,0,0,1,0,0,0,1,0]  for kk in 1:tmp_num_obs]
        else
            init_x0 = [noisy_expert_traj_list[ii][jj][kk].x[1]  for kk in 1:tmp_num_obs]
        end
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
    end
end

