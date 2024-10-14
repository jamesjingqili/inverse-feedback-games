#=
    Imports and setup 
=#
using Pkg
using Infiltrator
using Revise 


Pkg.activate("/home/chrisstrong/MRI_Driving/inverse-feedback-games/")
using iLQGames
import iLQGames: dx
using Plots
using ForwardDiff
using iLQGames:
    SystemTrajectory
using iLQGames:
    LinearSystem
using Optim
using LinearAlgebra
using Distributed
using Dates
using Statistics

include("/home/chrisstrong/MRI_Driving/inverse-feedback-games/src/diff_solver.jl")
include("/home/chrisstrong/MRI_Driving/inverse-feedback-games/src/experiment_utils.jl") # NOTICE!! Many functions are defined there.

# added imports by Chris 
ENV["PYTHON"] = "/home/chrisstrong/miniconda3/envs/brain_env"
Pkg.build("PyCall")
using PyCall
import Base.length
using BenchmarkTools
using Flux
using ImageFiltering
using LazySets
using LinearAlgebra
using NPZ
using Plots
using Random

include("/home/chrisstrong/MRI_Driving/inverse-feedback-games/src/geometric_utils.jl")
include("/home/chrisstrong/MRI_Driving/inverse-feedback-games/src/data_processing_utils.jl")
include("/home/chrisstrong/MRI_Driving/inverse-feedback-games/src/visualization_utils.jl")
include("/home/chrisstrong/MRI_Driving/inverse-feedback-games/src/game_setup_utils.jl")


# Set some parameters
k_closest_vehicles = 15
k_closest_pedestrians = 0
road_width_cm = 1200
scaling_factor = 2000

#=
 Load the data 
=# 
# demonstration data
train_demonstrations_turns = load_pickle("/data/fmri/full_dataset/SP/generated/demonstrations/train_demonstrations_turns_k_closest_vehicles=15_k_closest_pedestrians=0.pkl")
train_point_sequences = load_pickle("/data/fmri/full_dataset/SP/generated/demonstrations/train_point_sequences.pkl")
train_demonstrations_turns_states = [train_demonstration[1] for train_demonstration in train_demonstrations_turns]

# map data 
centerline_filename = "/data/fmri/full_dataset/map_centerlines.npz"
centerlines = npzread(centerline_filename)["lines"]
centerlines = [LineSegment(centerlines[i, 1, :], centerlines[i, 2, :]) for i in 1:size(centerlines, 1)]

# scale the data 1
scaled_centerlines, scaled_demonstration_states, scaled_train_point_sequences, scaled_road_width = rescale_data(centerlines, train_demonstrations_turns_states, train_point_sequences, road_width_cm, k_closest_vehicles, k_closest_pedestrians, scaling_factor=scaling_factor)


# Test that you've loaded things correctly 
# visualize_road_from_key_points(scaled_train_point_sequences[1], scaled_road_width)
# savefig("/home/chrisstrong/MRI_Driving/inverse-feedback-games/examples/visualizations/test_trajectory_from_key_points.png")




using Plots
gr()
display(plot([1,3,2]))

hello = "world"

println("here")



#= 
    Finding parameters from initial conditions 
=# 

# Testing solving for the trajectory + computing loss for a single step 
demo_index = 1
start_index = 100
θ_initial = [600/scaling_factor, 0.02, 10.0, 5.0, 1.0 / (scaling_factor / 1000)^2 / (angular_control_scaling)^2, 
            1.0 / (scaling_factor / 1000)^2 / (velocity_control_scaling)^2, 900 / scaling_factor, 3.0, 10.0, 1.0,
            30.0, 0.2, 6.0]

costs = generate_ego_cost(scaled_demonstration_states[demo_index], scaled_train_point_sequences[demo_index])

player_inputs = (SVector(1,2),) # NOTE: the input is a tuple of inputs for each player
g = GeneralGame(game_horizon, player_inputs, dynamics, costs)
solver = iLQSolver(g, max_scale_backtrack=5, max_elwise_diff_step=Inf, equilibrium_type="FBNE")

# Forward pass, just running the solver
x0 = SVector(scaled_demonstration_states[demo_index][start_index, 1:4]..., (start_index-1) * ΔT)
c, expert_traj, strategies = solve(g, solver, SVector{nx}(vcat(x0, θ_initial)))

# Calculating loss and the gradient 
loss_val = loss(θ_initial, g, solver, scaled_demonstration_states[demo_index], start_index, 1:game_horizon, 1:nx, 1:nu)
grad = ForwardDiff.gradient(x -> loss(x, g, solver, demo, start_index, 1:game_horizon, 1:nx, 1:nu), θ_initial)
println("loss: ", loss_val)
println("grad: ", grad)


### Now try a loop where you sample batches and update theta eveyr time 
# first setup the costs, games, and solvers for each demonstration 
ego_vehicle_costs = [generate_ego_cost(demonstration, key_points) for (demonstration, key_points) in zip(scaled_demonstration_states, scaled_train_point_sequences)]
games = [GeneralGame(game_horizon, player_inputs, dynamics, costs) for costs in ego_vehicle_costs]
solvers = [iLQSolver(game,max_scale_backtrack=40, max_elwise_diff_converged = 0.05, max_elwise_diff_step = 0.1, equilibrium_type="FBNE") for game in games];

# next, setup your dataset of batches 
demo_indices = []
start_indices = []

stride_length = 1

for i = 1:length(scaled_demonstration_states)
    demo_length = size(scaled_demonstration_states[i])[1]
    for j = 1: stride_length: (demo_length - game_horizon + 1)
        push!(demo_indices, i)
        push!(start_indices, j)
    end
end

data_loader = Flux.DataLoader(collect(zip(demo_indices, start_indices)), batchsize=20, shuffle=true)

# finally, run batch gradient descent 
start_time = time()

#theta_0 = [0.0]
# theta_0 = [600/scaling_factor, 0.02, 10.0, 5.0, 1.0 / (scaling_factor / 1000)^2 / (angular_control_scaling)^2,
# 1.0 / (scaling_factor / 1000)^2 / (velocity_control_scaling)^2, 900 / scaling_factor, 3.0, 10.0, 1.0,
# 30.0, 0.2, 6.0, 900 / scaling_factor]
theta_0 = [600/scaling_factor, 0.02, 10.0, 5.0, 1.0 / (scaling_factor / 1000)^2 / (angular_control_scaling)^2, 
1.0 / (scaling_factor / 1000)^2 / (velocity_control_scaling)^2, 900 / scaling_factor, 3.0, 10.0, 1.0,
30.0, 0.2]


cur_θ = theta_0
step_size = 0.01

thetas = []
losses = []
losses_theta_0 = []

max_batches = 50 

for (i, batch) in enumerate(data_loader)
    demo_indices = [data[1] for data in batch]
    start_indices = [data[2] for data in batch]

    println(demo_indices)
    println(start_indices)

    # compute the batch loss 
    
    cur_loss = batch_loss(cur_θ, games, solvers, scaled_demonstration_states, demo_indices, start_indices, 1:game_horizon, 1:nx, 1:nu)
    println("batch loss: ", cur_loss)

    loss_theta_0 = batch_loss(theta_0, games, solvers, scaled_demonstration_states, demo_indices, start_indices, 1:game_horizon, 1:nx, 1:nu)


    grad = ForwardDiff.gradient(x -> batch_loss(x,  games, solvers, scaled_demonstration_states, demo_indices, start_indices, 1:game_horizon, 1:nx, 1:nu), cur_θ)

    cur_θ = cur_θ - step_size * grad
    println("theta: ", cur_θ)
    println("grad: ", grad)
    # println("computing them in loop")
    # for j = 1:length(demo_indices)
    #     demo_index = demo_indices[j]
    #     start_index = start_indices[j]
    #     println("demo index: ", demo_index, " is in before: ", demo_index in demo_indices[1:j-1])
    #     temp_loss = loss(cur_θ, games[demo_index], solvers[demo_index], scaled_demonstration_states[demo_index], start_index, 1:game_horizon, 1:nx, 1:nu)
    #     println("time to ", j, " iteration: ", time() - start_time)
    # end
    println("time after batch ", i, " ", time() - start_time)

    push!(thetas, cur_θ)
    push!(losses, cur_loss)
    push!(losses_theta_0, loss_theta_0)

    if i > max_batches
        break 
    end
end
