#=
    Imports and setup 
=#

include_things = true
if include_things 
    using Pkg
    Pkg.activate("/home/chrisstrong/MRI_Driving/inverse-feedback-games/")

    using Infiltrator
    using Revise 

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
    #Pkg.build("PyCall")
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

end

pyplot()


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
visualize_road_from_key_points(scaled_train_point_sequences[1], scaled_road_width)
savefig("/home/chrisstrong/MRI_Driving/inverse-feedback-games/examples/visualizations/test_trajectory_from_key_points.png")


#= 
    Finding parameters from initial conditions 
=# 
# Testing solving for the trajectory + computing loss for a single step 
demo_index = 1
start_index = 500

θ_initial = [600/scaling_factor, 0.02, 10.0, 5.0, 1.0 / (scaling_factor / 1000)^2 / (angular_control_scaling)^2, 
            1.0 / (scaling_factor / 1000)^2 / (velocity_control_scaling)^2, 900 / scaling_factor, 3.0, 10.0, 1.0,
            30.0, 0.2, 900 / scaling_factor]

# θ_initial = zeros(12)
# θ_initial[5] = 1.0
# θ_initial[6] = 1.0

# include("/home/chrisstrong/MRI_Driving/inverse-feedback-games/src/game_setup_utils.jl")


costs = generate_ego_cost(scaled_demonstration_states[demo_index], scaled_train_point_sequences[demo_index], scaled_road_width, 6.0)

player_inputs = (SVector(1,2),) # NOTE: the input is a tuple of inputs for each player
g = GeneralGame(game_horizon, player_inputs, dynamics, costs)
solver = iLQSolver(g, max_scale_backtrack=5, max_elwise_diff_step=Inf, equilibrium_type="FBNE")

# Forward pass, just running the solver
x0 = SVector(scaled_demonstration_states[demo_index][start_index, 1:4]..., (start_index-1) * ΔT)
c, expert_traj, strategies = solve(g, solver, SVector{nx}(vcat(x0, θ_initial)))


# Calculating loss and the gradient 
loss_val = loss(θ_initial, g, solver, scaled_demonstration_states[demo_index], start_index, 1:game_horizon, 1:nx, 1:nu)
grad = ForwardDiff.gradient(x -> loss(x, g, solver, scaled_demonstration_states[demo_index], start_index, 1:game_horizon, 1:nx, 1:nu), θ_initial)
println("loss: ", loss_val)
println("grad: ", grad)

θ_after = θ_initial - 0.1 * grad
loss_val_after = loss(θ_after, g, solver, scaled_demonstration_states[demo_index], start_index, 1:game_horizon, 1:nx, 1:nu)
println("loss after: ", loss_val_after)


# visualize the difference between the trajectories from this single step 
trajectory_plan = trajectory_plan_from_key_points(scaled_train_point_sequences[demo_index], scaled_road_width)
traj_before = trajectory_from_solver(g, solver, scaled_demonstration_states[demo_index], start_index, θ_initial)
traj_after = trajectory_from_solver(g, solver, scaled_demonstration_states[demo_index], start_index, θ_after)

anim = animate_trajectories([traj_before, traj_after], ["red", "yellow"], ["before theta update", "after theta update"], scaled_demonstration_states[demo_index][start_index:end, :], trajectory_plan, scaled_road_width, scaled_centerlines, field_of_view = 10000/scaling_factor, time_spacing=5)
gif(anim, string("visualizations/initial_checks/initial_vis_demo=", demo_index, "_startindex=", start_index, ".gif"), fps=15)

# also rollout in closed loop and visualize the difference 
rollout_before = rollout_trajectory(g, solver, dynamics, scaled_demonstration_states[demo_index], start_index, θ = θ_initial, num_steps=100)
rollout_after = rollout_trajectory(g, solver, dynamics, scaled_demonstration_states[demo_index], start_index, θ = θ_after, num_steps=100)

anim_before_closedloop = animate_trajectories([rollout_before, rollout_after], ["red", "yellow"], ["before theta update", "after theta update"], scaled_demonstration_states[demo_index][start_index:end, :], trajectory_plan, scaled_road_width, scaled_centerlines, field_of_view = 10000/scaling_factor, time_spacing=5)
gif(anim_before_closedloop, string("visualizations/initial_checks/initial_vis_closedloop_demo=", demo_index, "_startindex=", start_index, ".gif"), fps=15)


### Now try a loop where you sample batches and update theta every time 
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

# data_loader = Flux.DataLoader(collect(zip(demo_indices, start_indices)), batchsize=20, shuffle=true)

# next setup the costs, games, and solvers for each d, emonstration 
for desired_velocity_sigmoid_scaling ∈ [1.0, 2.0, 5.0, 10.0]
    step_size = 0.1 # was 0.05 for a while 
    max_batches = 20
    max_anims = 10
    
    println("exploring desired velocity sigmoid scaling: ", desired_velocity_sigmoid_scaling)
    
    data_loader = Flux.DataLoader(collect(zip(demo_indices, start_indices)), batchsize=20, shuffle=true)

    ego_vehicle_costs = [generate_ego_cost(demonstration, key_points, scaled_road_width, desired_velocity_sigmoid_scaling) for (demonstration, key_points) in zip(scaled_demonstration_states, scaled_train_point_sequences)]
    games = [GeneralGame(game_horizon, player_inputs, dynamics, costs) for costs in ego_vehicle_costs]
    solvers = [iLQSolver(game,max_scale_backtrack=40, max_elwise_diff_converged = 0.05, max_elwise_diff_step = 0.1, equilibrium_type="FBNE") for game in games];
    trajectory_plans = [trajectory_plan_from_key_points(key_points, scaled_road_width) for key_points in scaled_train_point_sequences] # just for visualization 


    # finally, run batch gradient descent 
    start_time = time()

    # theta_0 = [0.0]
    # theta_0 = [600/scaling_factor, 0.02, 10.0, 5.0, 1.0 / (scaling_factor / 1000)^2 / (angular_control_scaling)^2,
    # 1.0 / (scaling_factor / 1000)^2 / (velocity_control_scaling)^2, 900 / scaling_factor, 3.0, 10.0, 1.0,
    # 30.0, 0.2, 6.0, 900 / scaling_factor]
    theta_guess = [600/scaling_factor, 0.02, 10.0, 5.0, 1.0 / (scaling_factor / 1000)^2 / (angular_control_scaling)^2, 
    1.0 / (scaling_factor / 1000)^2 / (velocity_control_scaling)^2, 900 / scaling_factor, 3.0, 10.0, 1.0,
    30.0, 0.2, 900 / scaling_factor]

    param_rescaling = 10
    indices_to_scale = [3, 4, 8, 9, 11]
    theta_guess[indices_to_scale] /= param_rescaling

    # theta_0[indices_to_scale] /= param_rescaling


    # start with random on interval [0, 1]
    #theta_0 = rand(n_params_to_fit)

    #theta_0 = 1 * ones(13)
    #Random.seed!(1234)
    #Random.seed!(1111)

    #
    """
    include("/home/chrisstrong/MRI_Driving/inverse-feedback-games/src/geometric_utils.jl")
    include("/home/chrisstrong/MRI_Driving/inverse-feedback-games/src/data_processing_utils.jl")
    include("/home/chrisstrong/MRI_Driving/inverse-feedback-games/src/visualization_utils.jl")
    include("/home/chrisstrong/MRI_Driving/inverse-feedback-games/src/game_setup_utils.jl")

    """

    n_params = 13
    seed = 1000
    Random.seed!(seed)
    theta_0 = rand(n_params)
    theta_0[5] = 0.25 # 1.0
    theta_0[6] = 0.25 # 1.0

    cur_θ = theta_0

    thetas = Vector{Float64}[]
    losses = Float64[]
    losses_theta_0 = Float64[]
    losses_theta_guess = Float64[]

    indices_to_amplify_gradient = [2, 3, 7, 8, 10, 11, 12, 13]
    grad_amplification = 1 


    for (i, batch) in enumerate(data_loader)
        demo_indices = [data[1] for data in batch]
        start_indices = [data[2] for data in batch]

        println(demo_indices)
        println(start_indices)
        # compute the batch loss 
        
        cur_loss = batch_loss(cur_θ, games, solvers, scaled_demonstration_states, demo_indices, start_indices, 1:game_horizon, 1:nx, 1:nu)
        println("batch loss: ", cur_loss)

        loss_theta_0 = batch_loss(theta_0, games, solvers, scaled_demonstration_states, demo_indices, start_indices, 1:game_horizon, 1:nx, 1:nu)
        println("loss theta 0: ", loss_theta_0)

        loss_theta_guess = batch_loss(theta_guess, games, solvers, scaled_demonstration_states, demo_indices, start_indices, 1:game_horizon, 1:nx, 1:nu)
        println("loss theta guess: ", loss_theta_guess)

        grad = ForwardDiff.gradient(x -> batch_loss(x,  games, solvers, scaled_demonstration_states, demo_indices, start_indices, 1:game_horizon, 1:nx, 1:nu), cur_θ)

        println("grad before amplification: ", grad)
        grad[indices_to_amplify_gradient] *= grad_amplification

        cur_θ = cur_θ - step_size * grad
        println("theta: ", cur_θ)
        println("grad: ", grad)
        println("time after batch ", i, " ", time() - start_time)

        push!(thetas, cur_θ)
        push!(losses, cur_loss)
        push!(losses_theta_0, loss_theta_0)
        push!(losses_theta_guess, loss_theta_guess)

        if i >= max_batches
            break 
        end
    end
    folder_name = joinpath(string("learning_rate=", step_size), string("temp_gradamp=", grad_amplification, "_desiredvelocitysigmoidscaling=", desired_velocity_sigmoid_scaling, "seed=", seed)) # "matched_initialguess_control_cost_modified_vdesired_scaling_diffseed"
    directory = joinpath("/home/chrisstrong/MRI_Driving/inverse-feedback-games/examples/visualizations", folder_name)
    mkpath(directory)

    # save the thetas, losses
    npzwrite(joinpath(directory, "thetas.npy"), reduce(hcat, thetas))
    npzwrite(joinpath(directory, "losses.npy"), losses)
    npzwrite(joinpath(directory, "losses_theta_0.npy"), losses_theta_0)
    npzwrite(joinpath(directory, "losses_theta_guess.npy"), losses_theta_guess)

    print("final theta: ", thetas[end])

    plot(losses, xlabel="batch", ylabel="loss", label="losses")
    plot!(losses_theta_0, label="losses with theta_0")
    plot!(losses_theta_guess, label="losses with theta_guess")
    plot!(ylims=[0, 0.25])

    savefig(joinpath(directory, "losses.png"))

    plot(losses, xlabel="batch", ylabel="loss", label="losses")
    plot!(losses_theta_guess, label="losses with theta_guess")
    plot!(ylims=[0, 0.25])

    savefig(joinpath(directory, "losses_no_theta_0.png"))

    # for each theta, plot it 
    for i = 1:length(cur_θ)
        plot([thetas[j][i] for j = 1:length(thetas)], xlabel="batch", ylabel="parameter value", ylim=[0, 1])
        savefig(joinpath(directory, "theta_" * string(i) * ".png"))
    end

    plot()
    for i = 1:length(cur_θ)
        plot!([thetas[j][i] for j = 1:length(thetas)], xlabel="batch", ylabel="parameter value")
    end
    savefig(joinpath(directory, "thetas_all.png"))

    # animate start vs. end theta vs. theta_guess on a few examples 
    mkpath(joinpath(directory, "animations"))
    for (i, (demo_index, start_index)) in enumerate(rand(collect(data_loader))) # sample a random batch 
        println("visualizing demo index ", demo_index, " at start index ", start_index)
        # also rollout in closed loop and visualize the difference 
        rollout_length = min(200, size(scaled_demonstration_states[demo_index], 1) - start_index - game_horizon) # so don't rollout too far if we don't have the space 
        println("rollout length: ", rollout_length)

        rollout_before = rollout_trajectory(games[demo_index], solvers[demo_index], dynamics, scaled_demonstration_states[demo_index], start_index, θ = thetas[1], num_steps=rollout_length)
        rollout_after = rollout_trajectory(games[demo_index], solvers[demo_index], dynamics, scaled_demonstration_states[demo_index], start_index, θ = thetas[end], num_steps=rollout_length)
        rollout_theta_guess = rollout_trajectory(games[demo_index], solvers[demo_index], dynamics, scaled_demonstration_states[demo_index], start_index, θ = theta_guess, num_steps=rollout_length)

        rollouts = []
        colors = []
        labels = []

        if ~isnothing(rollout_before)
            push!(rollouts, rollout_before)
            push!(colors, "red")
            push!(labels, "theta_0")
        end

        if ~isnothing(rollout_after)
            push!(rollouts, rollout_after)
            push!(colors, "yellow")
            push!(labels, "theta_final")
        end

        if ~isnothing(rollout_theta_guess)
            push!(rollouts, rollout_theta_guess)
            push!(colors, "pink")
            push!(labels, "theta_guess")
        end

        if length(rollouts) > 0
            anim_before_closedloop = animate_trajectories(rollouts, colors, labels, scaled_demonstration_states[demo_index][start_index:end, :], trajectory_plans[demo_index], scaled_road_width, scaled_centerlines, field_of_view = 10000/scaling_factor, time_spacing=5)    
            gif(anim_before_closedloop, joinpath(directory, string("animations/batch_training_demo=", demo_index, "_startindex=", start_index, ".gif")), fps=15)
        else
            println("no successful rollouts!!!!!!!")
        end

        if i >= max_anims
            break
        end
    end
end

# # Poke into specific demonstration / start indices 
# theta = thetas[end]
# theta[7] = 0.3 # comfortable distance 
# theta[10] = 0.57 # weight on comfortable distance 

theta = copy(thetas[end])
theta[2] = 1.0
theta_2 = copy(thetas[end])
theta_2[2] = 15.0
demo_index = 6
start_index = 648
rollout = rollout_trajectory(games[demo_index], solvers[demo_index], dynamics, scaled_demonstration_states[demo_index], start_index, θ = theta, num_steps=200)
rollout2 = rollout_trajectory(games[demo_index], solvers[demo_index], dynamics, scaled_demonstration_states[demo_index], start_index, θ = theta_2, num_steps=200)
anim = animate_trajectories([rollout, rollout2], ["yellow", "red"], ["theta1", "theta2"], scaled_demonstration_states[demo_index][start_index:end, :], trajectory_plans[demo_index], scaled_road_width, scaled_centerlines, field_of_view = 10000/scaling_factor, time_spacing=5)    

# anim = animate_trajectories([], [], [], scaled_demonstration_states[demo_index][start_index:end, :], trajectory_plans[demo_index], scaled_road_width, scaled_centerlines, field_of_view = 10000/scaling_factor, time_spacing=5)    


gif(anim, joinpath(directory, string("animations/temp_test=", demo_index, "_startindex=", start_index, ".gif")), fps=15)



# TODO:
# try adding back in fitting of the desired velocity sigmoid. 
# if that doesn't work just tweak it a bit and see? and then do feature generation for this setup. 