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

# straight line data too 
train_demonstrations_straights = load_pickle("/data/fmri/full_dataset/SP/generated/demonstrations/train_demonstrations_k_closest_vehicles=15_k_closest_pedestrians=0.pkl")
train_demonstrations_straights_states = [train_demonstration[1] for train_demonstration in train_demonstrations_straights]

# map data 
centerline_filename = "/data/fmri/full_dataset/map_centerlines.npz"
centerlines = npzread(centerline_filename)["lines"]
centerlines = [LineSegment(centerlines[i, 1, :], centerlines[i, 2, :]) for i in 1:size(centerlines, 1)]

# scale the data 1
scaled_centerlines, scaled_demonstration_states, scaled_train_point_sequences, scaled_road_width = rescale_data(centerlines, train_demonstrations_turns_states, train_point_sequences, road_width_cm, k_closest_vehicles, k_closest_pedestrians, scaling_factor=scaling_factor)
trajectory_plans = [trajectory_plan_from_key_points(key_points, scaled_road_width) for key_points in scaled_train_point_sequences] # just for visualization 


# Test that you've loaded things correctly 
visualize_road_from_key_points(scaled_train_point_sequences[1], scaled_road_width)
savefig("/home/chrisstrong/MRI_Driving/inverse-feedback-games/examples/visualizations/test_trajectory_from_key_points.png")

#= 
    Trying to find a way of getting the indices of the turns 
=# 

# attempt 1 - see what indices are in the straight run demonstrations 
n_runs = 18

indices_in_straights_per_run = [Dict() for i = 1:n_runs]
for demo in train_demonstrations_straights_states
    for i = 1:size(demo, 1)
        # get the run and sample index 
        run_index = Int(demo[i, end]) + 1
        sample_index = Int(demo[i, end-1])
        indices_in_straights_per_run[run_index][sample_index] = true
    end
end

# attempt 2 - check the type of segment that the position gets projected onto 
indices_in_turns_per_demo = [Dict() for i = 1:length(train_demonstrations_turns_states)]
for (i, demo) in enumerate(scaled_demonstration_states)
    for j = 1:size(demo, 1)
        position = demo[j, 1:2]
        _, proj_index = project_and_argmin(position, trajectory_plans[i])
        segment = trajectory_plans[i].subtrajectories[proj_index]
        # println("demo, index: ", (i, j))

        # println("position: ", position)
        # println("proj index: ", proj_index)
        # println("segment: ", segment)
        if typeof(segment) <: Arc
            println("----------added in-------")
            println("demo, index: ", (i, j))
            indices_in_turns_per_demo[i][j] = true 
        end
    end
end
    




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

# ForwardDiff.jacobian(y -> ForwardDiff.jacobian(x -> [loss(x, g, solver, scaled_demonstration_states[demo_index], start_index, 1:game_horizon, 1:nx, 1:nu)], y), θ_initial)
# hess = ForwardDiff.hessian(x -> loss(x, g, solver, scaled_demonstration_states[demo_index], start_index, 1:game_horizon, 1:nx, 1:nu), θ_initial)
# println("hessian: ", hess)

# θ_after_hess = θ_initial - inv(hess) * grad
# loss_val_after_hess = loss(θ_after_hess, g, solver, scaled_demonstration_states[demo_index], start_index, 1:game_horizon, 1:nx, 1:nu)
# println("loss after newton: ", loss_val_after_hess)

θ = θ_initial
#grad_flux = Flux.gradient(x -> loss(x, g, solver, scaled_demonstration_states[demo_index], start_index, 1:game_horizon, 1:nx, 1:nu), θ)
grad_forwarddiff = ForwardDiff.gradient(x -> loss(x, g, solver, scaled_demonstration_states[demo_index], start_index, 1:game_horizon, 1:nx, 1:nu), θ)
opt = Descent(0.1) # Gradient descent with learning rate 0.1
Flux.update!(opt, θ, grad)

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


only_turns = false
for i = 1:length(scaled_demonstration_states)
    demo = scaled_demonstration_states[i]
    demo_length = size(demo)[1]
    for j = 1: stride_length: (demo_length - game_horizon + 1)
        if only_turns 
            if haskey(indices_in_turns_per_demo[i], j)
                push!(demo_indices, i)
                push!(start_indices, j)
            end 
        else 
            push!(demo_indices, i)
            push!(start_indices, j)
        end
    end
end

# data_loader = Flux.DataLoader(collect(zip(demo_indices, start_indices)), batchsize=20, shuffle=true)

# next setup the costs, games, and solvers for each d, emonstration 
desired_velocity_sigmoid_scaling = 5.0
#for desired_velocity_sigmoid_scaling ∈ [5.0]

    step_size = 0.005 # was 0.05 for a while 
    max_batches = 200
    max_anims = 10
    seed = 1000
    indices_to_amplify_gradient = [2, 3, 7, 8, 10, 11, 12, 13]
    grad_amplification = 1 

    # opt = Descent(step_size)
    opt = ADAM(step_size, (0.9, 0.999))

    folder_name = joinpath(string("learning_rate=", step_size), string("12_20_2024_adamteststraights_batches=", max_batches, "_gradamp=", grad_amplification, "_desiredvelocitysigmoidscaling=", desired_velocity_sigmoid_scaling, "_seed=", seed)) # "matched_initialguess_control_cost_modified_vdesired_scaling_diffseed"

    
    println("exploring desired velocity sigmoid scaling: ", desired_velocity_sigmoid_scaling)
    
    data_loader = Flux.DataLoader(collect(zip(demo_indices, start_indices)), batchsize=20, shuffle=true)

    ego_vehicle_costs = [generate_ego_cost(demonstration, key_points, scaled_road_width, desired_velocity_sigmoid_scaling) for (demonstration, key_points) in zip(scaled_demonstration_states, scaled_train_point_sequences)]
    games = [GeneralGame(game_horizon, player_inputs, dynamics, costs) for costs in ego_vehicle_costs]
    solvers = [iLQSolver(game,max_scale_backtrack=40, max_elwise_diff_converged = 0.05, max_elwise_diff_step = 0.1, equilibrium_type="FBNE") for game in games];


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

    Random.seed!(seed)


    # start with random on interval [0, 1]
    #theta_0 = rand(n_params_to_fit)

    #theta_0 = 1 * ones(13)
    #Random.seed!(1234)
    #Random.seed!(1111)

    n_params = 13
    theta_0 = rand(n_params)
    theta_0[5] = 0.25 # 1.0
    theta_0[6] = 0.25 # 1.0

    cur_θ = copy(theta_0)

    thetas = Vector{Float64}[]
    losses = Float64[]
    losses_theta_0 = Float64[]
    losses_theta_guess = Float64[]
    grad_norms = Float64[]


    for (i, batch) in enumerate(data_loader)      
        global cur_θ

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
        push!(grad_norms, norm(grad))

        println("grad before amplification: ", grad)
        grad[indices_to_amplify_gradient] *= grad_amplification

        push!(thetas, copy(cur_θ)) # record pre gradient update theta

        update!(opt, cur_θ, grad)
        #cur_θ = cur_θ - step_size * grad
        println("theta: ", cur_θ)
        println("grad: ", grad)
        println("time after batch ", i, " ", time() - start_time)

        push!(losses, cur_loss)
        push!(losses_theta_0, loss_theta_0)
        push!(losses_theta_guess, loss_theta_guess)

        if i >= max_batches
            break 
        end
    end

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

    plot()
    plot(grad_norms, xlabel="batch", ylabel="grad norm", label="grad norm")
    savefig(joinpath(directory, "grad_norms.png"))

    # animate start vs. end theta vs. theta_guess on a few examples 
    mkpath(joinpath(directory, "animations"))
    for (i, (demo_index, start_index)) in enumerate(rand(collect(data_loader))) # sample a random batch 
        println("visualizing demo index ", demo_index, " at start index ", start_index)
        # also rollout in closed loop and visualize the difference 
        rollout_length = min(100, size(scaled_demonstration_states[demo_index], 1) - start_index - game_horizon) # so don't rollout too far if we don't have the space 
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

        println("successful rollouts: ", labels)

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
# end

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



#=
    recreating stuff to visualize 

=# 
demo_index = 43
start_index = 818

thetas_fulldata = npzread("/home/chrisstrong/MRI_Driving/inverse-feedback-games/examples/visualizations/learning_rate=0.05/temp_withdiagonals_gradamp=1_desiredvelocitysigmoidscaling=5.0seed=1000/thetas.npy")
thetas_turns = npzread("/home/chrisstrong/MRI_Driving/inverse-feedback-games/examples/visualizations/learning_rate=0.05/12_19_2024_adamtest=200_gradamp=1_desiredvelocitysigmoidscaling=5.0_seed=1000/thetas.npy")
thetas_adam_fulldata = npzread("/home/chrisstrong/MRI_Driving/inverse-feedback-games/examples/visualizations/learning_rate=0.005/12_19_2024_adamteststraights_batches=200_gradamp=1_desiredvelocitysigmoidscaling=5.0_seed=1000/thetas.npy")


loss_final = loss(thetas_fulldata[:, end], games[demo_index], solvers[demo_index], scaled_demonstration_states[demo_index], start_index, 1:game_horizon, 1:nx, 1:nu)

loss_guess = loss(theta_guess, games[demo_index], solvers[demo_index], scaled_demonstration_states[demo_index], start_index, 1:game_horizon, 1:nx, 1:nu)


# bar plot comparing the two 
labels = ["v desired", "θ along curve", "θ from curve", "θ velocity", "θ u1", "θ u2", "comf. distance", "distance scaling", "comf. dist. σ scale", "θ distance", "dist. curve σ scale", "dist. curve σ center", "v des. σ thresh."]

labels = ["v desired", "θ along curve", "θ from curve", "θ velocity", "θ u1", "θ u2", "comf. distance", "distance scaling", "comf. dist. σ scale", "θ distance", "dist. curve σ scale", "dist. curve σ center", "v des. σ thresh."]

xs = 1:5:5*length(labels)

scatter(xs, thetas_fulldata[:, end], xticks=(xs, labels), ylim=[-0.1, 1.0], size=(1500, 300), markersize=5, label="Full Data")
scatter!(xs, thetas_turns[:, end], label="Turns", markersize=5, alpha=0.5)
savefig(joinpath(directory, "thetas_fulldata_bar.png"))



# animate start vs. end theta vs. theta_guess on a few examples 
mkpath(joinpath(directory, "animations/comparison/"))
for (i, (demo_index, start_index)) in enumerate(rand(collect(data_loader))) # sample a random batch 
    # start the index a second back 
    start_index = start_index - 15
    
    println("visualizing demo index ", demo_index, " at start index ", start_index)
    # also rollout in closed loop and visualize the difference 
    rollout_length = min(100, size(scaled_demonstration_states[demo_index], 1) - start_index - game_horizon) # so don't rollout too far if we don't have the space 
    println("rollout length: ", rollout_length)

    rollout_before = rollout_trajectory(games[demo_index], solvers[demo_index], dynamics, scaled_demonstration_states[demo_index], start_index, θ = thetas[1], num_steps=rollout_length)
    rollout_after = rollout_trajectory(games[demo_index], solvers[demo_index], dynamics, scaled_demonstration_states[demo_index], start_index, θ = thetas_adam_fulldata[:, end], num_steps=rollout_length)
    rollout_theta_guess = rollout_trajectory(games[demo_index], solvers[demo_index], dynamics, scaled_demonstration_states[demo_index], start_index, θ = theta_guess, num_steps=rollout_length)

    rollout_theta_fulldata = rollout_trajectory(games[demo_index], solvers[demo_index], dynamics, scaled_demonstration_states[demo_index], start_index, θ = thetas_fulldata[:, end], num_steps=rollout_length)

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
        push!(labels, "theta ADAM full data")
    end

    if ~isnothing(rollout_theta_guess)
        push!(rollouts, rollout_theta_guess)
        push!(colors, "pink")
        push!(labels, "theta_guess")
    end

    if ~isnothing(rollout_theta_fulldata)
        push!(rollouts, rollout_theta_fulldata)
        push!(colors, "black")
        push!(labels, "theta full data")
    end

    println("successful rollouts: ", labels)

    if length(rollouts) > 0
        anim_before_closedloop = animate_trajectories(rollouts, colors, labels, scaled_demonstration_states[demo_index][start_index:end, :], trajectory_plans[demo_index], scaled_road_width, scaled_centerlines, field_of_view = 10000/scaling_factor, time_spacing=5)    
        gif(anim_before_closedloop, joinpath(directory, string("animations/comparison/batch_training_demo=", demo_index, "_startindex=", start_index, ".gif")), fps=15)
    else
        println("no successful rollouts!!!!!!!")
    end

    if i >= max_anims
        break
    end
end


# bar(xs, thetas_fulldata[:, end], xticks=xs, labels=labels, bar_width=0.5)
# savefig(joinpath(directory, "thetas_fulldata_bar.png"))

# bar(labels, thetas_fulldata[:, end], xticks=(1:5:5*length(labels), labels), bar_width=0.5)
# savefig(joinpath(directory, "thetas_fulldata_bar.png"))



# desired_velocity_sigmoid_scaling = 5.0
# grad_amplification = 1
# step_size = 0.05
# seed = 1000
# max_anims = 10


# ego_vehicle_costs = [generate_ego_cost(demonstration, key_points, scaled_road_width, desired_velocity_sigmoid_scaling) for (demonstration, key_points) in zip(scaled_demonstration_states, scaled_train_point_sequences)]
# games = [GeneralGame(game_horizon, player_inputs, dynamics, costs) for costs in ego_vehicle_costs]
# solvers = [iLQSolver(game,max_scale_backtrack=40, max_elwise_diff_converged = 0.05, max_elwise_diff_step = 0.1, equilibrium_type="FBNE") for game in games];
# trajectory_plans = [trajectory_plan_from_key_points(key_points, scaled_road_width) for key_points in scaled_train_point_sequences] # just for visualization 


data_loader = Flux.DataLoader(collect(zip(demo_indices, start_indices)), batchsize=20, shuffle=true)
folder_name = joinpath(string("learning_rate=", step_size), string("temp_withdiagonals_gradamp=", grad_amplification, "_desiredvelocitysigmoidscaling=", desired_velocity_sigmoid_scaling, "seed=", seed)) # "matched_initialguess_control_cost_modified_vdesired_scaling_diffseed"

directory = joinpath("/home/chrisstrong/MRI_Driving/inverse-feedback-games/examples/visualizations", folder_name)
 # animate start vs. end theta vs. theta_guess on a few examples 
 mkpath(joinpath(directory, "animations"))
 for (i, (demo_index, start_index)) in enumerate(rand(collect(data_loader))) # sample a random batch 
     println("visualizing demo index ", demo_index, " at start index ", start_index)
     # also rollout in closed loop and visualize the difference 
     rollout_length = min(100, size(scaled_demonstration_states[demo_index], 1) - start_index - game_horizon) # so don't rollout too far if we don't have the space 
     println("rollout length: ", rollout_length)

     rollout_before = rollout_trajectory(games[demo_index], solvers[demo_index], dynamics, scaled_demonstration_states[demo_index], start_index, θ = thetas[:, 1], num_steps=rollout_length)
     rollout_after = rollout_trajectory(games[demo_index], solvers[demo_index], dynamics, scaled_demonstration_states[demo_index], start_index, θ = thetas[:, end], num_steps=rollout_length)
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

     println("successful rollouts: ", labels)

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