using Distributed
using Pkg
Pkg.activate("/home/chrisstrong/MRI_Driving/inverse-feedback-games/")

nprocs = Distributed.nprocs()

num_desired_procs = 20
if num_desired_procs > Distributed.nprocs() 
    println("adding ", num_desired_procs - Distributed.nprocs(), " processes")
    Distributed.addprocs(num_desired_procs - Distributed.nprocs(), exeflags=`--project=$(Base.active_project())`)
end

@everywhere begin 
    #= 
        Packages and setup
    =#

    using Pkg
    Pkg.activate("/home/chrisstrong/MRI_Driving/inverse-feedback-games/")
    Pkg.instantiate()

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

    pyplot()

    # Set some parameters
    k_closest_vehicles = 15
    k_closest_pedestrians = 0
    road_width_cm = 1200
    scaling_factor = 2000



    #=
    Load the data 
    =# 
    # Demonstration data
    # training
    train_demonstrations_turns = load_pickle("/data/fmri/full_dataset/SP/generated/demonstrations/train_demonstrations_turns_k_closest_vehicles=15_k_closest_pedestrians=0.pkl")
    train_point_sequences = load_pickle("/data/fmri/full_dataset/SP/generated/demonstrations/train_point_sequences.pkl")
    train_demonstrations_turns_states = [train_demonstration[1] for train_demonstration in train_demonstrations_turns]

    # validation 
    validation_demonstrations_turns = load_pickle("/data/fmri/full_dataset/SP/generated/demonstrations/validation_demonstrations_turns_k_closest_vehicles=15_k_closest_pedestrians=0.pkl")
    validation_point_sequences = load_pickle("/data/fmri/full_dataset/SP/generated/demonstrations/validation_point_sequences.pkl")
    validation_demonstrations_turns_states = [validation_demonstration[1] for validation_demonstration in validation_demonstrations_turns]

    # # straight line data too 
    # train_demonstrations_straights = load_pickle("/data/fmri/full_dataset/SP/generated/demonstrations/train_demonstrations_k_closest_vehicles=15_k_closest_pedestrians=0.pkl")
    # train_demonstrations_straights_states = [train_demonstration[1] for train_demonstration in train_demonstrations_straights]


    # map data 
    centerline_filename = "/data/fmri/full_dataset/map_centerlines.npz"
    centerlines = npzread(centerline_filename)["lines"]
    centerlines = [LineSegment(centerlines[i, 1, :], centerlines[i, 2, :]) for i in 1:size(centerlines, 1)]

    # scale the data 1
    train_scaled_centerlines, train_scaled_demonstration_states, train_scaled_point_sequences, scaled_road_width = rescale_data(centerlines, train_demonstrations_turns_states, train_point_sequences, road_width_cm, k_closest_vehicles, k_closest_pedestrians, scaling_factor=scaling_factor)
    validation_scaled_centerlines, validation_scaled_demonstration_states, validation_scaled_point_sequences, _ = rescale_data(centerlines, validation_demonstrations_turns_states, validation_point_sequences, road_width_cm, k_closest_vehicles, k_closest_pedestrians, scaling_factor=scaling_factor)

    # Test that you've loaded things correctly 
    visualize_road_from_key_points(train_scaled_point_sequences[1], scaled_road_width)
    savefig("/home/chrisstrong/MRI_Driving/inverse-feedback-games/examples/visualizations/test_trajectory_from_key_points.png")

    # load parameters
    theta_alldata = npzread("/home/chrisstrong/MRI_Driving/inverse-feedback-games/examples/visualizations/learning_rate=0.005/1_21_2025_adamtestalldata_batches=40_gradamp=1_desiredvelocitysigmoidscaling=5.0_seed=1000/thetas.npy")[:, end]
    theta_turns = npzread("/home/chrisstrong/MRI_Driving/inverse-feedback-games/examples/visualizations/learning_rate=0.005/1_21_2025_adamtestturns_batches=40_gradamp=1_desiredvelocitysigmoidscaling=5.0_seed=1000/thetas.npy")[:, end]
    theta_0 = npzread("/home/chrisstrong/MRI_Driving/inverse-feedback-games/examples/visualizations/learning_rate=0.005/2_3_2025_adamtestturns_batches=5_gradamp=1_desiredvelocitysigmoidscaling=5.0_seed=1000/theta_0.npy")
    theta_guess = npzread("/home/chrisstrong/MRI_Driving/inverse-feedback-games/examples/visualizations/learning_rate=0.005/2_3_2025_adamtestturns_batches=5_gradamp=1_desiredvelocitysigmoidscaling=5.0_seed=1000/theta_guess.npy")
    
    cur_theta = theta_guess
    base_path = "/data/fmri/full_dataset/SP/generated/models/2_4_2025/turning_model_features_from_demonstrations/testing/theta_guess/"
    use_train_data = false


    #=
    Test feature generation for a single moment
    =# 
    # Try gathering features from a single moment 
    # TODO: update desired_velocity_sigmoid_scaling
    desired_velocity_sigmoid_scaling = 5.0
    costs, feature_generation_function = cost_function_feature_gen_for_demonstration(train_scaled_demonstration_states[1], train_scaled_point_sequences[1], scaled_road_width, desired_velocity_sigmoid_scaling) 
    player_inputs = (SVector(1,2),) # NOTE: the input is a tuple of inputs for each player
    
    g = GeneralGame(game_horizon, player_inputs, dynamics, costs)
    solver = iLQSolver(g,max_scale_backtrack=40, max_elwise_diff_converged = 0.05, max_elwise_diff_step = 0.1, equilibrium_type="FBNE")
    
    cur_features, cur_feature_names, success_status = feature_generation_function(train_scaled_demonstration_states[1][1, 1:4], cur_theta, solver, g, 1) 
    println("cur features: ", cur_features)
    println("cur feature names: ", cur_feature_names)

    #= 
    Setup to generate features for a full demonstration 
    =# 
    function generate_features_from_demonstration(scaled_demonstration_states, scaled_train_point_sequence, scaled_road_width, θ)
        # Find the demonstration specific cost function (incldues the plan and ground truth other vehicle states)
        costs, feature_generation_function = cost_function_feature_gen_for_demonstration(scaled_demonstration_states, scaled_train_point_sequence, scaled_road_width, desired_velocity_sigmoid_scaling)
        player_inputs = (SVector(1,2),) # NOTE: the input is a tuple of inputs for each player
    
        g = GeneralGame(game_horizon, player_inputs, dynamics, costs)
        solver = iLQSolver(g,max_scale_backtrack=40, max_elwise_diff_converged = 0.05, max_elwise_diff_step = 0.1, equilibrium_type="FBNE")
    
        n_features = length(feature_generation_function(scaled_demonstration_states[1, 1:4], θ, solver, g, 1)[1])
        n_times = size(scaled_demonstration_states, 1)
    
        features = zeros(n_times, n_features)
        n_failures = 0
        start_time = time()
        for i = 1:n_times
            if (i-1) % 100 == 0
                percent_done = (i-1) / n_times * 100
                println("Percent done: ", round(percent_done, digits=2), "% in ", round(time() - start_time, digits=2), " seconds, out of ", n_times, " frames       ", " percent failure: ", round(100*(n_failures/i), digits=2))
            end
            # If we don't have enough remaining to populate the other vehicle positions then skip
            if (n_times - i) < game_horizon 
                continue 
            else            
                cur_features, cur_feature_names, success_status = feature_generation_function(scaled_demonstration_states[i, 1:4], θ, solver, g, i) 
                features[i, :] = cur_features 
                if success_status == false 
                    n_failures += 1
                end
            end 
        end
    
        println("Failures: ", n_failures, ", ", n_failures/n_times * 100, "%")
        return features, n_failures
    end
    
    # test it out 
    #feats, feature_names = generate_features_from_demonstration(train_scaled_demonstration_states[1], train_scaled_point_sequences[1], scaled_road_width, cur_theta) 
end

#= 
Setup to run it all on our data 
=# 
if use_train_data
    scaled_demonstration_states = train_scaled_demonstration_states
    scaled_point_sequences = train_scaled_point_sequences 
else 
    scaled_demonstration_states = validation_scaled_demonstration_states
    scaled_point_sequences = validation_scaled_point_sequences
end


n_demonstrations = length(scaled_demonstration_states)
total_frames = sum([size(scaled_demonstration_states[i], 1) for i = 1:n_demonstrations])
n_failures_list = zeros(n_demonstrations)

println("Running ", n_demonstrations, " demonstrations with ", total_frames, " frames")
start_time = time() 

futures = [@spawnat :any generate_features_from_demonstration(scaled_demonstration_states[i], scaled_point_sequences[i], scaled_road_width, cur_theta) for i = 1:n_demonstrations]
results = [fetch(f) for f in futures]

feature_matrices = [result[1] for result in results]
n_failures_list = [result[2] for result in results]

end_time = time()

total_failures = sum(n_failures_list)
println("Full feature generation took: ", round((end_time - start_time)/60, digits=3), " minutes")
println("Failures: ", total_failures, " out of ", total_frames, "   ", round(total_failures / total_frames * 100, digits=3), "%")
    
    
# Save out feature_matrices using NPZ
if use_train_data 
    filepath = joinpath(base_path, "train")
else 
    filepath = joinpath(base_path, "validation")
end

mkpath(filepath)
[npzwrite(joinpath(filepath, "demonstration_"*string(i)*".npy"), feature_matrices[i]) for i = 1:n_demonstrations]

# also save out the feature names 
feature_name_file = joinpath(base_path, "feature_names.npy")
py"""
import numpy as np
"""
py"np.save"(feature_name_file, cur_feature_names)




