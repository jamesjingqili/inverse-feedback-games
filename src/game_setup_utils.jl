# Some baseline parameters
n_params_to_fit = 13
nx, nu, ΔT, game_horizon = 5+n_params_to_fit, 2, 1/15, 30
ΔT_data = 1/15


# Setup the dynamics 
angular_control_scaling = 1 # 3
velocity_control_scaling = 1 # 3

struct SingleUnicycleAugmentedTT <: ControlSystem{ΔT,nx,nu} end
# state: (px, py, phi, v)
dx(cs::SingleUnicycleAugmentedTT, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1] / angular_control_scaling, u[2] / velocity_control_scaling, 0.0, zeros(n_params_to_fit)...) # last element of state is the initial time 
dynamics = SingleUnicycleAugmentedTT()

# Other cost parameters
k_closest_vehicles_for_cost = 5



function generate_ego_cost(demo, key_points, scaled_road_width, desired_velocity_sigmoid_scaling)
    ego_cost_fcn, feature_gen_fcn = cost_function_feature_gen_for_demonstration(demo, key_points, scaled_road_width, desired_velocity_sigmoid_scaling)
    return ego_cost_fcn
end

function cost_function_feature_gen_for_demonstration(demo, key_points, scaled_road_width, desired_velocity_sigmoid_scaling)
    trajectory_plan = trajectory_plan_from_key_points(key_points, scaled_road_width)
    
    function wrapper_fcn(point)
        dist_along_curve, dist_from_curve, _ = project_frenet_coordinates(point, trajectory_plan)
        return [dist_along_curve, dist_from_curve]
    end 
    
    function ego_vehicle_cost(g, x, u, t)
        # cost for ego vehicle
        ego_x, ego_y, ego_phi, ego_v, start_time = x[1:5]

        # parameters 
        ego_v_desired, ego_weight_distance_along_curve, ego_weight_distance_from_curve, ego_weight_velocity, ego_weight_control_1,
        ego_weight_control_2, ego_comfortable_distance, scaling_distance_from_curve, comfortable_distance_sigmoid_scaling, ego_weight_distance,
        distance_from_curve_sigmoid_scaling, distance_from_curve_sigmoid_center, desired_velocity_sigmoid_threshold = x[6:18]
        
        # rescaling
        # 1    0.3 - small - ego_v_desired 
        # 2    0.02 - small - ego_weight_distance_along_curve --- with new version of it should be large
        # 3   10.0 - large - ego_weight_distance_from_curve
        # 4    5.0 - large - ego_weight_velocity
        # 5    0.25 - small - ego_weight_control_1
        # 6    0.25 - small - ego_weight_control_2
        # 7    0.45 - small - ego_comfortable_distnace 
        # 8    3.0 - large - scaling_distance_from_curve
        # 9   10.0 - large - comfortable_distance_sigmoid_scaling
        # 10    1.0 - small - ego_weight_distance 
        # 11   30.0 - large - distance_from_curve_sigmoid_scaling
        # 12    0.2 - small - distance_from_curve_sigmoid_center 

        # 13    0.7 - small - desired_velocity_sigmoid_threshold

        # parameters that are large we'd like to scale to be a similar size, so we'll scale them up by a factor of 20 here 
        # and then the initial theta we can initialize to some random 0 to 1 thing?  
        param_rescaling = 10
        
        ego_weight_distance_along_curve = ego_weight_distance_along_curve * param_rescaling

        ego_weight_distance_from_curve = ego_weight_distance_from_curve * param_rescaling 
        ego_weight_velocity = ego_weight_velocity * param_rescaling

        scaling_distance_from_curve = scaling_distance_from_curve * param_rescaling 
        comfortable_distance_sigmoid_scaling = comfortable_distance_sigmoid_scaling * param_rescaling 
        distance_from_curve_sigmoid_scaling = distance_from_curve_sigmoid_scaling * param_rescaling

        ego_distance_along_curve, ego_distance_from_curve = wrapper_fcn([ego_x, ego_y])
        cur_cost = 0

        # get index of this time step 
        # round t / delta T to an int 
        t_index = floor(Int, (start_time + t) / ΔT) + 1
        closest_car_in_front = Inf # wonder if the inf causing issues, try just making it large? Inf

        car_in_front_loc = nothing
        for i = 1:k_closest_vehicles_for_cost
            vehicle_col = vehicle_index_to_col(i)
            vehicle_x, vehicle_y, vehicle_theta, vehicle_v, vehicle_a = demo[t_index, vehicle_col:vehicle_col + 4]
            vehicle_distance_along_curve, vehicle_distance_from_curve = wrapper_fcn([vehicle_x, vehicle_y])
            if isnothing(ego_distance_from_curve) || isnothing(vehicle_distance_from_curve) || isnothing(ego_distance_along_curve) || isnothing(vehicle_distance_along_curve)
                println("ego dist: ", ego_distance_from_curve)
                println("vehicle dist: ", vehicle_distance_from_curve)
                println("ego dist along: ", ego_distance_along_curve)
                println("vehicle dist along: ", vehicle_distance_along_curve)

                println("vehicle x: ", vehicle_x)
                println("vehicle y: ", vehicle_y) 
                println("ego x, y: ", [ForwardDiff.value(ego_x), ForwardDiff.value(ego_y)])
            end
            cur_distance = sqrt(scaling_distance_from_curve*(vehicle_distance_from_curve - ego_distance_from_curve)^2 + (vehicle_distance_along_curve - ego_distance_along_curve)^2)
            cur_cost += ego_weight_distance * (ego_comfortable_distance - cur_distance)^2 * sigmoid(comfortable_distance_sigmoid_scaling*(ego_comfortable_distance - cur_distance))  #relu(ego_comfortable_distance - cur_distance)^2

            diff_along_curve = vehicle_distance_along_curve - ego_distance_along_curve
            # tell if the vehicle is in front of you: it's ahead along the curve, is facing forward, and is within a road width of the plan (maybe will catch some that are at Ts depending on their angle?)
            if diff_along_curve > 0 && (abs((vehicle_theta - ego_phi) % (2*pi)) < pi/4) && vehicle_distance_from_curve < road_width_cm/scaling_factor 
                if diff_along_curve < closest_car_in_front
                    car_in_front_loc = [vehicle_x, vehicle_y]
                end
                closest_car_in_front = min(closest_car_in_front, diff_along_curve)
            end
        end



        # TODO: try sigmoid on the weight for v desired rather than on the desired velocity itself 
        v_desired_modified = ego_v_desired * sigmoid(desired_velocity_sigmoid_scaling * (closest_car_in_front - desired_velocity_sigmoid_threshold)) #ego_v_desired * sigmoid(desired_velocity_sigmoid_scaling * (closest_car_in_front - ego_comfortable_distance)) # desired_velocity_sigmoid_threshold)) # 1 / (1 + exp(-sigmoid_scaling*(closest_car_in_front - sigmoid_threshold))) * ego_v_desired 
        distance_from_curve_component = ego_weight_distance_from_curve * ego_distance_from_curve^2 * sigmoid(distance_from_curve_sigmoid_scaling*(ego_distance_from_curve - distance_from_curve_sigmoid_center))

        # --- up to here in converting to feature gen function 


        # tracking along the curve, velocity, and control costs 
        cur_cost += distance_from_curve_component + ego_weight_velocity * (ego_v - v_desired_modified)^2 + ego_weight_control_1 * u[1]^2 + ego_weight_control_2 * u[2]^2

        # account for progress by adding distance along curve at start and subtracting distance along curve at end. 
        cost_fcn_time_index = round(Int, t / ΔT)
        if cost_fcn_time_index == 0
            cur_cost += ego_weight_distance_along_curve * ego_distance_along_curve
        elseif cost_fcn_time_index == (game_horizon - 1)
            cur_cost -= ego_weight_distance_along_curve * ego_distance_along_curve # lower cost the farther you got 
        end

        return cur_cost
    end

    # The previous cost function but accumulating features as we go 
    # and returning that list of features and feature names

    function features_from_moment_in_horizon(g, x, u, t, i, x0)
        features = []
        feature_names = []

        # cost for ego vehicle
        ego_x, ego_y, ego_phi, ego_v, start_time = x[1:5]

        t_str = "t="*string(round(t, digits=3))*"_"

        # parameters 
        ego_v_desired, ego_weight_distance_along_curve, ego_weight_distance_from_curve, ego_weight_velocity, ego_weight_control_1,
        ego_weight_control_2, ego_comfortable_distance, scaling_distance_from_curve, comfortable_distance_sigmoid_scaling, ego_weight_distance,
        distance_from_curve_sigmoid_scaling, distance_from_curve_sigmoid_center, desired_velocity_sigmoid_threshold = x[6:18]
        
        param_rescaling = 10
        ego_weight_distance_along_curve = ego_weight_distance_along_curve * param_rescaling
        ego_weight_distance_from_curve = ego_weight_distance_from_curve * param_rescaling 
        ego_weight_velocity = ego_weight_velocity * param_rescaling
        scaling_distance_from_curve = scaling_distance_from_curve * param_rescaling 
        comfortable_distance_sigmoid_scaling = comfortable_distance_sigmoid_scaling * param_rescaling 
        distance_from_curve_sigmoid_scaling = distance_from_curve_sigmoid_scaling * param_rescaling

        ego_distance_along_curve, ego_distance_from_curve = wrapper_fcn([ego_x, ego_y])
        cur_cost = 0

        # add computation of info from x_0
        ego_x_0, ego_y_0, ego_phi_0, ego_v_0 = x0[1:4]
        ego_distance_along_curve_0, ego_distance_from_curve_0 = wrapper_fcn([ego_x_0, ego_y_0])


        # absolute coordinates 
        append!(features, [ego_x, ego_y, ego_phi, ego_v, ego_distance_along_curve, ego_distance_from_curve]) # 1-4
        append!(feature_names, ["ego_x_global", "ego_y_global", "ego_phi_global", "ego_v_global", "ego_distance_along_curve_global", "ego_distance_from_curve_global"])

        if t > 0.0
            # append relative features
            append!(features, [ego_x - ego_x_0, ego_y - ego_y_0, ego_phi - ego_phi_0, ego_v - ego_v_0, ego_distance_along_curve - ego_distance_along_curve_0, ego_distance_from_curve - ego_distance_from_curve_0]) # TODO: wrapping for angles? 
            append!(feature_names, ["ego_x_relative", "ego_y_relative", "ego_phi_relative", "ego_v_relative", "ego_distance_along_curve_local", "ego_distance_from_curve_local"])
        end

        # TODO: test how many iterations the actual cost will have, game_horizon or game_horizon + 1?

        # get index of this time step 
        # round t / delta T to an int 
        t_index = floor(Int, (start_time + t) / ΔT) + 1
        t_index_start_time = floor(Int, start_time / ΔT_data) + 1

        closest_car_in_front = Inf # wonder if the inf causing issues, try just making it large? Inf

        car_in_front_loc = nothing
        for i = 1:k_closest_vehicles_for_cost
            vehicle_col = vehicle_index_to_col(i)
            vehicle_x, vehicle_y, vehicle_theta, vehicle_v, vehicle_a = demo[t_index, vehicle_col:vehicle_col + 4]
            vehicle_distance_along_curve, vehicle_distance_from_curve = wrapper_fcn([vehicle_x, vehicle_y])
            
            # assert that we don't have any nans which means that this vehicle exists 
            @assert ~any(isnan.(demo[t_index, vehicle_col:vehicle_col + 4]))

            # adding in initial state 
            # TODO: this will be weird if which actual car corresponds to that # closest car changes over the horizon 
            # also, do we want distances relative to the ego vehicle? 
            vehicle_x_0, vehicle_y_0, vehicle_theta_0, vehicle_v_0, vehicle_a_0 = demo[t_index_start_time, vehicle_col:vehicle_col + 4]
            vehicle_distance_along_curve_0, vehicle_distance_from_curve_0 = wrapper_fcn([vehicle_x_0, vehicle_y_0])
                
            vehicle_str = "vehicle" * string(i) * "_"
            # Global state 
            append!(features, [vehicle_x, vehicle_y, vehicle_theta, vehicle_v, vehicle_a, vehicle_distance_along_curve, vehicle_distance_from_curve])
            append!(feature_names, vehicle_str .* ["x_global", "y_global", "theta_global", "v_global", "a_global", "distance_along_curve_global", "distance_from_curve_global"])

            # Local state
            if t > 0.0 
                append!(features, [vehicle_x - vehicle_x_0, vehicle_y - vehicle_y_0, vehicle_theta - vehicle_theta_0, vehicle_v - vehicle_v_0, vehicle_a - vehicle_a_0, vehicle_distance_along_curve - vehicle_distance_along_curve_0, vehicle_distance_from_curve - vehicle_distance_from_curve_0])
                append!(feature_names, vehicle_str .* ["x_relative", "y_relative" ,"theta_relative", "v_relative", "a_relative", "distance_along_curve_relative", "distance_from_curve_relative"])
            end 

            # also add the states relative to the ego state 
            ego_a = u[2] / velocity_control_scaling # TODO: check that this is legit, that I've gotten the scale right to compare with vehicle_a
            append!(features, [vehicle_x - ego_x, vehicle_y - ego_y, vehicle_theta - ego_phi, vehicle_v - ego_v, vehicle_a - ego_a, vehicle_distance_along_curve - ego_distance_along_curve, vehicle_distance_from_curve - ego_distance_from_curve]) 
            append!(feature_names, vehicle_str .* ["x_relative_to_ego", "y_relative_to_ego", "theta_relative_to_ego", "v_relative_to_ego", "a_relative_to_ego", "distance_along_curve_relative_to_ego", "distance_from_curve_relative_to_ego"])
            
            if isnothing(ego_distance_from_curve) || isnothing(vehicle_distance_from_curve) || isnothing(ego_distance_along_curve) || isnothing(vehicle_distance_along_curve)
                println("ego dist: ", ego_distance_from_curve)
                println("vehicle dist: ", vehicle_distance_from_curve)
                println("ego dist along: ", ego_distance_along_curve)
                println("vehicle dist along: ", vehicle_distance_along_curve)

                println("vehicle x: ", vehicle_x)
                println("vehicle y: ", vehicle_y) 
                println("ego x, y: ", [ForwardDiff.value(ego_x), ForwardDiff.value(ego_y)])
            end
            cur_distance = sqrt(scaling_distance_from_curve*(vehicle_distance_from_curve - ego_distance_from_curve)^2 + (vehicle_distance_along_curve - ego_distance_along_curve)^2)
            append!(features, [(vehicle_distance_from_curve - ego_distance_from_curve)^2, (vehicle_distance_along_curve - ego_distance_along_curve)^2, scaling_distance_from_curve*(vehicle_distance_from_curve - ego_distance_from_curve)^2 + (vehicle_distance_along_curve - ego_distance_along_curve)^2, cur_distance])
            append!(feature_names, vehicle_str .* ["distance_from_curve_relative_to_ego_squared", "distance_along_curve_relative_to_ego_squared", "weighted_distance_squared", "weighted_distance"])

            other_vehicle_sigmoid_term = sigmoid(comfortable_distance_sigmoid_scaling*(ego_comfortable_distance - cur_distance))
            cur_cost += ego_weight_distance * (ego_comfortable_distance - cur_distance)^2 * other_vehicle_sigmoid_term  #relu(ego_comfortable_distance - cur_distance)^2

            diff_along_curve = vehicle_distance_along_curve - ego_distance_along_curve
            # tell if the vehicle is in front of you: it's ahead along the curve, is facing forward, and is within a road width of the plan (maybe will catch some that are at Ts depending on their angle?)
            if diff_along_curve > 0 && (abs((vehicle_theta - ego_phi) % (2*pi)) < pi/4) && vehicle_distance_from_curve < road_width_cm/scaling_factor 
                if diff_along_curve < closest_car_in_front
                    car_in_front_loc = [vehicle_x, vehicle_y]
                end
                closest_car_in_front = min(closest_car_in_front, diff_along_curve)
                
                # binary feature if the vehicle is in front of you 
                append!(features, [1.0])
                append!(feature_names, vehicle_str .* ["in_front"])
            else 
                append!(features, [0.0])
                append!(feature_names, vehicle_str .* ["in_front"])
            end

            # Other vehicle distance features 
            append!(features, [(ego_comfortable_distance - cur_distance)^2, other_vehicle_sigmoid_term, (ego_comfortable_distance - cur_distance)^2 * other_vehicle_sigmoid_term])
            append!(feature_names, vehicle_str .* ["distance_from_comfortable_squared", "comfortable_distance_sigmoid_output", "comfortable_distance_full_term"])
        end        

        v_desired_modified = ego_v_desired * sigmoid(desired_velocity_sigmoid_scaling * (closest_car_in_front - desired_velocity_sigmoid_threshold)) #ego_v_desired * sigmoid(desired_velocity_sigmoid_scaling * (closest_car_in_front - ego_comfortable_distance)) # desired_velocity_sigmoid_threshold)) # 1 / (1 + exp(-sigmoid_scaling*(closest_car_in_front - sigmoid_threshold))) * ego_v_desired 

        # min vehicle distance 
        if closest_car_in_front == Inf
            append!(features, [0.0, 0.0, 0.0]) # TODO: is this legit to have it be 0 for the first one?
        else
            append!(features, [closest_car_in_front, sigmoid(desired_velocity_sigmoid_scaling * (closest_car_in_front - desired_velocity_sigmoid_threshold)), 1.0])
        end
        append!(feature_names, ["closest_car_in_front_gap", "comfortable_distance_sigmoid", "has_car_in_front"])

        distance_from_curve_component = ego_weight_distance_from_curve * ego_distance_from_curve^2 * sigmoid(distance_from_curve_sigmoid_scaling*(ego_distance_from_curve - distance_from_curve_sigmoid_center))

        append!(features, [ego_distance_from_curve^2, sigmoid(distance_from_curve_sigmoid_scaling*(ego_distance_from_curve - distance_from_curve_sigmoid_center)), distance_from_curve_component]) # TODO: double check fine to ignore the affine things here
        append!(feature_names, ["ego_distance_from_curve_squared", "ego_distance_from_curve_sigmoid", "distance_from_curve_term"])

        # tracking along the curve, velocity, and control costs 
        cur_cost += distance_from_curve_component + ego_weight_velocity * (ego_v - v_desired_modified)^2 + ego_weight_control_1 * u[1]^2 + ego_weight_control_2 * u[2]^2

        append!(features, [(ego_v - v_desired_modified)^2]) 
        append!(feature_names, ["ego_v_relative_to_desired_squared"])

        if i < game_horizon
            append!(features, [u[1]^2, u[2]^2, u[1], u[2]])
            append!(feature_names, ["u_1_squared", "u_2_squared", "u_1", "u_2"])
        else
            #print("final control: ", u) 
        end 

        # account for progress by adding distance along curve at start and subtracting distance along curve at end. 
        cost_fcn_time_index = round(Int, t / ΔT)
        if cost_fcn_time_index == 0
            cur_cost += ego_weight_distance_along_curve * ego_distance_along_curve
        elseif cost_fcn_time_index == (game_horizon - 1)
            cur_cost -= ego_weight_distance_along_curve * ego_distance_along_curve # lower cost the farther you got 
        end
        
        # no additional features from this bc the ego_distance_along_curve - ego_distance_along_curve_0 is already included at every time point 

        # add the t_str on 
        feature_names = t_str .* feature_names

        return features, feature_names, cur_cost
    end

    function feature_generation_function(state, θ, solver, g, index)
        # state is just indices 1 to 4 (x, y, phi, v)
        # TODO: figure out ordering / separation of other vehicles, depending on the side of the road that they're on
        # have state relative to the initial position as well? or both absolute and relative? 
        # TODO; figure out how to round the angles in the features 

        x0 = SVector(state..., (index-1) * ΔT)    
        extended_x0 = SVector{nx}(vcat(x0, θ))

        converged, expert_traj, strategies = solve(g, solver, extended_x0)
        predicted_trajectory = zeros(game_horizon, nx)
        for i = 1:nx
            predicted_trajectory[:, i] = [expert_traj.x[t][i] for t in 1:game_horizon]
        end

        predicted_controls = zeros(game_horizon, 2)
        for i = 1:2
            predicted_controls[:, i] = [expert_traj.u[t][i] for t in 1:game_horizon]
        end
        
        ts = [ΔT * (i-1) for i = 1:game_horizon]

        features = []
        feature_names = []

        # get the features over the time horizon 
        total_cost = 0.0
        for i = 1:game_horizon
            cur_features, cur_feature_names, cur_cost = features_from_moment_in_horizon(g, predicted_trajectory[i, :], predicted_controls[i, :], ts[i], i, extended_x0)
            append!(features, cur_features)
            append!(feature_names, cur_feature_names)
            total_cost += cur_cost
        end

        # add on the total cost as a feature 
        push!(features, total_cost)
        push!(feature_names, "total_cost")


        if converged == false
            println("index: ", index, " did not converge ")
            return zeros(size(features)), feature_names, false
        else
            return features, feature_names, true
        end 
    end

    return (FunctionPlayerCost(ego_vehicle_cost),), feature_generation_function
end

function loss(
    θ, 
    nominal_game,
    nominal_solver,
    demo,
    start_index, 
    obs_time_list, 
    obs_state_list, 
    obs_control_list,  
    debug_mode=false
    ) 
    # don't include the extra cost params in x0 yet 
    x0 = SVector(demo[start_index, 1:4]..., (start_index-1) * ΔT)
    expert_traj = demo[start_index:start_index + game_horizon - 1, :]

    extended_x0 = SVector{nx}(vcat(x0, θ)) # where we concatenate the cost parameters (with dual) to the initial state
    purified_x0 = SVector{nx}(vcat(x0, ForwardDiff.value.(θ))) # where we remove the dual number associated with the cost parameters
    # solve the nominal game under the current θ:
    nominal_converged, nominal_traj, nominal_strategies = solve(nominal_game, nominal_solver, purified_x0)

    if ~nominal_converged
        println("solver didn't converge")
        return -1
    end

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
    
    loss_value = sum([(traj.x[t][i] - expert_traj[t, i])^2 for t in 1:game_horizon for i = 1:2]) / game_horizon # only sum the positions 

    # just add the first step's loss
    # loss_value = sum((traj.x[2][1:2] - expert_traj[2][1:2]).^2)

    # expert_data = transpose(mapreduce(permutedims, vcat, Vector([Vector(expert_traj.u[t][obs_control_list]) for t in obs_time_list])))
    # prediction = transpose(mapreduce(permutedims, vcat, Vector([Vector(traj.u[t][obs_control_list]) for t in obs_time_list])))
    # loss_value = norm(expert_data - prediction)^2
    if debug_mode
        println("expert_data: ", expert_data)
        println("prediction: ", prediction)
        println("loss_value: ", loss_value)
        return loss_value, expert_data, prediction, nominal_converged, nominal_traj, traj, nominal_strategies, local_strategies
    else
        return loss_value
    end
end

# to run: loss(cur_θ, games[53], solvers[53], scaled_demonstration_states[53], 1488, 1:game_horizon, 1:nx, 1:nu)

function batch_loss(
    θ, 
    nominal_games,
    nominal_solvers,
    demos,
    demo_indices,
    start_indices,
    obs_time_list, 
    obs_state_list, 
    obs_control_list,  
    debug_mode=false
    )

    total = 0.0
    num_converged = 0 
    for (demo_index, start_index) in zip(demo_indices, start_indices)
        # println("demo index: ", demo_index, " sample: ", start_index)
        cur_loss = loss(θ, nominal_games[demo_index], nominal_solvers[demo_index], demos[demo_index], start_index, obs_time_list, obs_state_list, obs_control_list, debug_mode)
        if cur_loss != -1 
            total += cur_loss 
            num_converged += 1 
        end
    end
    println(num_converged, " converged out of ", length(start_indices))
    return total / num_converged 
end 
