function visualize_road_segment(line_segment, road_width; color="gray", kwargs...)
    p, q = line_segment.p, line_segment.q
    v = p - q
    v = v / norm(v)

    perp = [0 -1; 1 0] * v
    rectangle_p1 = p + road_width / 2 * perp
    rectangle_p2 = p - road_width / 2 * perp
    rectangle_p3 = q - road_width / 2 * perp
    rectangle_p4 = q + road_width / 2 * perp
    plot!(polygon_from_points([rectangle_p1, rectangle_p2, rectangle_p3, rectangle_p4]); color=color, label="", kwargs...)
    
    # plot a yellow dashed line along the center and white lines along the edges 
    return plot!([p[1], q[1]], [p[2], q[2]], color="yellow", label="")
end

function visualize_map(centerlines, road_width=1200; kwargs...)
    for centerline in centerlines
        visualize_road_segment(centerline, road_width; kwargs...)
    end
    return plot!()
end

rectangle(w, h, x, y) = Shape(x + [0,w,w,0], y + [0,0,h,h])
polygon_from_points(points) = Shape([p[1] for p in points], [p[2] for p in points])

function visualize_road_from_key_points(key_points, road_width; kwargs...)
    # between each key point draw a rectangle of width road_width 
    for i in 1:length(key_points) - 1
        p1 = key_points[i]
        p2 = key_points[i + 1]

        v = p2 - p1
        v = v / norm(v)

        # extend line to be road_width / 2 longer 
        # except for the first and last lines
        if i > 1
            p1 = p1 + road_width / 2 * v
        end
        if i < length(key_points) - 1
            p2 = p2 + road_width / 2 * v
        end

        visualize_road_segment(LineSegment(p1, p2), road_width; kwargs...)
    end
    return plot!(aspect_ratio=:equal)
end

function plot_car_at_state(x, y, heading, velocity; width=185/scaling_factor, height=470/scaling_factor, color="black", kwargs...)
    # plot a rectangle at x, y, with given heading 

    front_right = [x + height / 2 * cos(heading) + width / 2 * cos(pi/2 - heading), y + height / 2 * sin(heading) - width / 2 * sin(pi/2 - heading)]
    front_left = [x + height / 2 * cos(heading) - width / 2 * cos(pi/2 - heading), y + height / 2 * sin(heading) + width / 2 * sin(pi/2 - heading)]
    back_left = [x - height / 2 * cos(heading) - width / 2 * cos(pi/2 - heading), y - height / 2 * sin(heading) + width / 2 * sin(pi/2 - heading)]
    back_right = [x - height / 2 * cos(heading) + width / 2 * cos(pi/2 - heading), y - height / 2 * sin(heading) - width / 2 * sin(pi/2 - heading)]
    
    # plot rectangle 
    plot!(polygon_from_points([front_right, front_left, back_left, back_right]), color=color, label="", kwargs...)
    # plot arrow from center in direction of heading 
    arrow_length = height*1.5
    arrow_delta_x = arrow_length * cos(heading)
    arrow_delta_y = arrow_length * sin(heading)
    #return quiver!([x], [y], quiver=([arrow_delta_x], [arrow_delta_y]), color=color, label="", kwargs...)
    #return plot!([x + arrow_delta_x, x], [y + arrow_delta_y, y], color=color, arrow=false, label="", kwargs...)
    return plot!()
end

function animate_trajectories(trajectories, colors, labels, demonstration, trajectory_plan, road_width, centerlines; time_spacing=5, field_of_view=5000, k_closest_vehicles=15, kwargs...)
    """
    trajectories: list of trajectories, each of which are a num_times x 4 array with x, y, heading, v
    """
    ego_states = demonstration[:, 1:4]

    if length(trajectories) > 0
        length_to_animate = min(minimum([size(trajectory, 1) for trajectory in trajectories]), size(demonstration, 1))
    else
        length_to_animate = size(demonstration, 1)
    end

    # plot the vehicles from the trajectories
    anim = @animate for t = 1:time_spacing:length_to_animate
        plot(aspect_ratio=:equal, xflip=true)
        visualize_map(centerlines, road_width)
        # plot the trajectories 
        if length(trajectories) > 0 
            for (i, trajectory) in enumerate(trajectories)
                x, y, heading, v = trajectory[t, :]
                color = colors[i]
                label = labels[i]
                plot_car_at_state(x, y, heading, v, color=color)
                # also plot their trajectory so far
                plot!(trajectory[1:t, 1], trajectory[1:t, 2], color=color, label=label)
            end
        end

        # visualize the trajectory plan 
        visualize(trajectory_plan, color="black")

        # plot the ground truth 
        x, y, heading, v = ego_states[t, :]
        plot_car_at_state(x, y, heading, v, color="blue")
        plot!(ego_states[1:t, 1], ego_states[1:t, 2], color="blue", label="ego ground truth")

        # plot the other vehicles 
        for i in 1:k_closest_vehicles
            other_x, other_y, other_heading, other_v = demonstration[t, vehicle_index_to_col(i):vehicle_index_to_col(i) + 3]
            plot_car_at_state(other_x, other_y, other_heading, other_v, color="black")
        end

        # set x limit based on the ego vehicle
        xlims = [x - field_of_view / 2, x + field_of_view / 2]
        ylims = [y - field_of_view / 2, y + field_of_view / 2]
        plot!(xlim=xlims, ylim=ylims)

        # add text label on the side showing t rounded to 2 digits  
        text_label = "t = $(round(t/15, digits=1))"
        annotate!(xlims[1], ylims[2], text_label)
    end
    return anim
end


function rollout_trajectory(game, solver, dynamics, demo, start_index; θ = nothing, theta_augmented = true, num_steps = 100)
    """
        game must have been built with a cost function that uses the appropriate demo
    """
    cur_x = SVector(demo[start_index, 1:4]..., (start_index - 1) * ΔT)
    if theta_augmented
        cur_x = SVector{5 + length(θ)}(vcat(cur_x, θ))
    end

    xs = [SVector{length(cur_x)}(zeros(length(cur_x))) for i = 1:(num_steps+1)]
    xs[1] = cur_x
    for t = 1:num_steps
        c, expert_traj, strategies = solve(game, solver, xs[t])
        if c == false 
            print("failed at timestep ", t, " ")
            println("Solve failed, gave u: ", expert_traj.u[1])
            return nothing
        end
        
        first_u = expert_traj.u[1]
        xs[t+1] = xs[t] +  ΔT * dx(dynamics, xs[t], first_u, 0.)
        # update the last state, which is a slack variable to deal with the global time 
        time_update = zeros(size(xs[t+1]))
        time_update[5] = ΔT
        xs[t+1] = xs[t + 1] + time_update
    end
    
    # pull out just the state from the trajectory 
    predicted_trajectory = zeros(num_steps + 1, 4)
    predicted_trajectory[:, 1] = [xs[t][1] for t = 1:num_steps + 1]
    predicted_trajectory[:, 2] = [xs[t][2] for t = 1:num_steps + 1]
    predicted_trajectory[:, 3] = [xs[t][3] for t = 1:num_steps + 1]
    predicted_trajectory[:, 4] = [xs[t][4] for t = 1:num_steps + 1]
    return predicted_trajectory
end

function trajectory_from_solver(game, solver, demo, start_index, θ=nothing, theta_augmented = true)
    x_0 = SVector(demo[start_index, 1:4]..., (start_index - 1) * ΔT)
    if theta_augmented
        x_0 = SVector{5 + length(θ)}(vcat(x_0, θ))
    end
    return trajectory_from_solver(game, solver, x_0)
end

function trajectory_from_solver(game, solver, x0)
    """ Just a convenient wrapper to get out the solver's converged trajectory in an easy form to visualize """ 
    c, expert_traj, strategies = solve(game, solver, x0)
    if c == false
        print("did not converge!")
        return nothing
    end

    game_horizon = length(expert_traj.x)
    predicted_trajectory = zeros(game_horizon, 4)
    predicted_trajectory[:, 1] = [expert_traj.x[t][1] for t in 1:game_horizon]
    predicted_trajectory[:, 2] = [expert_traj.x[t][2] for t in 1:game_horizon]
    predicted_trajectory[:, 3] = [expert_traj.x[t][3] for t in 1:game_horizon]
    predicted_trajectory[:, 4] = [expert_traj.x[t][4] for t in 1:game_horizon]

    return predicted_trajectory 
end


