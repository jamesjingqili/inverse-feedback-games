function vehicle_index_to_col(index)
    return 6 + 5*(index - 1)
end

function pedestrian_index_to_col(index, k_closest_vehicles)
    return 6 + 5*k_closest_vehicles + 5*(index - 1)
end

function centerline_col(k_closest_vehicles, k_closest_pedestrians)
    return 6 + 5*k_closest_vehicles + 5*k_closest_pedestrians
end

function relu(x)
    return max.(0, x)
end

function rescale_data(centerlines, demonstration_states, train_point_sequences, road_width, k_closest_vehicles, k_closest_pedestrians; scaling_factor=2000)

    # Rescale the map
    scaled_centerlines = [LineSegment(centerline.p / scaling_factor, centerline.q / scaling_factor) for centerline in centerlines]

    # Rescale the train demonstration states 
    indices_to_scale = [1, 2, 4, 5] # x, y, v, a
    for j = 1:k_closest_vehicles
        cur_col = vehicle_index_to_col(j)
        append!(indices_to_scale, [cur_col, cur_col + 1, cur_col + 3, cur_col + 4])
    end 
    centerline_index = centerline_col(k_closest_vehicles, k_closest_pedestrians)
    append!(indices_to_scale, [centerline_index, centerline_index + 1, centerline_index + 2, centerline_index + 3])

    scaled_demonstration_states = deepcopy(demonstration_states)
    for i = 1:length(demonstration_states)
        scaled_demonstration_states[i][:, indices_to_scale] /= scaling_factor
    end

    # Scale the train_point_sequences 
    scaled_train_point_sequences = train_point_sequences / scaling_factor

    # Scale the road width
    scaled_road_width = road_width / scaling_factor

    return scaled_centerlines, scaled_demonstration_states, scaled_train_point_sequences, scaled_road_width
end


function road_shift_from_points(point1, point2, road_width)
    # Since we're using a left handed coordinate
    # if we're going in the positive x direction (left), the road shift is up in y 
    # if we're going in the negative x direction (right), the road shift is down in y 
    # if we're going in the positive y direction (up), the road shift is negative in x (right)
    # if we're going in the negative y direction (down), the road shift is positive in y (left)
    start_x, start_y, second_x, second_y = point1[1], point1[2], point2[1], point2[2]
    # assert x is constant or y is constant 
    @assert (start_x ≈ second_x || start_y ≈ second_y)

    if start_x < second_x
        road_shift = [0, road_width / 4]
    elseif start_x > second_x
        road_shift = [0, -road_width / 4]
    elseif start_y < second_y
        road_shift = [-road_width / 4, 0]
    elseif start_y > second_y
        road_shift = [road_width / 4, 0]
    end
    return road_shift
end

function shorten_segment(segment::LineSegment, dist_to_cut, direction="both")
    p, q = segment.p, segment.q
    v = q - p
    v = v / norm(v)
    if direction == "both"
        return LineSegment(p + v * dist_to_cut, q - v * dist_to_cut)
    elseif direction == "start"
        return LineSegment(p + v * dist_to_cut, q)
    elseif direction == "end"
        return LineSegment(p, q - v * dist_to_cut)
    else 
        error("Invalid direction")
    end
end 

function is_right_turn(direction_1, direction_2)
    # in left-handed coordinates, check to see if the turn is right. 
    # direction_1 should be the direction you start heading, and direction 2 should be the direction you end up heading 
    # if the cross product of the two vectors is positive, then it's a right turn
    return cross([direction_1; 0], [direction_2; 0])[3] > 0
end

function trajectory_plan_from_key_points(key_points, road_width)
    # go along alternating between straights and arcs, is that always right? 
    # we'll note that we're in a left handed coordinate system, so left is more positive xs and right is 
    trajectory_segments = []
    
    # From the first two points get a line 
    point1, point2 = key_points[1], key_points[2]
    road_shift = road_shift_from_points(point1, point2, road_width)
    line_segment = LineSegment(point1 + road_shift, point2 + road_shift)
    # shorten the end of the line segment by 0.5*road_width if there are more points that we'll be adding on
    # (otherwise we just have a single road segment which is fine to keep unshortened)
    if length(key_points) > 2  
        line_segment = shorten_segment(line_segment, 0.5*road_width, "end")
    end
    push!(trajectory_segments, line_segment)

    # now for each trio we'll add a curve then a straight 
    for i = 1:(length(key_points) - 2)
        point1, point2, point3 = key_points[i], key_points[i+1], key_points[i+2]

        # get the vector from point1 to point2
        direction_1 = (point2 - point1) / norm(point1 - point2)
        # get the vector from point2 to point3
        direction_2 = (point3 - point2) / norm(point3 - point2)

        # add on the segment from point 2 to point 3 
        road_shift = road_shift_from_points(point2, point3, road_width)
        line_segment = LineSegment(point2 + road_shift, point3 + road_shift)
        # shorten line segment on both sides by 0.5*road_width unless we're on the last index
        # in which case we just shorten the start and not the end  
        if i == length(key_points) - 2
            line_segment = shorten_segment(line_segment, 0.5*road_width, "start")
        else 
            line_segment = shorten_segment(line_segment, 0.5*road_width, "both")
        end

        right_turn = is_right_turn(-direction_1, direction_2)
        center_shift_amount = road_width / 2 # shift from center or road point to the center of circle    #right_turn ? road_width / 4 : road_width / 2 # 
        circle_center = point2 - center_shift_amount * direction_1 + center_shift_amount * direction_2 # point2 - center_shift_amount * direction_1 + center_shift_amount * direction_2
        
        # now see the angle from the end of the previous line segment to this circle 
        previous_line_segment = trajectory_segments[end]

        previous_end = previous_line_segment.q
        start_angle = atan(previous_end[2] - circle_center[2], previous_end[1] - circle_center[1])
        
        # now see the angle from the start of the next line segment to this circle 
        next_line_segment = line_segment
        next_start = next_line_segment.p
        end_angle = atan(next_start[2] - circle_center[2], next_start[1] - circle_center[1])

        clockwise = right_turn 
        circle_radius = right_turn ? road_width * 3/4 : road_width / 4# in left-handed coordinate system right turns have larger radius 
        arc = Arc(circle_center, circle_radius, start_angle, end_angle, clockwise)
        
        push!(trajectory_segments, arc)
        push!(trajectory_segments, line_segment)
    end
    
    return ListTrajectoryPlan(trajectory_segments)
end 

py"""
import pickle
 
def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data
"""

load_pickle = py"load_pickle"

