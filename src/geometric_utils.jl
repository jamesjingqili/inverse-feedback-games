# Abstract TrajectoryPlan type which 
# we want to be able to get distance from and distance along 
abstract type TrajectoryPlan end

"""
    Arc setup and Arc specific functions
"""
# These are directional with respect to calculating progress 
struct Arc <: TrajectoryPlan
    center
    radius
    start_angle
    end_angle
    clockwise::Bool

    function Arc(center, radius, start_angle, end_angle, clockwise::Bool=false)
        # Make sure the angles are between 0 and 2pi
        start_angle = mod(start_angle, 2*pi)
        end_angle = mod(end_angle, 2*pi)
    
        return new(center, radius, start_angle, end_angle, clockwise)
    end
end

function ordered_start_end_angle(start_angle, end_angle, clockwise)
# Makes the angles in the range [0, 4*pi) and orders them so that if you did a linspace 
# from start_angle to end_angle, you would get the arc.

# if you're going clockwise then start angle will be larger than end angle 
# if you're going counter clockwise then start angle will be smaller than end angle 
    if clockwise
        if start_angle < end_angle
            start_angle += 2 * pi
        end
    else
        if end_angle < start_angle
            end_angle += 2 * pi
        end
    end
    return start_angle, end_angle
end

function is_angle_between(angle, start_angle, end_angle, clockwise)
    # all angles must be between [0, 2pi)
    if clockwise
        if start_angle >= end_angle
            return start_angle >= angle >= end_angle
        else
            return angle <= start_angle || angle >= end_angle
        end
    else
        if start_angle <= end_angle
            return start_angle <= angle <= end_angle
        else
            return angle >= start_angle || angle <= end_angle
        end
    end
end


function sampled_projection(point, arc::Arc; num_samples::Int=1000)
    start_angle, end_angle = ordered_start_end_angle(arc.start_angle, arc.end_angle, arc.clockwise)
    angles = LinRange(arc.start_angle, arc.end_angle, num_samples)
    points = [arc.center .+ arc.radius * [cos(angle), sin(angle)] for angle in angles]

    # project the point onto the arc
    distances = [norm(point - p) for p in points]
    min_distance, min_index = findmin(distances)

    return points[min_index]
end

function segment_length(arc::Arc)
    start_angle, end_angle = ordered_start_end_angle(arc.start_angle, arc.end_angle, arc.clockwise)
    return arc.radius * abs(end_angle - start_angle)
end

function contains(arc::Arc, point)
    angle = mod(atan(point[2] - arc.center[2], point[1] - arc.center[1]), 2*pi)
    return is_angle_between(angle, arc.start_angle, arc.end_angle, arc.clockwise)
end

function project(point, arc::Arc)
    # find the angle of the point with respect to the center of the arc
    angle = mod(atan(point[2] - arc.center[2], point[1] - arc.center[1]), 2*pi)
    start_angle, end_angle = arc.start_angle, arc.end_angle

    # now, check if the angle is within the arc. if it is, then the projected point 
    # is in the interior of the arc. otherwise, it's one of the edge points. 
    if is_angle_between(angle, start_angle, end_angle, arc.clockwise)
        projected_point = arc.center .+ arc.radius * [cos(angle), sin(angle)]
    else
        start_vertex = arc.center .+ arc.radius * [cos(start_angle), sin(start_angle)]
        end_vertex = arc.center .+ arc.radius * [cos(end_angle), sin(end_angle)]
        distance_start_vertex = norm(point - start_vertex)
        distance_end_vertex = norm(point - end_vertex)
        if distance_start_vertex < distance_end_vertex
            projected_point = start_vertex
        else
            projected_point = end_vertex
        end
    end 

    return projected_point
end 

function progress(point, arc::Arc)
    # Point should be on the arc TODO: assert that  
    angle = mod(atan(point[2] - arc.center[2], point[1] - arc.center[1]), 2*pi)
    if arc.clockwise 
        if angle > arc.start_angle 
            # then it will go all the way down to 0, then jump to 2pi and continue down until it hits it 
            angle_change = arc.start_angle + (2*pi - angle)
        else 
            angle_change = arc.start_angle - angle
        end
    else 
        if angle < arc.start_angle 
            # then it will go all the way up to 2pi, jump to 0, then continue until it hits it 
            angle_change = 2*pi - arc.start_angle + angle
        else
            angle_change = angle - arc.start_angle
        end
    end
    return angle_change * arc.radius

    # angle = mod(atan(point[2] - arc.center[2], point[1] - arc.center[1]), 2*pi)
    # return mod(angle - arc.start_angle, 2*pi) * arc.radius
end

function point_from_progress(progress, arc::Arc)
    angle_change = progress / arc.radius
    angle = arc.start_angle + sign(arc.end_angle - arc.start_angle) * angle_change
    return arc.center + arc.radius .* [cos(angle), sin(angle)]
end

function visualize(arc::Arc; kwargs...)
    start_angle, end_angle = arc.start_angle, arc.end_angle
    if arc.clockwise
        # have the values decreasing, so get it so that end_angle < start_angle. 
        if end_angle > start_angle
            end_angle = end_angle - 2*pi
        end
    else 
        if end_angle < start_angle 
            end_angle = end_angle + 2*pi
        end
    end
    angles = LinRange(start_angle, end_angle, 100)
    points = [arc.center .+ arc.radius * [cos(angle), sin(angle)] for angle in angles]

    return plot!([p[1] for p in points], [p[2] for p in points]; label="", kwargs...)
end

"""
    Line segment specific functions
"""

function segment_length(line::LineSegment)
    return norm(line.q - line.p)
end

function contains(line::LineSegment, point)
    # Check if the point is on the line segment 
    return norm(point - line.p) + norm(point - line.q) ≈ segment_length(line)
end

function project(point, line::LineSegment)
    # Project the point onto the line segment 
    v = line.q - line.p
    w = point - line.p
    c = dot(w, v) / dot(v, v)

    # if c is less than 0, then the closest point is the start point
    if c <= 0
        return line.p
    end

    # if c is greater than 1, then the closest point is the end point
    if c >= 1
        return line.q
    end

    return line.p + c * v
end

function progress(point, line::LineSegment)
    # Point should be on the line segment TODO: assert that 
    return norm(point - line.p) # distance from the start point 
end

function visualize(l::LineSegment; kwargs...)
    return plot!(l; label="", kwargs...)
end

"""
    Trajectory plan that contains several subtrajectories 
"""
struct ListTrajectoryPlan <: TrajectoryPlan
    subtrajectories
end

function segment_length(trajectory::ListTrajectoryPlan)
    return sum([segment_length(subtrajectory) for subtrajectory in trajectory.subtrajectories])
end

function contains(trajectory::ListTrajectoryPlan, point)
    for subtrajectory in trajectory.subtrajectories
        if contains(subtrajectory, point)
            return true
        end
    end
    return false
end

function project(point, trajectory::ListTrajectoryPlan)
    # Project the point onto each subtrajectory and return the closest one 
    projected_points = [project(point, subtrajectory) for subtrajectory in trajectory.subtrajectories]
    distances = [norm(point - projected_point) for projected_point in projected_points]
    min_distance, min_index = findmin(distances)

    return projected_points[min_index]
end

function progress(point, trajectory::ListTrajectoryPlan)
    accumulated_progress = 0
    for subtrajectory in trajectory.subtrajectories
        if contains(subtrajectory, point)
            return accumulated_progress + progress(point, subtrajectory)
        else
            accumulated_progress += segment_length(subtrajectory)
        end 
    end
    println("reached end, shouldn't reach here, point: ", point)
end

function visualize(trajectory::ListTrajectoryPlan; kwargs...)
    for subtrajectory in trajectory.subtrajectories
        visualize(subtrajectory; kwargs...)
    end
    return plot!()
end

"""
    Generic functions for any of our types of sets
"""
# For this to work for a set, need project(point, set) and progress(point, set) to be defined.
function project_frenet_coordinates(point, set)
    projected_point = project(point, set)

    # Distance along curve is the first new coordinate (the progress)
    distance_along_curve = progress(projected_point, set)

    # Distance from the curve is the second new coordinate 
    distance_from_curve = norm(point - projected_point)

    return distance_along_curve, distance_from_curve, projected_point 
end

function distance_and_projection(point, set)
    projected_point = project(point, set)
    distance = norm(point - projected_point)
    return distance, projected_point
end 
