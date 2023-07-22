using iLQGames
using StaticArrays
using LinearAlgebra


game_horizon = 10
ΔT = 0.1
A_single = @SMatrix([1.0 1.0; 0.0 1.0])
B_single = @SMatrix([1.0; 1.0])
x0 = SVector(1.0, 1.0, 1.0, 1.0)

# construct joint system from single-player matrices
# Note: I guess what bit you here is taht `ProductSystem` does not currently work
# for LinearSystem because linear systems skip linearization to save compute  and thus the
# `ProductSystem` type is forwarded to the inner solver loop which only supports LTISystem and
# LTVSystem.
joint_dynamics = let
    # creating a block-diagonal matrix for A and B via the kronecker product
    A_joint = SMatrix{4,4}(kron(I(2), A_single))
    B_joint = SMatrix{4,2}(kron(I(2), B_single))
    # this inner linear system type is unfortunately still needed for historical reasons
    # toe make the life of the solver a bit easier, we can directly provide the discretized
    # version of the dynamics (rather than having it discretize the system internally)
    raw_system = iLQGames.LinearSystem{ΔT}(A_joint, B_joint)
    # xyindex is just needed for visualization; we could also fill in nothing but doing it
    # properly here now
    xyindex = SVector(1, 2, 3, 4)
    LTISystem(raw_system, xyindex)
end

# Construc the multi-player version of the dynamics with the linear system above for each player
# as a subsystem
# create a tuple of cost functions for each player. Here just a trivial place holder
costs = (
    iLQGames.FunctionPlayerCost((g, x, u, t) -> sum(x .^ 2) + sum(u .^ 2)),
    iLQGames.FunctionPlayerCost((g, x, u, t) -> sum(x .^ 2) + sum(u .^ 2))
)

# indices of inputs that each player controls
player_inputs = (SVector(1), SVector(2))
game = iLQGames.GeneralGame(game_horizon, player_inputs, joint_dynamics, costs)

solver = iLQGames.iLQSolver(game)
converged, trajectory, strategies = iLQGames.solve(game, solver, x0)

