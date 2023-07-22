module iLQGames
    using DocStringExtensions
    using Reexport
    using Parameters
    using ForwardDiff
    using DiffResults
    @reexport using StaticArrays
    using LinearAlgebra
    using Parameters
    using Colors, ColorSchemes
    using LaTeXStrings
    using Plots
    gr()

    import Base:
        +,
        getindex,
        setindex!,
        zero,
        copy


    # game setup
    export ControlSystem, LTISystem, LTVSystem
    export PlayerCost, QuadraticPlayerCost, FunctionPlayerCost
    export AffineStrategy
    # games solution
    export GeneralGame
    export iLQSolver, solve, solve!, cost
    # export inverse_game_loss, inverse_game_gradient, inverse_game_gradient_descent
    export solve_lq_game_OLNE_KKT!, solve_lq_game_OLNE_with_costate!, solve_lq_game_FBNE_KKT!, solve_lq_game_FBNE_with_costate!
    # visualization
    export plot_traj, plot_states, plot_inputs, @animated
    export trajectory!

    # some macro sugar to make life easier
    include("sugar.jl")
    include("performance.jl")

    # game utils
    include("strategy.jl")
    include("system_trajectory.jl")
    include("player_cost.jl")

    # interface for feedback linearizable system
    include("flat.jl")
    # dynamics abstraction
    include("control_system.jl")
    include("linear_system.jl")
    include("product_system.jl")
    # game abstraction
    include("game.jl")
    include("prealloc.jl")
    include("game_impl.jl")
    # ad quadraticization
    include("quadraticize.jl")

    # the solver implementations
    include("solve_lq_game_OLNE.jl")
    include("solve_lq_game_FBNE.jl")
    include("solve_lq_game_OLNE_KKT.jl")
    include("solve_lq_game_FBNE_KKT.jl")
    include("solve_lq_game_OLNE_with_costate.jl")
    include("solve_lq_game_FBNE_with_costate.jl")
    # include("solve_lq_game_Stackelberg.jl")
    include("ilq_solver.jl")

    # simulation
    include("sim.jl")

    # some handy tools for problem description
    include("cost_design_utils.jl")

    # some dynamical systems to work with
    include("lorenz.jl")
    include("unicycle_4D.jl")
    include("n_player_navigation_game.jl")
    include("n_player_unicycle_game.jl")
    include("n_player_2Ddoubleintegrator_cost.jl")

    # some helpful tooling
    include("plot_utils.jl")
    include("test_utils.jl")


    # inverse game solver
    include("inverse_game_solver.jl")

end # module
