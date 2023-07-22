# inverse-iLQGames
This repo contains the essential codes for the following paper:

Li, Jingqi, Chih-Yuan Chiu, Lasse Peters, Somayeh Sojoudi, Claire Tomlin, and David Fridovich-Keil. "Cost Inference for Feedback Dynamic Games from Noisy Partial State Observations and Incomplete Trajectories." arXiv preprint arXiv:2301.01398 (2023).

Julia version:  1.6.6


First thing to do:
1. cd to /src
2. run: "julia --project"
3. within Julia REPL terminal interface, run "import Pkg; Pkg.instantiate()"
4. copy the codes in "/examples/inverse_2_player_dubins_car.jl" to the Julia REPL terminal interface and wait for results


Important files:
1. experiment_utils.jl: we defined the loss and the gradient descent in the functions loss() and inverse_game_gradient_descent(), respectively;
2. diff_solver.jl: we removed the type checking in the LQ game solver in iLQGames.jl such that ForwardDiff.jl can evaluate gradient properly. 
