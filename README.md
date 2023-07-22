# inverse-iLQGames
This repo contains the essential codes for the following paper:

[Cost Inference for Feedback Dynamic Games from Noisy Partial State Observations and Incomplete Trajectories](https://arxiv.org/pdf/2301.01398.pdf), by Jingqi Li, Chih-Yuan Chiu, Lasse Peters, Somayeh Sojoudi, Claire Tomlin, and David Fridovich-Keil, AAMAS 2023

Julia version:  1.6.6


First thing to do:
1. cd to /src
2. run: "julia --project"
3. within Julia REPL terminal interface, run "import Pkg; Pkg.instantiate()"
4. copy the codes in "/examples/inverse_2_player_dubins_car.jl" to the Julia REPL terminal interface and wait for results


Important files:
1. experiment_utils.jl: we defined the loss in the function loss(). We defined the alternating gradient descent of theta and x0 in the functions inverse_game_gradient_descent_with_x0();
2. diff_solver.jl: we removed the type checking in the LQ game solver in iLQGames.jl such that ForwardDiff.jl can evaluate gradient properly. 
