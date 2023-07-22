using ProgressBars
using Infiltrator
using Distributions
"
Important functions:
loss()
inverse_game_gradient_descent()
inverse_game_gradient_descent_with_x0()


Note that the 'inverse_game_gradient_descent() is an older 
version of 'inverse_game_gradient_descent_with_x0()', where the former one does not
support taking gradient descent in x0. The former one only takes gradient descent for θ.


The later one takes alternating gradient descent in θ and x0.


"


function loss(θ, dynamics, equilibrium_type, expert_traj, gradient_mode = true, specified_solver_and_traj = false, 
                nominal_solver=[], nominal_traj=[], obs_time_list = 1:game_horizon-1, 
                obs_state_list = 1:nx, obs_control_list = 1:nu, no_control=false, x0_mode=false, x0=[]) 
    if x0_mode==false
        x0 = first(expert_traj.x)
    end
    tmp1 = transpose(mapreduce(permutedims, vcat, Vector([Vector(expert_traj.x[t][obs_state_list]) for t in obs_time_list])))
    

    if gradient_mode == false
        nominal_game = GeneralGame(game_horizon, player_inputs, dynamics, parameterized_cost(ForwardDiff.value.(θ)))
        nominal_solver = iLQSolver(nominal_game, max_scale_backtrack=10, max_elwise_diff_step=Inf, equilibrium_type=equilibrium_type)
        nominal_converged, nominal_traj, nominal_strategies = solve(nominal_game, nominal_solver, x0)
        tmp2 = transpose(mapreduce(permutedims, vcat, Vector([Vector(nominal_traj.x[t][obs_state_list]) for t in obs_time_list])))    
        loss_value = norm(tmp2 - tmp1)^2
        return loss_value, nominal_traj, nominal_strategies, nominal_solver
    else
        if x0_mode==true
            if specified_solver_and_traj == false
                nominal_game = GeneralGame(game_horizon, player_inputs, dynamics, parameterized_cost(ForwardDiff.value.(θ)))
                nominal_solver = iLQSolver(nominal_game, max_scale_backtrack=10, max_elwise_diff_step=Inf, equilibrium_type=equilibrium_type)
                nominal_converged, nominal_traj, nominal_strategies = solve(nominal_game, nominal_solver, SVector{length(x0)}(ForwardDiff.value.(x0)))
            end
            costs = parameterized_cost(θ)
            game = GeneralGame(game_horizon, player_inputs, dynamics, costs)
            lqg = Differentiable_Solvers.lq_approximation(game, nominal_traj, nominal_solver)
            if equilibrium_type=="OLNE_KKT" || equilibrium_type=="OLNE_costate" || equilibrium_type=="OLNE"
                traj = Differentiable_Solvers.trajectory(x0, game, Differentiable_Solvers.solve_lq_game_OLNE(lqg), nominal_traj)
            elseif equilibrium_type=="FBNE_KKT" || equilibrium_type=="FBNE_costate" || equilibrium_type=="FBNE"
                traj = Differentiable_Solvers.trajectory(x0, game, Differentiable_Solvers.solve_lq_game_FBNE(lqg), nominal_traj)
            else
                @warn "equilibrium_type is wrong!"
            end
            tmp2 = transpose(mapreduce(permutedims, vcat, Vector([Vector(traj.x[t][obs_state_list]) for t in obs_time_list])))
            loss_value = norm(tmp1-tmp2)^2
            return loss_value
        else
            if specified_solver_and_traj == false
                nominal_game = GeneralGame(game_horizon, player_inputs, dynamics, parameterized_cost(ForwardDiff.value.(θ)))
                nominal_solver = iLQSolver(nominal_game, max_scale_backtrack=10, max_elwise_diff_step=Inf, equilibrium_type=equilibrium_type)
                nominal_converged, nominal_traj, nominal_strategies = solve(nominal_game, nominal_solver, x0)
            end
            costs = parameterized_cost(θ)
            game = GeneralGame(game_horizon, player_inputs, dynamics, costs)
            lqg = Differentiable_Solvers.lq_approximation(game, nominal_traj, nominal_solver)
            if equilibrium_type=="OLNE_KKT" || equilibrium_type=="OLNE_costate" || equilibrium_type=="OLNE"
                traj = Differentiable_Solvers.trajectory(x0, game, Differentiable_Solvers.solve_lq_game_OLNE(lqg), nominal_traj)
            elseif equilibrium_type=="FBNE_KKT" || equilibrium_type=="FBNE_costate" || equilibrium_type=="FBNE"
                traj = Differentiable_Solvers.trajectory(x0, game, Differentiable_Solvers.solve_lq_game_FBNE(lqg), nominal_traj)
            else
                @warn "equilibrium_type is wrong!"
            end
            tmp2 = transpose(mapreduce(permutedims, vcat, Vector([Vector(traj.x[t][obs_state_list]) for t in obs_time_list])))
            loss_value = norm(tmp1-tmp2)^2
            return loss_value
        end
    end
end

function inverse_game_gradient_descent(θ::Vector, g::GeneralGame, expert_traj::SystemTrajectory, x0::SVector, 
                                        max_LineSearch_num::Int, parameterized_cost, equilibrium_type=[], Bayesian_belief_update=false, 
                                        specify_current_loss_and_solver=false, current_loss=[], current_traj=[], current_solver=[],
                                        obs_time_list = 1:game_horizon-1, obs_state_list = 1:nx, obs_control_list = 1:nu, initial_step_size=2.0, 
                                        normalization = false, which_example=[],no_control=false)
    "In this function, we first evaluate the loss function, which involves solve an iLQGames under the currest cost parameters θ.
    A byproduct is the linearized dynamics f̃ and the quadraticized costs {∑ⱼ θⱼⁱ b̃ⱼⁱ} for each player i.
    We solve an LQ game with the above linearized dynamics and quadraticized costs, and obtain a linear strategy.
    We simulate that linear strategy to obtain a new trajectory, and evaluate the loss function again.
    "

    # initialization
    α, θ_next, new_loss, new_traj, new_solver = initial_step_size, θ, 0.0, zero(SystemTrajectory, g), current_solver
    if Bayesian_belief_update==true
        equilibrium_type = inverse_game_update_belief(θ, g, expert_traj, x0, parameterized_cost, "FBNE_costate", "OLNE_costate")
    end
    if specify_current_loss_and_solver == false
        current_loss, current_traj, current_str, current_solver = loss(θ,iLQGames.dynamics(g), equilibrium_type, expert_traj, false, false,[],[],
                                                                        obs_time_list, obs_state_list, obs_control_list, no_control)
    end
    gradient_value = ForwardDiff.gradient(x -> loss(x,iLQGames.dynamics(g), equilibrium_type, expert_traj, true, true, current_solver, current_traj,
                                                                        obs_time_list, obs_state_list, obs_control_list, no_control), θ)
    for iter in 1:max_LineSearch_num
        step_size = α
        θ_next = θ-step_size*gradient_value
        while minimum(θ_next) <= -0.0
            step_size = step_size*0.5^2
            θ_next = θ-step_size*gradient_value
        end
        new_loss, new_traj, new_str, new_solver = loss(θ_next, iLQGames.dynamics(g),equilibrium_type, expert_traj, false, false,[],[],
                                                        obs_time_list, obs_state_list, obs_control_list, no_control)
        if new_loss < current_loss
            # println("Inverse Game Line Search Step Size: ", α)
            return θ_next, new_loss, gradient_value, equilibrium_type, new_traj, new_solver
            break
        end
        step_size = step_size*0.5
    end
    if normalization == true
        if which_example == "NL"
            sum1 = (θ_next[1] + θ_next[2])
            sum2 = (θ_next[3] + θ_next[4])
            θ_next[1] = 6*θ_next[1]/sum1
            θ_next[2] = 6*θ_next[2]/sum1
            θ_next[3] = 6*θ_next[3]/sum2
            θ_next[4] = 6*θ_next[4]/sum2
        end
        if which_example == "LQ"
            sum1 = (θ_next[1] + θ_next[2])
            sum2 = (θ_next[3] + θ_next[4])
            θ_next[1] = 4*θ_next[1]/sum1
            θ_next[2] = 4*θ_next[2]/sum1
            θ_next[3] = 4*θ_next[3]/sum2
            θ_next[4] = 4*θ_next[4]/sum2
        end
    end
    return θ_next, new_loss, gradient_value, equilibrium_type, new_traj, new_solver
end


function inverse_game_gradient_descent_with_x0(θ::Vector, g::GeneralGame, expert_traj::SystemTrajectory, x0::SVector,
                                        max_LineSearch_num::Int, parameterized_cost, equilibrium_type=[], Bayesian_belief_update=false, 
                                        specify_current_loss_and_solver=false, current_loss=[], current_traj=[], current_solver=[],
                                        obs_time_list = 1:game_horizon-1, obs_state_list = 1:nx, obs_control_list = 1:nu, initial_step_size=2.0, 
                                        which_example=[],no_control=false, x0_GD_num=5, step_size_x0 = 0.1, x0_GD_stepsize_shrink_factor=0.1) # x0_as_well, step_size_x0 and x0_GD_stepsize_shrink_factor are new.
    α, θ_next, new_loss, new_traj, new_solver = initial_step_size, θ, 0.0, zero(SystemTrajectory, g), current_solver
    if Bayesian_belief_update==true
        equilibrium_type = inverse_game_update_belief(θ, g, expert_traj, x0, parameterized_cost, "FBNE_costate", "OLNE_costate")
    end
    # evaluate loss function
    if specify_current_loss_and_solver == false
        current_loss, current_traj, current_str, current_solver = loss(θ,iLQGames.dynamics(g), equilibrium_type, expert_traj, false, false,[],[],
                                                                        obs_time_list, obs_state_list, obs_control_list, no_control, true, x0)
    end
    # evaluate gradient of loss function
    gradient_x0 = ForwardDiff.gradient(x -> loss(θ, iLQGames.dynamics(g), equilibrium_type, expert_traj, true, true, current_solver, current_traj,
                                                                        obs_time_list, obs_state_list, obs_control_list, no_control, true, x), x0 )
    # line search over x0
    x0_next = x0

    for iter in 1:x0_GD_num
        x0_next = x0 - step_size_x0*gradient_x0
        new_loss, new_traj, new_str, new_solver = loss(θ, iLQGames.dynamics(g),equilibrium_type, expert_traj, false, false,[],[],
                                                        obs_time_list, obs_state_list, obs_control_list, no_control, true, x0_next)
        # @infiltrate
        if new_loss < current_loss
            break
        end
        step_size_x0 = step_size_x0*x0_GD_stepsize_shrink_factor
    end
    gradient_value = ForwardDiff.gradient(x -> loss(x,iLQGames.dynamics(g), equilibrium_type, expert_traj, true, true, current_solver, current_traj,
                                                                        obs_time_list, obs_state_list, obs_control_list, no_control, true, x0_next), θ)
    # line search over cost
    for iter in 1:max_LineSearch_num
        step_size = α
        θ_next = θ - step_size*gradient_value
        while minimum(θ_next) <= -0.0
            step_size = step_size*0.5^2
            θ_next = θ-step_size*gradient_value
        end
        new_loss, new_traj, new_str, new_solver = loss(θ_next, iLQGames.dynamics(g),equilibrium_type, expert_traj, false, false,[],[],
                                                        obs_time_list, obs_state_list, obs_control_list, no_control, true, x0_next)
        if new_loss < current_loss
            # println("Inverse Game Line Search Step Size: ", α)
            return  x0_next, θ_next, new_loss, gradient_value, equilibrium_type, new_traj, new_solver
            break
        end
        step_size = step_size*0.5
    end
    return x0_next, θ_next, new_loss, gradient_value, equilibrium_type, new_traj, new_solver
end





function objective_inference_with_partial_obs(x0, θ, expert_traj, g, max_GD_iteration_num, equilibrium_type = "FBNE",
                            Bayesian_update=false, max_LineSearch_num=15, tol_LineSearch = 1e-6, 
                            obs_time_list = 1:game_horizon-1, obs_state_list = 1:nx, obs_control_list = 1:nu, 
                            convergence_tol = 0.01, terminate_when_small_progress = true, initial_step_size=2.0, ground_truth_expert=[], 
                            fast_convergence=false, normalization=false, which_example=[], no_control=false)
    θ_dim = length(θ)
    sol = [zeros(θ_dim) for iter in 1:max_GD_iteration_num+1]
    sol[1] = θ
    loss_values = zeros(max_GD_iteration_num+1)
    # loss_values[1],_,_ = inverse_game_loss(sol[1], g, expert_traj, x0, parameterized_cost, equilibrium_type)
    loss_values[1] = loss(θ, iLQGames.dynamics(g), equilibrium_type, expert_traj, true, false, 
                [], [], obs_time_list, obs_state_list, obs_control_list, no_control)
    gradient = [zeros(θ_dim) for iter in 1:max_GD_iteration_num]
    equilibrium_type_list = ["" for iter in 1:max_GD_iteration_num]
    converged = false
    keep_non_progressing_counter = 0
    getting_stuck_in_local_solution_counter = 0
    consistent_information_pattern = true
    ground_truth_loss = zeros(max_GD_iteration_num)
    for iter in 1:max_GD_iteration_num
        sol[iter+1], loss_values[iter+1], gradient[iter], equilibrium_type_list[iter] = inverse_game_gradient_descent(sol[iter], 
                                                                                g, expert_traj, x0, max_LineSearch_num, 
                                                                                parameterized_cost, equilibrium_type, Bayesian_update, false, [], [],[], 
                                                                                obs_time_list, obs_state_list, obs_control_list, initial_step_size, 
                                                                                normalization, which_example, no_control,
                                                                                )
        println("iteration: ", iter)
        println("current_loss: ", loss_values[iter+1])
        # println("equilibrium_type: ", equilibrium_type_list[iter])
        println("Current solution: ", sol[iter+1])
        if ground_truth_expert != []
            ground_truth_loss[iter] = loss(sol[iter+1], iLQGames.dynamics(g), equilibrium_type, ground_truth_expert, true)
        end
        if iter >1 && equilibrium_type_list[iter-1] != equilibrium_type_list[iter]
            consistent_information_pattern = false
        end

        if fast_convergence == true
            if iter >4
                if loss_values[iter]>loss_values[iter-1] && loss_values[iter-1]>loss_values[iter-2]
                    keep_non_progressing_counter += 1
                else
                    keep_non_progressing_counter = 0
                end
                
                if loss_values[iter-1] - loss_values[iter] <tol_LineSearch && loss_values[iter-2] - loss_values[iter-1] < tol_LineSearch
                    getting_stuck_in_local_solution_counter += 1
                else 
                    getting_stuck_in_local_solution_counter = 0
                end

                if getting_stuck_in_local_solution_counter > 3 || keep_non_progressing_counter > 3 || abs(loss_values[iter+1]-loss_values[iter])<tol_LineSearch
                    break
                end
            end
            if terminate_when_small_progress == true
                if loss_values[iter+1]<convergence_tol || abs(loss_values[iter+1] - loss_values[iter])<convergence_tol*0.01  # convergence tolerence
                    converged = true
                    break
                end
            else
                if loss_values[iter+1]<convergence_tol
                    converged = true
                end
            end
        end
    end
    return converged, sol, loss_values, gradient, equilibrium_type_list, consistent_information_pattern, ground_truth_loss
end

function objective_inference_with_partial_obs_x0(x0, θ, expert_traj, g, max_GD_iteration_num, equilibrium_type = "FBNE",
                            Bayesian_update=false, max_LineSearch_num=15, tol_LineSearch = 1e-6, 
                            obs_time_list = 1:game_horizon-1, obs_state_list = 1:nx, obs_control_list = 1:nu, 
                            convergence_tol = 0.01, terminate_when_small_progress = true, initial_step_size=2.0, ground_truth_expert=[], 
                            fast_convergence=false, normalization=false, which_example=[], no_control=false,
                            x0_GD_num=5, step_size_x0=0.1, x0_GD_stepsize_shrink_factor=0.1) # x0_as_well, step_size_x0 and x0_GD_stepsize_shrink_factor are new!
    θ_dim = length(θ)
    sol = [zeros(θ_dim) for iter in 1:max_GD_iteration_num+1]
    sol[1] = θ
    loss_values = zeros(max_GD_iteration_num+1)
    # loss_values[1],_,_ = inverse_game_loss(sol[1], g, expert_traj, x0, parameterized_cost, equilibrium_type)
    loss_values[1] = loss(θ, iLQGames.dynamics(g), equilibrium_type, expert_traj, true, false, 
                [], [], obs_time_list, obs_state_list, obs_control_list, no_control)
    gradient = [zeros(θ_dim) for iter in 1:max_GD_iteration_num]
    equilibrium_type_list = ["" for iter in 1:max_GD_iteration_num]
    converged = false
    keep_non_progressing_counter = 0
    getting_stuck_in_local_solution_counter = 0
    consistent_information_pattern = true
    ground_truth_loss = zeros(max_GD_iteration_num)
    x0_estimate = [x0 for iter in 1:max_GD_iteration_num+1]
    for iter in 1:max_GD_iteration_num
        # @infiltrate
        x0_tmp, sol[iter+1], loss_values[iter+1], gradient[iter], equilibrium_type_list[iter] = inverse_game_gradient_descent_with_x0(sol[iter], 
                                                                                g, expert_traj, x0_estimate[iter], max_LineSearch_num, 
                                                                                parameterized_cost, equilibrium_type, Bayesian_update, false, [], [],[], 
                                                                                obs_time_list, obs_state_list, obs_control_list, initial_step_size, 
                                                                                which_example, no_control, x0_GD_num, step_size_x0, x0_GD_stepsize_shrink_factor
                                                                                )
        x0_estimate[iter+1] = [x0_tmp[ii] for ii in 1:length(x0_tmp)]
        println("iteration: ", iter)
        println("current_loss: ", loss_values[iter+1])
        println("estimated x0: ", x0_estimate[iter+1])
        # println("equilibrium_type: ", equilibrium_type_list[iter])
        println("Current solution: ", sol[iter+1])
        if ground_truth_expert != []
            ground_truth_loss[iter] = loss(sol[iter+1], iLQGames.dynamics(g), equilibrium_type, ground_truth_expert, true)
        end
        if iter >1 && equilibrium_type_list[iter-1] != equilibrium_type_list[iter]
            consistent_information_pattern = false
        end

        if fast_convergence == true
            if iter >4
                if loss_values[iter]>loss_values[iter-1] && loss_values[iter-1]>loss_values[iter-2]
                    keep_non_progressing_counter += 1
                else
                    keep_non_progressing_counter = 0
                end
                
                if loss_values[iter-1] - loss_values[iter] <tol_LineSearch && loss_values[iter-2] - loss_values[iter-1] < tol_LineSearch
                    getting_stuck_in_local_solution_counter += 1
                else 
                    getting_stuck_in_local_solution_counter = 0
                end

                if getting_stuck_in_local_solution_counter > 3 || keep_non_progressing_counter > 3 || abs(loss_values[iter+1]-loss_values[iter])<tol_LineSearch
                    break
                end
            end
            if terminate_when_small_progress == true
                if loss_values[iter+1]<convergence_tol || abs(loss_values[iter+1] - loss_values[iter])<convergence_tol*0.01  # convergence tolerence
                    converged = true
                    break
                end
            else
                if loss_values[iter+1]<convergence_tol
                    converged = true
                end
            end
        end
    end
    return converged, x0_estimate, sol, loss_values, gradient, equilibrium_type_list, consistent_information_pattern, ground_truth_loss
end



# θ represents the initialization in gradient descent
function run_experiments_with_baselines(g, θ, x0_set, expert_traj_list, parameterized_cost, 
                                                max_GD_iteration_num, max_LineSearch_num=15, tol_LineSearch=1e-6, record_time=false, Bayesian_update=false,
                                                all_equilibrium_types = ["FBNE_costate","FBNE_costate"])
    # In the returned table, the rows coresponding to Bayesian, FB, OL
    n_data = length(x0_set)
    n_equi_types = length(all_equilibrium_types)
    sol_table  = [[[] for jj in 1:n_data] for ii in 1:n_equi_types+1]
    grad_table = [[[] for jj in 1:n_data] for ii in 1:n_equi_types+1]
    equi_table = [[[] for jj in 1:n_data] for ii in 1:n_equi_types+1]
    comp_time_table = [[0.0 for jj in 1:n_data] for ii in 1:n_equi_types+1]
    conv_table = [[false for jj in 1:n_data] for ii in 1:n_equi_types+1] # converged_table
    loss_table = [[[] for jj in 1:n_data] for ii in 1:n_equi_types+1]
    total_iter_table = zeros(1+n_equi_types, n_data)
    for iter in 1:n_data
        println(iter)
        x0 = x0_set[iter]
        expert_traj = expert_traj_list[iter]
        if record_time==true    time_stamp = time()  end
        conv_table[1][iter], sol_table[1][iter], loss_table[1][iter], grad_table[1][iter], equi_table[1][iter], consistent_information_pattern=objective_inference(x0,
                                                                        θ,expert_traj,g,max_GD_iteration_num,"FBNE_costate", true, max_LineSearch_num, tol_LineSearch)
        # @infiltrate
        if record_time==true    comp_time_table[1][iter] = time() - time_stamp  end
        total_iter_table[1,iter] = iterations_taken_to_converge(equi_table[1][iter])
        for index in 1:n_equi_types
            # @infiltrate
            if consistent_information_pattern==true && equi_table[1][iter][1] == all_equilibrium_types[index]
                # @infiltrate
                conv_table[index+1][iter], sol_table[index+1][iter], loss_table[index+1][iter], grad_table[index+1][iter], equi_table[index+1][iter] = conv_table[1][iter], sol_table[1][iter], loss_table[1][iter], grad_table[1][iter], equi_table[1][iter]
                
            else
                if record_time==true    time_stamp = time()  end        
                conv_table[index+1][iter], sol_table[index+1][iter], loss_table[index+1][iter], grad_table[index+1][iter], equi_table[index+1][iter],_=objective_inference(x0,
                                                                    θ,expert_traj,g,max_GD_iteration_num, all_equilibrium_types[index], false, max_LineSearch_num, tol_LineSearch)
                if record_time==true    comp_time_table[index+1][iter] = time() - time_stamp  end
                total_iter_table[index+1,iter] = iterations_taken_to_converge(equi_table[index+1][iter])
            end
        end
    end
    return conv_table, sol_table, loss_table, grad_table, equi_table, total_iter_table, comp_time_table
end


function run_experiment_x0(g, θ, x0_set, expert_traj_list, parameterized_cost, 
                        max_GD_iteration_num, max_LineSearch_num=15, tol_LineSearch=1e-6,
                        obs_time_list = 1:game_horizon-1, obs_state_list = 1:nx, obs_control_list = 1:nu, equilibrium_type = "FBNE_costate", convergence_tol=0.01, 
                        terminate_when_small_progress=true, initial_step_size=2.0, ground_truth_expert=[], fast_convergence=false, 
                        normalization=false, which_example=[], no_control=false,
                        x0_GD_num=10, step_size_x0=0.1, x0_GD_stepsize_shrink_factor = 0.1)
    n_data = length(x0_set)
    sol_table  = [[] for jj in 1:n_data]
    x0_table  = [[] for jj in 1:n_data]
    grad_table = [[] for jj in 1:n_data]
    conv_table = [false for jj in 1:n_data] # converged_table
    loss_table = [[] for jj in 1:n_data]
    total_iter_table = zeros(n_data)
    equi_table = [[] for jj in 1:n_data]
    ground_truth_loss= [[] for jj in 1:n_data]

    for iter in 1:n_data
        println(iter)
        x0 = x0_set[iter]
        expert_traj = expert_traj_list[iter]
        conv_table[iter], x0_table[iter], sol_table[iter], loss_table[iter], grad_table[iter], equi_table[iter], consistent_information_pattern, ground_truth_loss[iter] = objective_inference_with_partial_obs_x0(x0,
                                                                        θ, expert_traj, g, max_GD_iteration_num, equilibrium_type,false, max_LineSearch_num, tol_LineSearch,
                                                                        obs_time_list, obs_state_list, obs_control_list, convergence_tol, terminate_when_small_progress, initial_step_size, 
                                                                        ground_truth_expert, fast_convergence, normalization, which_example, no_control,
                                                                        x0_GD_num, step_size_x0, x0_GD_stepsize_shrink_factor)
        total_iter_table[iter] = iterations_taken_to_converge(equi_table[iter])
    end
    return conv_table, x0_table, sol_table, loss_table, grad_table, equi_table, total_iter_table, ground_truth_loss
end






# Given θ and equilibrium type, for each initial condition in x0_set, compute prediction loss
function generalization_loss(g, θ, x0_set, expert_traj_list, parameterized_cost, equilibrium_type_list)
    num_samples = length(x0_set)
    loss_list = zeros(num_samples)
    traj_list = [zero(SystemTrajectory, g) for ii in 1:num_samples]

    for iter in 1:length(x0_set)
        loss_list[iter], tmp_traj, _, _ = loss(θ, iLQGames.dynamics(g), equilibrium_type_list[iter], expert_traj_list[iter], false)
        for t in 1:g.h
            traj_list[iter].x[t] = tmp_traj.x[t]
            traj_list[iter].u[t] = tmp_traj.u[t]
        end
    end
    return loss_list, traj_list
end


# generate expert trajectories for initial conditions in x0_set
function generate_traj(g, x0_set, parameterized_cost, equilibrium_type_list )
    n_data = length(x0_set)
    conv = [false for ii in 1:n_data]
    expert_traj_list = [zero(SystemTrajectory, g) for ii in 1:n_data]
    expert_equi_list = ["" for ii in 1:n_data]
    for item in 1:n_data
        if rand(1)[1]>0.5
            expert_equi_list[item] = equilibrium_type_list[1]
        else
            expert_equi_list[item] = equilibrium_type_list[2]
        end
        solver = iLQSolver(g, max_scale_backtrack=10, max_elwise_diff_step=Inf, 
                            equilibrium_type=expert_equi_list[item])
        conv[item], tmp, _ = solve(g, solver, x0_set[item])
        for t in 1:g.h
            expert_traj_list[item].x[t] = tmp.x[t]
            expert_traj_list[item].u[t] = tmp.u[t]
        end
    end
    return conv, expert_traj_list, expert_equi_list
end

function generate_noisy_observation(nx, nu, g, expert_traj, noise_level, number_of_perturbed_traj_needed)
    # nx is the dimension of the states
    # nu is the dimension of the actions
    # g is the game
    # g.h represents the horizon
    perturbed_trajectories_list = [zero(SystemTrajectory, g) for ii in 1:number_of_perturbed_traj_needed]
    for ii in 1:number_of_perturbed_traj_needed
        for t in 1:g.h
            perturbed_trajectories_list[ii].x[t] = expert_traj.x[t] + rand(Normal(0, noise_level), nx)
            perturbed_trajectories_list[ii].u[t] = expert_traj.u[t] + rand(Normal(0, noise_level), nu)
        end
    end
    return perturbed_trajectories_list
end


function iterations_taken_to_converge(equi_list)
    return sum(equi_list[ii]!="" for ii in 1:length(equi_list))
end



# Get the best possible reward estimate
function get_the_best_possible_reward_estimate(x0_set, all_equilibrium_types, sol_table, loss_table, equi_list)
    n_data = length(x0_set)
    n_equi_types = length(all_equilibrium_types)
    θ_list = [[[] for ii in 1:n_data] for index in 1:n_equi_types+1]
    index_list = [[0 for ii in 1:n_data] for index in 1:n_equi_types+1]
    optim_loss_list = [[0.0 for ii in 1:n_data] for index in 1:n_equi_types+1]
    for index in 1:n_equi_types+1
        for ii in 1:n_data
            if minimum(loss_table[index][ii])==0.0
                index_list[index][ii] = iterations_taken_to_converge(equi_list[index][ii])
                θ_list[index][ii] = sol_table[index][ii][index_list[index][ii]]
                optim_loss_list[index][ii] = loss_table[index][ii][index_list[index][ii]]
            else
                index_list[index][ii] = argmin(loss_table[index][ii])
                # @infiltrate
                θ_list[index][ii] = sol_table[index][ii][index_list[index][ii]]
                optim_loss_list[index][ii] = loss_table[index][ii][index_list[index][ii]]
            end
        end
    end
    return θ_list, index_list, optim_loss_list
end

function get_the_best_possible_reward_estimate_single(x0_set, all_equilibrium_types, sol_table, loss_table, equi_list)
    n_data = length(x0_set)
    θ_list = [[] for ii in 1:n_data]
    index_list = [0 for ii in 1:n_data]
    optim_loss_list = [0.0 for ii in 1:n_data]    
    for ii in 1:n_data
        if minimum(loss_table[ii])==0.0
            index_list[ii] = iterations_taken_to_converge(equi_list[ii])+1
            θ_list[ii] = sol_table[ii][index_list[ii]]
            optim_loss_list[ii] = loss_table[ii][index_list[ii]]
        else
            index_list[ii] = argmin(loss_table[ii])
            θ_list[ii] = sol_table[ii][index_list[ii]]
            optim_loss_list[ii] = loss_table[ii][index_list[ii]]
        end
    end

    return θ_list, index_list, optim_loss_list
end





function generate_LQ_problem_and_traj(game_horizon, ΔT, player_inputs, costs, x0_set, equilibrium_type_list, num_LQs=1)
    games, solvers,expert_trajs, expert_equi, converged_expert,  = [], [], [], [], [] 
    for index in 1:num_LQs
        if index <= 4
            joint_dynamics = LTISystem(LinearSystem{ΔT}(SMatrix{4,4}(Matrix(1.0*I,4,4)), SMatrix{4,4}(Matrix(1.0*I,4,4))), 
                SVector(1, 2, 3, 4))
        else
            joint_dynamics = LTISystem(LinearSystem{ΔT}(SMatrix{4,4}(rand(4,4)-0.5*ones(4,4)), SMatrix{4,4}(Matrix(1.0*I,4,4))), 
                SVector(1, 2, 3, 4))
        end

        game = GeneralGame(game_horizon,player_inputs, joint_dynamics, costs)
        
        if rand(1)[1]>0.5
            equi = equilibrium_type_list[1]
        else
            equi = equilibrium_type_list[2]
        end
        solver = iLQSolver(game, max_scale_backtrack=10, max_elwise_diff_step=Inf, equilibrium_type=equi)
        converged, traj, _ = solve(game, solver, x0_set[index])
        push!(games, game)
        push!(expert_trajs, traj)
        push!(expert_equi, equi)
        push!(solvers, solver)
        push!(converged_expert, converged)
    
    end
    return games, expert_trajs, expert_equi, solvers, converged_expert
end

function generate_expert_traj(game, solver, x0_set, num_trajs=1)
    expert_trajs, converged_expert  = [], [] 
    for index in 1:num_trajs        
        converged, traj, _ = solve(game, solver, x0_set[index])
        push!(expert_trajs, traj)
        push!(converged_expert, converged)
    end
    return expert_trajs, converged_expert
end

