using ProgressBars
using Infiltrator


function new_loss(θ, dynamics, equilibrium_type, expert_traj, gradient_mode = true, specified_solver_and_traj = false, 
                nominal_solver=[], nominal_traj=[], obs_time_list = 1:game_horizon-1, 
                obs_state_list = 1:nx, obs_control_list = 1:nu, no_control=false, x0_mode=false, x0=[],    static_game=[],static_solver=[], true_game_nx=12+1) 
    # last three items new
    n_θ=5
    if x0_mode==false
        x0 = first(expert_traj.x)
    end
    tmp1 = transpose(mapreduce(permutedims, vcat, Vector([Vector(expert_traj.x[t][obs_state_list]) for t in obs_time_list])))
    if no_control == true
        ctrl_coeff = 0
    else
        ctrl_coeff = 1
    end
    if gradient_mode == false
        nominal_converged, nominal_traj, nominal_strategies = solve(static_game, static_solver, SVector{true_game_nx+n_θ}([x0[1:true_game_nx]; SVector{n_θ}(θ)]))
        tmp2 = transpose(mapreduce(permutedims, vcat, Vector([Vector(nominal_traj.x[t][obs_state_list]) for t in obs_time_list])))    
        loss_value = norm(tmp2 - tmp1)^2 + regularization_size*sum(x0[end-(n_θ-1):end].*x0[end-(n_θ-1):end])
        return loss_value, nominal_traj, nominal_strategies, static_solver
    else
        if x0_mode==true
            if specified_solver_and_traj == false
                nominal_converged, nominal_traj, nominal_strategies = solve(static_game, static_solver, SVector{true_game_nx+n_θ}([SVector{true_game_nx}(ForwardDiff.value.(x0[1:true_game_nx]));  SVector{n_θ}(ForwardDiff.value.(θ))])  )
            end
            lqg = Differentiable_Solvers.lq_approximation(static_game, nominal_traj, static_solver)
            if equilibrium_type=="OLNE_KKT" || equilibrium_type=="OLNE_costate" || equilibrium_type=="OLNE"
                traj = Differentiable_Solvers.trajectory(SVector{true_game_nx+n_θ}([x0[1:true_game_nx]; SVector{n_θ}(θ)]), static_game, Differentiable_Solvers.solve_lq_game_OLNE(lqg), nominal_traj)
            elseif equilibrium_type=="FBNE_KKT" || equilibrium_type=="FBNE_costate" || equilibrium_type=="FBNE"
                traj = Differentiable_Solvers.trajectory(SVector{true_game_nx+n_θ}([x0[1:true_game_nx]; SVector{n_θ}(θ)]), static_game, Differentiable_Solvers.solve_lq_game_FBNE(lqg), nominal_traj)
            else
                @warn "equilibrium_type is wrong!"
            end
            tmp2 = transpose(mapreduce(permutedims, vcat, Vector([Vector(traj.x[t][obs_state_list]) for t in obs_time_list])))
            loss_value = norm(tmp1-tmp2)^2 + regularization_size*sum(x0[end-(n_θ-1):end].*x0[end-(n_θ-1):end])
            return loss_value
        else
            if specified_solver_and_traj == false
                nominal_converged, nominal_traj, nominal_strategies = solve(static_game, static_solver, SVector{true_game_nx+n_θ}([SVector{true_game_nx}(ForwardDiff.value.(x0[1:true_game_nx]));  SVector{n_θ}(ForwardDiff.value.(θ))]))
            end
            lqg = Differentiable_Solvers.lq_approximation(static_game, nominal_traj, static_solver)
            if equilibrium_type=="OLNE_KKT" || equilibrium_type=="OLNE_costate" || equilibrium_type=="OLNE"
                traj = Differentiable_Solvers.trajectory(SVector{true_game_nx+n_θ}([x0[1:true_game_nx]; SVector{n_θ}(θ)]), static_game, Differentiable_Solvers.solve_lq_game_OLNE(lqg), nominal_traj)
            elseif equilibrium_type=="FBNE_KKT" || equilibrium_type=="FBNE_costate" || equilibrium_type=="FBNE"
                traj = Differentiable_Solvers.trajectory(SVector{true_game_nx+n_θ}([x0[1:true_game_nx]; SVector{n_θ}(θ)]), static_game, Differentiable_Solvers.solve_lq_game_FBNE(lqg), nominal_traj)
            else
                @warn "equilibrium_type is wrong!"
            end
            tmp2 = transpose(mapreduce(permutedims, vcat, Vector([Vector(traj.x[t][obs_state_list]) for t in obs_time_list])))
            loss_value = norm(tmp1-tmp2)^2 + regularization_size*sum(x0[end-(n_θ-1):end].*x0[end-(n_θ-1):end])
            return loss_value
        end
    end
end




function new_inverse_game_gradient_descent_with_x0(θ::Vector, g::GeneralGame, expert_traj::SystemTrajectory, x0::SVector,
                                        max_LineSearch_num::Int, parameterized_cost, equilibrium_type=[], Bayesian_belief_update=false, 
                                        specify_current_loss_and_solver=false, current_loss=[], current_traj=[], current_solver=[],
                                        obs_time_list = 1:game_horizon-1, obs_state_list = 1:nx, obs_control_list = 1:nu, initial_step_size=2.0, 
                                        which_example=[],no_control=false, x0_GD_num=5, step_size_x0 = 0.1, x0_GD_stepsize_shrink_factor=0.1,
                                        static_game=[], static_solver=[], true_game_nx=12+1) # x0_as_well, step_size_x0 and x0_GD_stepsize_shrink_factor are new.
    α, θ_next, new_loss_value, new_traj, new_solver = initial_step_size, θ, 0.0, zero(SystemTrajectory, g), current_solver
    if Bayesian_belief_update==true
        equilibrium_type = inverse_game_update_belief(θ, g, expert_traj, x0, parameterized_cost, "FBNE_costate", "OLNE_costate")
    end
    if specify_current_loss_and_solver == false
        current_loss, current_traj, current_str, current_solver = new_loss(θ,iLQGames.dynamics(g), equilibrium_type, expert_traj, false, false,[],[],
                                                                        obs_time_list, obs_state_list, obs_control_list, no_control, true, x0,
                                                                        static_game,static_solver,true_game_nx)
    end
    # cfg1=
    # cfg2=
    gradient_x0 = ForwardDiff.gradient(x -> new_loss(θ, iLQGames.dynamics(g), equilibrium_type, expert_traj, true, true, current_solver, current_traj,
                                                                        obs_time_list, obs_state_list, obs_control_list, no_control, true, x,static_game,static_solver,true_game_nx), x0 )
    x0_next = x0[1:true_game_nx]
    for iter in 1:x0_GD_num
        x0_next = x0[1:true_game_nx] - step_size_x0*gradient_x0[1:true_game_nx]
        new_loss_value, new_traj, new_str, new_solver = new_loss(θ, iLQGames.dynamics(g),equilibrium_type, expert_traj, false, false,[],[],
                                                        obs_time_list, obs_state_list, obs_control_list, no_control, true, x0_next,static_game,static_solver,true_game_nx)
        # @infiltrate
        if new_loss_value < current_loss
            break
        end
        step_size_x0 = step_size_x0*x0_GD_stepsize_shrink_factor
    end
    gradient_value = ForwardDiff.gradient(x -> new_loss(x,iLQGames.dynamics(g), equilibrium_type, expert_traj, true, true, current_solver, current_traj,
                                                                        obs_time_list, obs_state_list, obs_control_list, no_control, true, x0_next,static_game,static_solver,true_game_nx), θ)
    # line search over cost
    for iter in 1:max_LineSearch_num
        step_size = α
        θ_next = θ - step_size*gradient_value
        while minimum(θ_next) <= -0.0
            step_size = step_size*0.5^2
            θ_next = θ-step_size*gradient_value
        end
        new_loss_value, new_traj, new_str, new_solver = new_loss(θ_next, iLQGames.dynamics(g),equilibrium_type, expert_traj, false, false,[],[],
                                                        obs_time_list, obs_state_list, obs_control_list, no_control, true, x0_next,static_game,static_solver,true_game_nx)
        if new_loss_value < current_loss
            # println("Inverse Game Line Search Step Size: ", α)
            return  x0_next, θ_next, new_loss_value, gradient_value, equilibrium_type, new_traj, new_solver
            break
        end
        step_size = step_size*0.5
    end
    return x0_next, θ_next, new_loss_value, gradient_value, equilibrium_type, new_traj, new_solver
end


function new_objective_inference_with_partial_obs_x0(x0, θ, expert_traj, g, max_GD_iteration_num, equilibrium_type = "FBNE",
                            Bayesian_update=false, max_LineSearch_num=15, tol_LineSearch = 1e-6, 
                            obs_time_list = 1:game_horizon-1, obs_state_list = 1:nx, obs_control_list = 1:nu, 
                            convergence_tol = 0.01, terminate_when_small_progress = true, initial_step_size=2.0, ground_truth_expert=[], 
                            fast_convergence=false, normalization=false, which_example=[], no_control=false,
                            x0_GD_num=5, step_size_x0=0.1, x0_GD_stepsize_shrink_factor=0.1,
                            static_game=[],static_solver=[],true_game_nx=13) # x0_as_well, step_size_x0 and x0_GD_stepsize_shrink_factor are new!
    θ_dim = length(θ)
    sol = [zeros(θ_dim) for iter in 1:max_GD_iteration_num+1]
    sol[1] = θ
    loss_values = zeros(max_GD_iteration_num+1)
    # loss_values[1],_,_ = inverse_game_loss(sol[1], g, expert_traj, x0, parameterized_cost, equilibrium_type)
    loss_values[1] = new_loss(θ, iLQGames.dynamics(g), equilibrium_type, expert_traj, true, false, 
                [], [], obs_time_list, obs_state_list, obs_control_list, no_control, false, [],static_game,static_solver,true_game_nx)
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
        x0_tmp, sol[iter+1], loss_values[iter+1], gradient[iter], equilibrium_type_list[iter] = new_inverse_game_gradient_descent_with_x0(sol[iter], 
                                                                                g, expert_traj, x0_estimate[iter], max_LineSearch_num, 
                                                                                parameterized_cost, equilibrium_type, Bayesian_update, false, [], [],[], 
                                                                                obs_time_list, obs_state_list, obs_control_list, initial_step_size, 
                                                                                which_example, no_control, x0_GD_num, step_size_x0, x0_GD_stepsize_shrink_factor
                                                                                ,static_game,static_solver,true_game_nx)
        # @infiltrate
        x0_estimate[iter+1] = [[x0_tmp[ii] for ii in 1:length(x0_tmp)]; sol[iter+1]]
        println("iteration: ", iter)
        println("current_loss: ", loss_values[iter+1])
        println("estimated x0: ", x0_estimate[iter+1])
        # println("equilibrium_type: ", equilibrium_type_list[iter])
        println("Current solution: ", sol[iter+1])
        if ground_truth_expert != []
            ground_truth_loss[iter] = new_loss(sol[iter+1], iLQGames.dynamics(g), equilibrium_type, ground_truth_expert, true, false, [],[],1:game_horizon-1, 1:true_game_nx-1, 1:nu, 
                                                false, false, [],static_game,static_solver,true_game_nx)
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






function new_run_experiment_x0(g, θ, x0_set, expert_traj_list, parameterized_cost, 
                        max_GD_iteration_num, max_LineSearch_num=15, tol_LineSearch=1e-6,
                        obs_time_list = 1:game_horizon-1, obs_state_list = 1:nx, obs_control_list = 1:nu, equilibrium_type = "FBNE_costate", convergence_tol=0.01, 
                        terminate_when_small_progress=true, initial_step_size=2.0, ground_truth_expert=[], fast_convergence=false, 
                        normalization=false, which_example=[], no_control=false,
                        x0_GD_num=10, step_size_x0=0.1, x0_GD_stepsize_shrink_factor = 0.1,
                        static_game=[],static_solver=[],true_game_nx=13)
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
        conv_table[iter], x0_table[iter], sol_table[iter], loss_table[iter], grad_table[iter], equi_table[iter], consistent_information_pattern, ground_truth_loss[iter] = new_objective_inference_with_partial_obs_x0(x0,
                                                                        θ, expert_traj, g, max_GD_iteration_num, equilibrium_type,false, max_LineSearch_num, tol_LineSearch,
                                                                        obs_time_list, obs_state_list, obs_control_list, convergence_tol, terminate_when_small_progress, initial_step_size, 
                                                                        ground_truth_expert, fast_convergence, normalization, which_example, no_control,
                                                                        x0_GD_num, step_size_x0, x0_GD_stepsize_shrink_factor,
                                                                        static_game,static_solver,true_game_nx)
        total_iter_table[iter] = iterations_taken_to_converge(equi_table[iter])
    end
    return conv_table, x0_table, sol_table, loss_table, grad_table, equi_table, total_iter_table, ground_truth_loss
end








