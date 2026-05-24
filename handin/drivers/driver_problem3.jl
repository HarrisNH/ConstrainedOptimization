using Random
using LinearAlgebra
using JuMP, Ipopt

# Setting seed for reproducibility
Random.seed!(1234)

include("../test_problems.jl") 
include("../problem3.jl")

# Generate test problem
n_dim = 25
n_con = 50
g, A, b_l, b_u, x_l, x_u, x = generate_test_problem_lp(n_dim, n_con)

# rewrite to standard form
A_std, b_std, g_std = to_standard_form(g, A, b_l, b_u, x_l, x_u)
n_std = length(g_std)

# find feasible point
A_fp, b_fp, g_fp, x0_fp = fp_standard_form(A_std, b_std)
result_fp = revised_simplex(A_fp, b_fp, g_fp, x0_fp)
x0_std = result_fp.x[1:n_std]

# solve with the primal dual interior point algorithm
x0_ip = ones(n_std)
x_ip, _, _, ip_history, _ = primal_dual_interior_LP(g_std, A_std, b_std, x0_ip, n_dim, x_l)
x_ip_orig = x_ip[1:n_dim] .+ x_l # recover non-shifted x

# solve with teh revised simplex algorithm
result_sim = revised_simplex(A_std, b_std, g_std, x0_std)
x_sim_orig = result_sim.x[1:n_dim] .+ x_l # recover non-shifted x

# solve with IPOPT
x_lib = library_solver_lp(g, A, b_l, b_u, x_l, x_u).x


# Note: specific tolerances are set within our algorithms 
println("Optimal point found with the primal dual interior algorithm:")
println(x_ip_orig)
println("")
println("Optimal point found with the revised simplex algorithm:")
println(x_sim_orig)
println("")
println("Optimal point found with IPOPT:")
println(x_lib)
println("")
println("2-norm of the error (assuming IPOPT gives the correct solution)")
println(norm(x_ip_orig-x_lib)," and ",norm(x_sim_orig-x_lib))
