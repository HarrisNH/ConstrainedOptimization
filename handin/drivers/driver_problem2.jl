using Random
using JuMP
# Setting seed for reproducibility
Random.seed!(1234)

include("../problem2.jl")
n = 10
m = 10
H, g, A, b_l, b_u, x_l, x_u = generate_test_problem_qp(n, m)

x_as, _, _, x_walk_act = solve_convex_problem(H, g, A, b_l, b_u, x_l, x_u, n)

# Interior-point walk
C, d = setup_qp_with_bounds(H, g, A, b_l, b_u, x_l, x_u)
x0_walk = x_walk_act[1]
mc = size(C, 2)
z0 = ones(mc)
s0 = ones(mc)
result_ip = primal_dual_qp_ineq(H, g, C, d, x0_walk, z0, s0)
x_ip = result_ip.x

# Ipopt reference
res_lib = solve_with_commercial(H, g, A, b_l, b_u, x_l, x_u)
x_lib = res_lib.x

# Output solution
println("Optimal point when using primal active-set:")
println(x_as)
println("")
println("Optimal point when using primal-dual interior-point:")
println(x_ip)
println("")
println("Optimal point found with IPOPT:")
println(x_lib)
println("")
println("2-norm of the error (assuming IPOPT gives the correct solution)")
println(norm(x_as-x_lib)," and ",norm(x_ip-x_lib))
