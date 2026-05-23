using Random

# Setting seed for reproducibility
Random.seed!(1234)

include("../problem4.jl")
x0 = [0.0, 0.0]
z0 = [0.0, 0.0] 
tr0 = 1.0
B0 = Matrix{Float64}(I, 2, 2)

x_tr, _, _, _, _ = SQP_trust_region(
    test_func, test_ineq, x0, z0, tr0;
    B0=B0, tol=1e-9, MAX_ITER=50
)


x0 = [0.0, 0.0]
z0 = [0.0, 0.0] # Initial multipliers for inequality constraints
B0 = Matrix{Float64}(I, 2, 2)

x_ls, _, _, _, _ = SQP_line_search(
    test_func, test_ineq, x0, z0;
    B0=B0, tol=1e-9, MAX_ITER=50
)

x_lib = HB_lib_sol()


# Output solution
println("Results for HimmelBlau")
println("Optimal point when using primal active-set:")
println(x_tr)
println("")
println("Optimal point when using primal-dual interior-point:")
println(x_ls)
println("")
println("Optimal point found with IPOPT:")
println(x_lib)
println("")
println("2-norm of the error (assuming IPOPT gives the correct solution)")
println(norm(x_tr-x_lib)," and ",norm(x_ls-x_lib))

println("")