using Random

# Setting seed for reproducibility
Random.seed!(1234)

include("../handin/problem1.jl")

# Problem parameters
n = 50
alpha = 0.5
density = 0.15
beta = 0.75

# Generate random sparse test problem
H, g, A, b = RandomEQP(n, alpha, density, beta, "sparse")

# Solving the problem as sparse
x_s, lambda_s = EqualityQPSolver(H, g, A, b, "sparse")

# Solving the problem as dense
x_d, lambda_d = EqualityQPSolver(H, g, A, b, "dense")

# Solving problem with IPOPT
x_I = EqualityQPSolverIPOPT(H, g, A, b)

# Output solution
println("Optimal point when solved as sparse:")
println(x_s)
println("")
println("Optimal point when solved as dense:")
println(x_d)
println("")
println("Optimal point found with IPOPT:")
println(x_I)
println("")
println("2-norm of the error (assuming IPOPT gives the correct solution)")
println(norm(x_s-x_I)," and ",norm(x_d-x_I))
