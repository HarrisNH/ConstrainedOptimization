using Random

# Setting seed for reproducibility
Random.seed!(1234)

include("../handin/problem1.jl")

# Problem parameters
n = 100
alpha = 0.5
density = 0.15
beta = 0.75

# Generate random sparse test problem
H, g, A, b = RandomEQP(n, alpha, density, beta, "sparse")

# Solving the problem
x_s, lambda_s = EqualityQPSolver(H, g, A, b, "sparse")

# Solving the problem as dense
x_d, lambda_d = EqualityQPSolver(H, g, A, b, "dense")

# Output solution
println("Optimal point when solved as sparse:")
println(x_s)
println("")
println("Optimal point when solved as dense:")
println(x_d)