using Printf
using LinearAlgebra

include("../problem4.jl")

starts = [
    [0.0, 0.0],
    [1.0, 1.0],
    [2.0, 2.0],
    [0.0, 2.0],
    [1.0, 5.0],
    [-1.0, 1.0],
    [3.0, 3.0],
    [-2.0, 0.0]
]

function is_feasible(x)
    c_val, _ = test_ineq(x)
    return all(c_val .>= -1e-8)
end

function is_global(x)
    return norm(x - [3.0, 2.0]) < 1e-3
end

println("Testing Himmelblau with different initial points x^{(0)}")
for x0 in starts
    println("x^{(0)} = $x0")
    z0 = [0.0, 0.0]
    B0 = Matrix{Float64}(I, 2, 2)

    x_line_search_final, _, _, iter_line_search, _ = redirect_stdout(devnull) do 
        SQP_line_search(
            test_func, test_ineq, x0, z0; B0 = copy(B0)
        )
    end 
    println("\tLine search: $x_line_search_final, iterations: $iter_line_search, global?: $(is_global(x_line_search_final))")

    x_trust_region_final, _, _, iter_trust_region, _ = redirect_stdout(devnull) do 
        SQP_trust_region(
            test_func, test_ineq, x0, z0, 1.0; B0 = copy(B0)
        )
    end 
    println("\tTrust region: $x_trust_region_final, iterations: $iter_trust_region, global?: $(is_global(x_trust_region_final))")
end
