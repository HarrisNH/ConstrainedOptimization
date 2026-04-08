using LinearAlgebra, Plots, SparseArrays
using JuMP
using Ipopt
using Random, Distributions
using Plots

# c. 
"""
We consider the *convex* QP in the form 
min_{x} phi = 1/2 x' H x + g' x 
s.t. b_l <= A' x <= b_u, 
     x_l <= x <= x_u. 

Since this a convex(!) program then H > 0, 
that is, H is positive definite. 
"""

# function generate_test_problem(n, m_a)
#     """
#     Generates H, g,A, b_l, b_u, x_l, x_u of size n. 

#     H, g, A, b_l, b_u, x_l, x_u = generate_test_problem(n, m_a)
#     """

#     # TODO: maybe find better numbers 
#     # and discuss the generation methods
#     M = rand(Uniform(-1, 1), n, n)
#     alpha = rand(Uniform(0.5, 2))
#     H = M * M' .+ alpha # this ensures that H > 0 

#     g = rand(Uniform(-5, 5), n)
    
#     b_l = rand(Uniform(-5, 5), m_a)
#     delta_b = rand(Uniform(0.5, 5), m_a)
#     b_u = b_l + delta_b

#     x_l = rand(Uniform(-5, 5), n)
#     delta_x = rand(Uniform(0.5, 5), n)
#     x_u = x_l + delta_x

#     A = rand(Uniform(-2, 2), n, m_a)

#     return H, g, A, b_l, b_u, x_l, x_u
# end

function generate_test_problem(n, m_a)
    """
    Generates H, g,A, b_l, b_u, x_l, x_u of size n. 

    H, g, A, b_l, b_u, x_l, x_u = generate_test_problem(n, m_a)
    """

    # TODO: maybe find better numbers 
    # and discuss the generation methods
    M = rand(Uniform(-1, 1), n, n)
    alpha = rand(Uniform(0.5, 2))
    H = M * M' .+ alpha # this ensures that H > 0 

    g = rand(Uniform(-5, 5), n)
    
    x = rand(n)
    diff_x = rand(Uniform(0, 5), n)
    x_l = x .- diff_x
    x_u = x .+ diff_x

    A = rand(Uniform(-2, 2), n, m_a)
    y = A' * x
    diff_Ax = rand(Uniform(0, 5), m_a)
    # print(size(diff_Ax))
    b_l = y .- diff_Ax
    b_u = y .+ diff_Ax
 
    return H, g, A, b_l, b_u, x_l, x_u
 end
 

function plot_qp(H, g, A, b_l, b_u, x_l, x_u; x_star=nothing)
    # ensure dimension is 2
    n = length(g)
    if n != 2
        error("Plotting only supported for 2D problems.")
    end

    # objective function
    f(x1, x2) = 0.5 * [x1, x2]' * H * [x1, x2] + g' * [x1, x2]

    # plotting grid
    x_range = range(x_l[1]-1, x_u[1]+1, length=200)
    y_range = range(x_l[2]-1, x_u[2]+1, length=200)

    Z = [f(x, y) for y in y_range, x in x_range]

    contour(
        x_range,
        y_range,
        Z,
        levels=30,
        linewidth=1,
        title="Quadratic objective with feasible region",
        xlabel="x₁",
        ylabel="x₂"
    )

    # draw box constraints
    plot!(
        [x_l[1], x_u[1], x_u[1], x_l[1], x_l[1]],
        [x_l[2], x_l[2], x_u[2], x_u[2], x_l[2]],
        lw=2,
        label="box constraints"
    )

    # linear inequality constraints
    for i in 1:size(A,2)
        a = A[:,i]

        if abs(a[2]) > 1e-8
            line(x) = (b_u[i] - a[1]*x) / a[2]
            plot!(x_range, line.(x_range), linestyle=:dash, label=false)

            line(x) = (b_l[i] - a[1]*x) / a[2]
            plot!(x_range, line.(x_range), linestyle=:dash, label=false)
        end
    end

    # optimal solution marker
    if x_star !== nothing
        scatter!([x_star[1]], [x_star[2]], markersize=6, label="solution")
    end
    savefig("test_plot.png")
    return current()
end

# d. 
function library_solver(H, g, A, b_l, b_u, x_l, x_u)
    """
    This solves the problem above using Ipopt. 
    """
    n = size(A)[1]
    model = Model(Ipopt.Optimizer)
    @variable(model, x[1:n])
    @constraint(model, b_l .<= A' * x .<= b_u)
    @constraint(model, x_l .<= x .<= x_u)
    @objective(model, Min, 1/2 * x' * H * x + g' * x)
    optimize!(model)

    return value.(x)
end

# H = [4 2; 2 6]
# g = [2, 3]
# A = [1; 1;;]
# b_l = [1]
# b_u = b_l .+ 3
# x_l = [1, 2]
# x_u = x_l .+ 3
# H = [1.7310033470644977 1.3380919827815996; 1.3380919827815996 1.1008601516900915]
# g = [-3.5636657217994094, -3.0242004638657405]
# A = [1.4995076809027266; -1.0640380938151917;;]
# b_l = [-0.023294879434302196]
# b_u = [1.4066347634539955]
# x_l = [-0.8632981723875321, -4.512339025674606]
# x_u = [2.717188398215795, -0.694963544245569]

H, g, A, b_l, b_u, x_l, x_u = generate_test_problem(30, 10000)
println(H)
println(g) 
println(A)
println(b_l) 
println(b_u)
println(x_l)
println(x_u)

solution = library_solver(H, g, A, b_l, b_u, x_l, x_u)
println("solution = $solution")
# plot_qp(H, g, A, b_l, b_u, x_l, x_u; x_star=nothing)