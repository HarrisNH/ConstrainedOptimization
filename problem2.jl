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
    # and discuss the generation method - the idea should be good. 
    M = rand(Uniform(-1, 1), n, n)
    alpha = rand(Uniform(0.5, 2))
    H = M * M' + alpha * I  # this ensures that H > 0 

    g = rand(Uniform(-5, 5), n)
    
    # x = rand(n)
    # diff_x = rand(Uniform(0, 5), n)
    # x_l = x .- diff_x
    # x_u = x .+ diff_x

    # Choose which bound constraints are active at start
    n_active = 2
    active_idx = randperm(n)[1:n_active]

    x = rand(Uniform(-3, 3), n)
    diff_x = rand(Uniform(0.5, 3), n)
    x_l = x .- diff_x
    x_u = x .+ diff_x

    # Force x to sit exactly on upper bound for active indices
    x[active_idx] .= x_u[active_idx]

    A = rand(Uniform(-2, 2), n, m_a)
    println("rank(A) = $(rank(A))")
    y = A' * x
    diff_Ax = rand(Uniform(0, 5), m_a)
    # print(size(diff_Ax))
    b_l = y .- diff_Ax
    b_u = y .+ diff_Ax
    
 
    return H, g, A, b_l, b_u, x_l, x_u, x
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
n_dim = 23
n_con = 30
H, g, A, b_l, b_u, x_l, x_u, x0 = generate_test_problem(n_dim, n_con) # TODO: remove x0 feasible 
#println("H: $H")
#println("g: $g") 
#println("A: $A")
#println("b_l: $b_l") 
#println("b_u: $b_u")
#println("x_l: $x_l")
#println("x_u: $x_u")
#println("Number of dimension: $n_dim")
#println("Number of constraints w. x constraints: $(n_con+n_dim) or $((n_con + n_dim) * 2) in standard form")

library_solution = library_solver(H, g, A, b_l, b_u, x_l, x_u)
println("library_solution = $library_solution")
# plot_qp(H, g, A, b_l, b_u, x_l, x_u; x_star=nothing)

# e and f 
""" 
We will implement a primal active-set algorithm. 
"""
function convex_active_set_solver(A, b, G, g, x0)
    """ 
    This solves the general 
    min 1/2 x' G x + g' x 
     x
    s.t. A' x >= b. 
    """
    
    # find feasible point (initial point)

    tol = 1e-14
    err = 1
    k = 1
    
    n_vars, m_const = size(A)
    x_list = [x0]

    # find initial working set
    # display(A)
    # display(x0)
    # display(b)
    # display(A' * x0 .== b)
    # println("n_vars, m_const = $n_vars, $m_const")

    W_set = Array(1:m_const)[A' * x0 .== b] #index of working set
    W_not_set = Array(1:m_const)[A' * x0 .!= b]
    
    converged = false 
    #TODO: right now it only checks if k < 100 since we DO NOT update err and converged!!
    # while err > tol && k < 10_000 && !converged
    while k < 50_000
        # solve the equality constraint

        # println("W_set = ")
        # display(W_set)
        A_W = A[:, W_set]
        x_k = x_list[end]
        k += 1

        # display(A_W)
        # println(size(A_W))
        n_W = size(A_W, 2)
        # println("det(G) = $(det(G))")
        KKT_matrix = [
            G       -A_W; 
            -A_W'   zeros(n_W, n_W) # check A.size[2] 
        ]
        # println("det(KKT_matrix) = $(det(KKT_matrix))")
        # display(KKT_matrix)


        KKT_rhs = -[
            G * x_k + g; 
            zeros(n_W) # again(!) check A.size[2]
        ] 

        res = KKT_matrix \ KKT_rhs

        p = res[1: n_vars]
        mu = res[n_vars+1:end]

        if norm(p, Inf) <= tol #||p^*|| = 0 
            if all(x -> x >= 0, mu)
                x_sol = x_k 
                mu_sol = mu # should only be those in W that are us rest should be zero
                push!(x_list, x_sol)
                # break # feasible solution is found
                break 
            else
                index_drop = argmin(mu)
                global_index = W_set[index_drop]
                
                W_set = deleteat!(W_set, index_drop)
                push!(x_list, x_k)

                push!(W_not_set, global_index)
                sort!(W_not_set) # just in case 
            end 
        else 
            # compute distance 

            new_set = A' * p .< 0 .&& in.(1:m_const, Ref(W_not_set))
            b_nw = b[new_set]
            A_nw = A[:, new_set]
            
            # println("shape b_nw: $(size(b_nw))")
            result = (b_nw - A_nw' * x_k) ./ (A_nw' * p)
            alpha = minimum(result)
            j = argmin(result) 

            
            if alpha < 1
                push!(x_list, x_k + alpha * p)
                # add constraint to working set
                # println("new_set", new_set, j)
                new_constraint_index_global = findall(new_set)[j] #findall returns index of all 
                push!(W_set, new_constraint_index_global)
                sort!(W_set)

                filter!(x -> x != new_constraint_index_global, W_not_set)  # ✅ add this


                #A_W_candidate = A[:, [W_set; new_constraint_index_global]]
                #if rank(A_W_candidate) > rank(A_W)
                #    push!(W_set, new_constraint_index_global)
                #    sort!(W_set)
                #end
            else
                push!(x_list, x_k + p)
                # keep working set constant to what it currently is
            end
        end
        
        # update converged 
        # rL = G * x_list[end] + g - A * mu 
 
        converged = false 
    end
        
    return x_list[end], k#, mu_sol 
end


I_matrix = Matrix{Float64}(I, n_dim, n_dim)
# println("I_matrix = ")
# display(I_matrix)
A_hat = [A -A I_matrix -I_matrix]
#print(A_hat)
b_hat = [b_l; -b_u; x_l; -x_u]
# println("Starting active set solver w.")
# println("A_shape: $(size(A_hat))")
# println("b_shape: $(size(b_hat))")
x_sol, k = convex_active_set_solver(A_hat, b_hat, H, g, x0)
x_sol, k = @time convex_active_set_solver(A_hat, b_hat, H, g, x0)
println("x_sol = $x_sol")
println("k: $k")
# display(mu_sol)
display(norm(x_sol - library_solution, Inf))
