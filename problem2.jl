using LinearAlgebra, Plots, SparseArrays
using JuMP
using Ipopt
using Random, Distributions
using Plots

include("problem2_presolve.jl")
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
 
function generate_random_test_problem(n,alpha,density) # following MATLAB implementation 
    m = 10 * n

    A = sprandn(n, m, density)

    b_l = -rand(m)
    b_u = rand(m)

    M = sprand(n, n, density)
    H = M * M' + alpha * I
    g = randn(n)

    x_l = -ones(n)
    x_u = ones(n)

    return H, g, b_l, b_u, x_l, x_u, nothing
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
n_dim = 30
n_con = 20
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

    tol = 1e-7
    err = 1 #?
    k = 1
    max_number_of_iterations = 10_000
    
    n_vars, m_const = size(A)
    x_list = [x0]

    # find initial working set
    # display(A)
    # display(x0)
    # display(b)
    # display(A' * x0 .== b)
    # println("n_vars, m_const = $n_vars, $m_const")
    # active_tolerance = tol 
    # W_set = Array(1:m_const)[A' * x0 - b .< active_tolerance] #index of working set
    # W_not_set = Array(1:m_const)[A' * x0 - b .> active_tolerance]
    W_set = Array(1:m_const)[A' * x0 .== b] #index of working set
    W_not_set = Array(1:m_const)[A' * x0 .!= b]

    converged = false 
    while !converged && k < max_number_of_iterations
        # solve the equality constraint
        A_W = A[:, W_set]

        x_k = x_list[end]
        k += 1

        n_W = size(A_W, 2)
        KKT_matrix = [
            G       -A_W; 
            -A_W'   zeros(n_W, n_W) # check A.size[2] 
        ]

        KKT_rhs = -[
            G * x_k + g; 
            zeros(n_W) 
        ] 

        res = KKT_matrix \ KKT_rhs

        p = res[1:n_vars]
        mu = res[n_vars+1:end]
        #display(mu)

        if norm(p, Inf) <= tol #||p^*|| = 0 
            if all(x -> x >= 0, mu)
                x_sol = x_k 
                mu_sol = mu # should only be those in W that are us rest should be zero
                push!(x_list, x_sol)
                converged = true
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
            
            result = (b_nw - A_nw' * x_k) ./ (A_nw' * p)
            alpha = minimum(result)
            j = argmin(result) 
            
            if alpha < 1
                push!(x_list, x_k + alpha * p)
                # add constraint to working set
                new_constraint_index_global = findall(new_set)[j] #findall returns index of all 
                push!(W_set, new_constraint_index_global)
                sort!(W_set)

                filter!(x -> x != new_constraint_index_global, W_not_set) 
            else
                push!(x_list, x_k + p)
                # keep working set constant to what it currently is
            end
        end
        
        # update converged 
        rL = G * x_list[end] + g - A_W * mu
        rx = x_list[end] - x_list[end-1]
        
        converged = norm(rL, Inf) < tol && norm(rx, Inf) < tol && norm(mu, Inf) < tol 
    end
    
    return x_list[end], k#, mu_sol 
end


# I_matrix = Matrix{Float64}(I, n_dim, n_dim)


# A_hat = [A -A I_matrix -I_matrix]
# b_hat = [b_l; -b_u; x_l; -x_u]
# x_sol, k = convex_active_set_solver(A_hat, b_hat, H, g, x0)
# x_sol, k = @time convex_active_set_solver(A_hat, b_hat, H, g, x0)
# println("x_sol = $x_sol")
# println("k: $k")
# display(norm(x_sol - library_solution, Inf))

# g and h 
print("\n\n\n\n")
println("h)")
function convex_dual_interior_point_solver(H, g, C, d, s, z)
    """ 
    Solves 
    min 1/2 x' H x + g' x 
    s.t. C' x >= d

    Call with (s, z) = (1, ..., 1). 
    """ 

    # see 6.5, slide 6 
    S = diagm(s)
    Z = diagm(z)

    _, nc = size(C) 
end







function primal_dual_qp_ineq(
    H::AbstractMatrix,
    g::AbstractVector,
    C::AbstractMatrix,
    d::AbstractVector;
    x0=nothing,
    z0=nothing,
    s0=nothing,
    tol=1e-8,
    maxiter=500,
    η=0.995
)
    n = size(H, 1)
    mc = size(C, 2)  # Number of inequality constraints
    
    # Initialize
    x = x0 === nothing ? zeros(n) : copy(x0)
    z = z0 === nothing ? ones(mc) : copy(z0)
    s = s0 === nothing ? ones(mc) : copy(s0)
    
    # Ensure strictly positive
    z .= max.(z, 1.0)
    s .= max.(s, 1.0)
    
    e = ones(mc)
    
    history = Dict(
        :μ => Float64[],
        :dual_res => Float64[],
        :primal_res => Float64[],
        :obj => Float64[]
    )
    
    for k in 1:maxiter
        # Compute residuals
        rL  = H * x + g - C * z  
        rC  = s + d - C' * x
        rsz = s .* z
        μ   = dot(s, z) / mc
        
        # Check convergence
        dual_res   = norm(rL, Inf)
        primal_res = norm(rC, Inf)
        obj        = 0.5 * dot(x, H * x) + dot(g, x)
        
        push!(history[:μ], μ)
        push!(history[:dual_res], dual_res)
        push!(history[:primal_res], primal_res)
        push!(history[:obj], obj)
        
        if maximum([dual_res, primal_res, μ]) < tol
            return (
                x = x,
                z = z,
                s = s,
                status = :optimal,
                iter = k,
                history = history
            )
            break
        end
        
        # affine Direction 
        S_inv_Z = Diagonal(z ./ s)  # S^{-1}Z
        Hbar = H + C * S_inv_Z * C'

        # Cholesky
        F = cholesky(Symmetric(Hbar))
        
        # Solve for affine direction
        r_hat_L = rL - C * S_inv_Z * (rC - rsz ./ z)
        dx_aff = -(F \ r_hat_L)
        
        # Back-substitution
        dz_aff = -S_inv_Z * C' * dx_aff + S_inv_Z * (rC - rsz ./ z)
        
        ds_aff = -(rsz ./ z) - (s ./ z) .* dz_aff
        
        # Maximum step length
        α_aff = compute_max_step(z, s, dz_aff, ds_aff, 1.0) # multiplies with 1 so all good! 
        
        # Affine duality gap
        μ_aff = dot(z + α_aff * dz_aff, s + α_aff * ds_aff) / mc
        
        # Centering parameter
        σ = (μ_aff / μ)^3
        
        # === Centering-Corrected Direction ===
        rsz_bar = rsz + diagm(ds_aff) * diagm(dz_aff) * e .- σ * μ * e  
        
        r_hat_L_bar = rL - C * S_inv_Z * (rC - rsz_bar ./ z)
        dx = -(F \ r_hat_L_bar)
        
        dz = -S_inv_Z * (C' * dx) + S_inv_Z * (rC - rsz_bar ./ z)
        ds = -(rsz_bar ./ z) - (s ./ z) .* dz
        
        # Maximum step length with damping
        α = compute_max_step(z, s, dz, ds, η)
        
        # Update
        x .+= α * dx 
        z .+= α * dz
        s .+= α * ds
        
        z .= max.(z, 1e-14)
        s .= max.(s, 1e-14)

        println("x[$k] = $x")
    end
    
    return (
        x = x,
        z = z,
        s = s,
        status = :maxiter,
        iter = maxiter,
        history = history
    )
end

function compute_max_step(z, s, dz, ds, η)
    # Find maximum α such that z + α*dz ≥ 0 and s + α*ds ≥ 0
    

    neg_dz = dz .< 0
    if any(neg_dz)
        α_z = minimum(-z[neg_dz] ./ dz[neg_dz])
    else
        α_z = Inf
    end
    
    neg_ds = ds .< 0
    if any(neg_ds)
        α_s = minimum(-s[neg_ds] ./ ds[neg_ds])
    else
        α_s = Inf
    end
    
    α_max = min(1.0, α_z, α_s)
    return η * α_max
end


function setup_qp_with_bounds(H, g, A, b_l, b_u, x_l, x_u)
    n = size(H, 1)      
    m = length(b_l)    
    
    C_transpose = [
        A'           # A'x ≥ b_l
        -A'          # -A'x ≥ -b_u 
        Matrix(I, n, n)   # x ≥ x_l
        -Matrix(I, n, n)  # -x ≥ -x_u 
    ]
    
    # Build d vector
    d = [
        b_l
        -b_u
        x_l
        -x_u
    ]
    # C is the transpose of C_transpose
    C = C_transpose'
    
    return C, d
end


H = [2.0 0.0; 0.0 2.0]
g = [-2.0; -5.0]
A = [1.0 1.0; 1.0 2.0]'  # 2 constraints
b_l = [5.0; 5.0]
b_u = [10.0; 10.0]
x_l = [0.0; 0.0]
x_u = [100.0; 100.0]


function solve_with_commercial(H, g, A, b_l, b_u, x_l, x_u)
    n = size(H, 1)
    
    # Create model
    model = Model(Ipopt.Optimizer)
    set_silent(model)  # Suppress output
    
    # Variables
    @variable(model, x_l[i] <= x[i=1:n] <= x_u[i])

    # Objective: ½x'Hx + g'x
    @objective(model, Min, 0.5 * x' * H * x + g' * x)
    
    # General linear constraints: b_l ≤ A'x ≤ b_u
    @constraint(model, b_l .<= A' * x .<= b_u)


    # Solve
    optimize!(model)
    
    # Extract solution
    return (
        x = value.(x),
        status = termination_status(model),
        obj = objective_value(model),
        solve_time = solve_time(model)
    )
end


# ===== Comparison =====

# Your implementation
C, d = setup_qp_with_bounds(H, g, A, b_l, b_u, x_l, x_u)
result_custom = primal_dual_qp_ineq(H, g, C, d)

# Commercial solver
result_commercial = solve_with_commercial(H, g, A, b_l, b_u, x_l, x_u)

# Compare
println("="^60)
println("SOLUTION COMPARISON")
println("="^60)
println("\nCustom Primal-Dual Interior-Point:")
println("  Status: ", result_custom.status)
println("  Iterations: ", result_custom.iter)
println("  x = ", round.(result_custom.x, digits=6))
println("  Objective = ", round(0.5 * dot(result_custom.x, H * result_custom.x) + dot(g, result_custom.x), digits=8))
println("  Final μ = ", round(result_custom.history[:μ][end], sigdigits=3))

println("\nCommercial Solver (Ipopt):")
println("  Status: ", result_commercial.status)
println("  x = ", round.(result_commercial.x, digits=6))
println("  Objective = ", round(result_commercial.obj, digits=8))
println("  Solve time = ", round(result_commercial.solve_time, digits=4), " seconds")

println("\nDifference:")
println("  ||x_custom - x_commercial|| = ", norm(result_custom.x - result_commercial.x))
println("  Objective difference = ", abs(
    0.5 * dot(result_custom.x, H * result_custom.x) + dot(g, result_custom.x) - 
    result_commercial.obj
))


p1 = plot(result_custom.history[:μ], 
          yscale=:log10, 
          xlabel="Iteration", 
          ylabel="Duality Gap μ",
          label="μ",
          lw=2,
          marker=:circle,
          title="Convergence History")

p2 = plot(result_custom.history[:dual_res], 
          yscale=:log10,
          xlabel="Iteration", 
          ylabel="Residual",
          label="Dual residual",
          lw=2,
          marker=:circle)
plot!(p2, result_custom.history[:primal_res], 
      label="Primal residual",
      lw=2,
      marker=:square)

p3 = plot(result_custom.history[:obj],
          xlabel="Iteration",
          ylabel="Objective Value",
          label="Objective",
          lw=2,
          marker=:circle)
hline!(p3, [result_commercial.obj], 
       label="Commercial solver",
       linestyle=:dash,
       lw=2)

plot(p1, p2, p3, layout=(1,3), size=(1200,400))
savefig("problem2-h.png")