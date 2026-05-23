using LinearAlgebra, Plots, SparseArrays
using JuMP
using Ipopt
using Random, Distributions
using Plots
using Measures

include("helpers.jl")
include("revised_simplex.jl")
include("test_problems.jl")

# c. 
"""
We consider the *convex* QP in the form 
min_{x} phi = 1/2 x' H x + g' x 
s.t. b_l <= A' x <= b_u, 
     x_l <= x <= x_u. 

Since this a convex(!) program then H > 0, 
that is, H is positive definite. 
"""


""" 
We will implement a primal active-set algorithm. 
"""
function convex_active_set_solver(A, b, H, g, x0)
    """ 
    This solves the general 
    min 1/2 x' H x + g' x 
     x
    s.t. A' x >= b. 
    taking as input the problem and a feasible point
    """

    tol = 1e-6
    k = 1
    max_number_of_iterations = 10_000
    
    n_vars, m_const = size(A)
    x_list = [x0]

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
            H       -A_W; 
            -A_W'   zeros(n_W, n_W) 
        ]

        KKT_rhs = -[
            H * x_k + g; 
            zeros(n_W) 
        ] 
        res = LDL_solver(KKT_matrix, KKT_rhs)
        p = res[1:n_vars]
        mu = res[n_vars+1:end]

        if norm(p, Inf) <= tol 
            if all(x -> x >= 0, mu)
                x_sol = x_k 
                mu_sol = mu # should only be those in W that are non-zero rest should be zero
                push!(x_list, x_sol)
                converged = true
                break 
            else
                index_drop = argmin(mu)
                global_index = W_set[index_drop]
                
                W_set = deleteat!(W_set, index_drop)
                push!(x_list, x_k)

                push!(W_not_set, global_index)
                sort!(W_not_set)
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
        rL = H * x_list[end] + g - A_W * mu
        rx = x_list[end] - x_list[end-1]
        converged = norm(rL, Inf) < tol && norm(rx, Inf) < tol && norm(mu, Inf) < tol 
    end
    return x_list[end], k, converged, x_list
end

function solve_convex_problem(H, g, A, b_l, b_u, x_l, x_u, n_dim)
    # Takes qp convex problem as input in form created by test problem:
    # min_{x} phi = 1/2 x' H x + g' x 
    # s.t. b_l <= A' x <= b_u, 
    # x_l <= x <= x_u. 

    #convert to standard form
    A_std, b_std, g_std = to_standard_form(g, A, b_l, b_u, x_l, x_u)
    n_std = length(g_std)

    #Define feasible point problem
    A_fp, b_fp, g_fp, x0 = fp_standard_form(A_std, b_std)

    # Run simplex on FP problem
    result_fp = revised_simplex(A_fp, b_fp, g_fp, x0)
    if !result_fp.optimal
        print(result_fp)
        error("Phase 1 failed - problem may be infeasible")
    end

    #Get feasible point (exclude slack vars)
    x0_shifted = result_fp.x[1:size(H)[1]]
    x0_true = x0_shifted + x_l # recover non-shifted x

    I_matrix = Matrix{Float64}(I, n_dim, n_dim)

    # Build our problem on standard form but now without slack variables which used only for simplex.
    # So Ax >= b
    A_hat = [A -A I_matrix -I_matrix]
    b_hat = [b_l; -b_u; x_l; -x_u]
    x_sol, k, converged, x_walk = convex_active_set_solver(A_hat, b_hat, H, g, x0_true)
    return x_sol, k, converged, x_walk
end

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




function primal_dual_qp_ineq(H, g, C, d, x0, z0, s0, tol=1e-6, maxiter=5000, η=0.995)
    n = size(H, 1) #number of vars
    mc = size(C, 2)  # Number of inequality constraints
    
    # Initialize
    x = x0 
    z = z0
    s = s0
    
    # Ensure strictly positive
    z .= max.(z, 1.0)
    s .= max.(s, 1.0)
    
    e = ones(mc)

    history = Dict(:μ, :dual_res, :primal_res, :obj, :x)

    dual_res   = Inf
    primal_res = Inf
    μ          = Inf

    push!(history[:x], copy(x))  # store initial point

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

        F = let
            δ = 0.0
            local fac
            success = false
            # this is how we handle possible singularity - by adding small value on diagonal
            while !success
                try
                    fac = cholesky(Symmetric(Hbar + δ * I))
                    success = true
                catch
                    δ = (δ == 0.0) ? 1e-8 : δ * 10 
                    δ > 1e-2 && error("Cholesky failed even with δ=$δ regularization")
                end
            end
            fac
        end
        
        # Solve for affine direction
        r_hat_L = rL - C * S_inv_Z * (rC - rsz ./ z)
        dx_aff = -(F \ r_hat_L)
        
        # Back-substitution
        dz_aff = -S_inv_Z * C' * dx_aff + S_inv_Z * (rC - rsz ./ z)
        
        ds_aff = -(rsz ./ z) - (s ./ z) .* dz_aff
        
        # Maximum step length
        α_aff = compute_max_step(z, s, dz_aff, ds_aff, 1.0) 
        
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

        z .= max.(z, 1e-8)
        s .= max.(s, 1e-8)
        push!(history[:x], copy(x))
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
    """
    Transform from 
        min_{x} phi = 1/2 x' H x + g' x 
        s.t. b_l <= A' x <= b_u, 
        x_l <= x <= x_u
    to
        min_{x} phi = 1/2 x' H x + g' x 
        s.t. Cx >= d,
    """
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


function solve_with_commercial(H, g, A, b_l, b_u, x_l, x_u)
    n = size(H, 1)
    
    # Create model
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6, "print_level" => 0))
    
    # Variables
    @variable(model, x_l[i] <= x[i=1:n] <= x_u[i])

    # Objective: 0.5x'Hx + g'x
    @objective(model, Min, 0.5 * x' * H * x + g' * x)
    
    # General linear constraints: b_l ≤ A'x ≤ b_u
    @constraint(model, b_l .<= A' * x .<= b_u)
    
    optimize!(model)
    # Extract solution
    return (
        x = value.(x),
        status = termination_status(model),
        obj = objective_value(model),
        solve_time = solve_time(model),
        iter = MOI.get(model, MOI.BarrierIterations())
    )
end

## For plotting
function benchmark_vs_nvars(n_range, m_fixed)
    times_lib = []
    times_act = []
    times_ip  = []
    iters_lib = []
    iters_act = []  
    iters_ip  = []

    for n in n_range
        println("  Variables range: n=$n, m=$m_fixed")
        H, g, A, b_l, b_u, x_l, x_u = generate_test_problem_qp(n, m_fixed)

        # Library (Ipopt)
        try
            t = @timed solve_with_commercial(H, g, A, b_l, b_u, x_l, x_u)
            push!(times_lib, t.time)
            push!(iters_lib, t.value.iter)
        catch e
            println("  Library failed: $e")
            push!(times_lib, NaN)
            push!(iters_lib, -1)
        end

        # Active-set
        try
            t = @timed solve_convex_problem(H, g, A, b_l, b_u, x_l, x_u, n)
            push!(times_act, t.time)
            _, k, conv, _ = t.value
            push!(iters_act, conv ? k : -1)
        catch e
            println("  Active-set failed: $e")
            push!(times_act, NaN)
            push!(iters_act, -1)
        end

        # Interior point (include setup in timing for fairness)
        try
            t = @timed begin
                C, d = setup_qp_with_bounds(H, g, A, b_l, b_u, x_l, x_u)
                primal_dual_qp_ineq(H, g, C, d)
            end
            push!(times_ip, t.time)
            push!(iters_ip, t.value.iter)
        catch e
            println("  Interior point failed: $e")
            push!(times_ip, NaN)
            push!(iters_ip, -1)
        end
    end

    return (times_lib=times_lib, iters_lib=iters_lib,
            times_act=times_act, iters_act=iters_act,
            times_ip=times_ip,   iters_ip=iters_ip)
end

function benchmark_vs_ncons(m_range, n_fixed)
    times_lib = []
    times_act = []
    times_ip  = []
    iters_lib = []
    iters_act = []
    iters_ip  = []

    for m in m_range
        println("Constraints range: n=$n_fixed, m=$m")
        H, g, A, b_l, b_u, x_l, x_u = generate_test_problem_qp(n_fixed, m)

        try
            t = @timed solve_with_commercial(H, g, A, b_l, b_u, x_l, x_u)
            push!(times_lib, t.time)
            push!(iters_lib, t.value.iter)
        catch e
            println("  Library failed: $e")
            push!(times_lib, NaN); push!(iters_lib, -1)
        end

        try
            t = @timed solve_convex_problem(H, g, A, b_l, b_u, x_l, x_u, n_fixed)
            push!(times_act, t.time)
            _, k, conv, _ = t.value
            push!(iters_act, conv ? k : -1)
        catch e
            println("  Active-set failed: $e")
            push!(times_act, NaN); push!(iters_act, -1)
        end

        try
            t = @timed begin
                C, d = setup_qp_with_bounds(H, g, A, b_l, b_u, x_l, x_u)
                primal_dual_qp_ineq(H, g, C, d)
            end
            push!(times_ip, t.time)
            push!(iters_ip, t.value.iter)
        catch e
            println("  Interior point failed: $e")
            push!(times_ip, NaN); push!(iters_ip, -1)
        end
    end

    return (times_lib=times_lib, iters_lib=iters_lib,
            times_act=times_act, iters_act=iters_act,
            times_ip=times_ip,   iters_ip=iters_ip)
end

# Convert -1 (non-convergence marker) to NaN for plotting
iters_as_float(v::Vector{Int}) = [i == -1 ? NaN : Float64(i) for i in v]

function create_benchmark_plots(res_vars, n_range, m_fixed,
                                 res_cons, m_range, n_fixed)
    p1 = plot(n_range, res_vars.times_lib,
              label="Library (Ipopt)", lw=2, marker=:circle,
              xlabel="Number of variables (n)", ylabel="CPU time [s]",
              title="CPU time vs. variables\n(m = $m_fixed fixed)", yscale=:log10,
              left_margin=8mm)
    plot!(p1, n_range, res_vars.times_act, label="Active-set",     lw=2, marker=:square)
    plot!(p1, n_range, res_vars.times_ip,  label="Interior point", lw=2, marker=:diamond)

    p2 = plot(n_range, iters_as_float(res_vars.iters_lib),
              label="Library (Ipopt)", lw=2, marker=:circle,
              xlabel="Number of variables (n)", ylabel="Iterations",
              title="Iterations vs. variables\n(m = $m_fixed fixed)",
              yscale=:log10, left_margin=8mm)
    plot!(p2, n_range, iters_as_float(res_vars.iters_act), label="Active-set",     lw=2, marker=:square)
    plot!(p2, n_range, iters_as_float(res_vars.iters_ip),  label="Interior point", lw=2, marker=:diamond)

    p3 = plot(m_range, res_cons.times_lib,
              label="Library (Ipopt)", lw=2, marker=:circle,
              xlabel="Number of constraints (m)", ylabel="CPU time [s]",
              title="CPU time vs. constraints\n(n = $n_fixed fixed)", yscale=:log10,
              left_margin=8mm)
    plot!(p3, m_range, res_cons.times_act, label="Active-set",     lw=2, marker=:square)
    plot!(p3, m_range, res_cons.times_ip,  label="Interior point", lw=2, marker=:diamond)

    p4 = plot(m_range, iters_as_float(res_cons.iters_lib),
              label="Library (Ipopt)", lw=2, marker=:circle,
              xlabel="Number of constraints (m)", ylabel="Iterations",
              title="Iterations vs. constraints\n(n = $n_fixed fixed)",
              yscale=:log10, left_margin=8mm)
    plot!(p4, m_range, iters_as_float(res_cons.iters_act), label="Active-set",     lw=2, marker=:square)
    plot!(p4, m_range, iters_as_float(res_cons.iters_ip),  label="Interior point", lw=2, marker=:diamond)

    fig = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 900), legend=:topleft)
    savefig(fig, "handin/im/problem2_benchmark.png")
    println("Saved benchmark plots to problem2_benchmark.png")
    return fig
end

function plot_qp_walks(H, g, A, b_l, b_u, x_l, x_u)
    n = length(g)
    @assert n == 2 "Walk plot only supported for 2D problems"

    # Active-set walk (includes x_walk in original space)
    x_act, _, _, x_walk_act = solve_convex_problem(H, g, A, b_l, b_u, x_l, x_u, n)

    # Feasible starting point for IP (reuse active-set's starting point)
    x0_walk = x_walk_act[1]

    # Interior-point walk
    C, d = setup_qp_with_bounds(H, g, A, b_l, b_u, x_l, x_u)
    result_ip = primal_dual_qp_ineq(H, g, C, d; x0=x0_walk)
    x_walk_ip = result_ip.history[:x]

    # Ipopt reference
    res_lib = solve_with_commercial(H, g, A, b_l, b_u, x_l, x_u)
    x_lib = res_lib.x

    # Determine plot range from all iterates
    all_pts = vcat(x_walk_act, x_walk_ip, [x_lib])
    all_x1  = [p[1] for p in all_pts]
    all_x2  = [p[2] for p in all_pts]
    pad = 1.0
    x1_range = range(min(minimum(all_x1), x_l[1]) - pad, max(maximum(all_x1), x_u[1]) + pad, length=300)
    x2_range = range(min(minimum(all_x2), x_l[2]) - pad, max(maximum(all_x2), x_u[2]) + pad, length=300)

    f(x1, x2) = 0.5 * [x1, x2]' * H * [x1, x2] + g' * [x1, x2]
    Z = [f(x1, x2) for x2 in x2_range, x1 in x1_range]

    plt = contour(x1_range, x2_range, Z,
                  levels=30, color=:viridis, linewidth=0.8,
                  xlabel="x₁", ylabel="x₂",
                  title="QP solution walks (n=2)",
                  colorbar=true, size=(900, 700))

    # Shade feasible region
    feasible = [
        Float64(all(b_l .<= A' * [x1; x2] .<= b_u) && all(x_l .<= [x1; x2] .<= x_u))
        for x2 in x2_range, x1 in x1_range
    ]
    contourf!(plt, x1_range, x2_range, feasible,
              levels=[0.5, 1.5], fillcolor=:green, fillalpha=0.15,
              linewidth=0, colorbar=false, label="Feasible")

    # Constraint boundaries
    for i in 1:size(A, 2)
        a = A[:, i]
        if abs(a[2]) > 1e-8
            f_u(x1) = (b_u[i] - a[1]*x1) / a[2]
            f_l(x1) = (b_l[i] - a[1]*x1) / a[2]
            plot!(plt, x1_range, clamp.(f_u.(x1_range), minimum(x2_range), maximum(x2_range)),
                  color=:gray, alpha=0.4, label=false, lw=1.0)
            plot!(plt, x1_range, clamp.(f_l.(x1_range), minimum(x2_range), maximum(x2_range)),
                  color=:gray, alpha=0.4, label=false, lw=1.0)
        end
    end

    # Box bounds
    plot!(plt,
          [x_l[1], x_u[1], x_u[1], x_l[1], x_l[1]],
          [x_l[2], x_l[2], x_u[2], x_u[2], x_l[2]],
          color=:black, lw=2.0, linestyle=:dash, label="bounds")

    # Active-set walk (vertex-to-vertex steps)
    act_x1 = [p[1] for p in x_walk_act]
    act_x2 = [p[2] for p in x_walk_act]
    plot!(plt, act_x1, act_x2,
          color=:blue, lw=2.0, marker=:square, markersize=4, label="Active-set")
    scatter!(plt, [act_x1[end]], [act_x2[end]],
             color=:blue, markersize=10, marker=:star5, label=false)

    # Interior-point walk (smooth interior path)
    ip_x1 = [p[1] for p in x_walk_ip]
    ip_x2 = [p[2] for p in x_walk_ip]
    plot!(plt, ip_x1, ip_x2,
          color=:red, lw=2.0, marker=:circle, markersize=4, label="Interior point")
    scatter!(plt, [ip_x1[1]], [ip_x2[1]],
             color=:red, markersize=7, marker=:diamond, label="IP start")
    scatter!(plt, [ip_x1[end]], [ip_x2[end]],
             color=:red, markersize=10, marker=:star5, label=false)

    # Ipopt reference
    scatter!(plt, [x_lib[1]], [x_lib[2]],
             color=:green, markersize=12, marker=:star5, label="Ipopt")

    savefig(plt, "handin/im/qp_solution_walks.png")
    println("Saved QP solution walk to qp_solution_walks.png")
    return plt
end

## Run benchmarks
n_range = collect(20:20:300)   
m_fixed = 100
n_fixed = 300
m_range = collect(10:10:100)  

println("Benchmarking for range of variables (m=$m_fixed fixed)")
res_vars = benchmark_vs_nvars(n_range, m_fixed)

println("Benchmarking for range of constraints (n=$n_fixed fixed)")
res_cons = benchmark_vs_ncons(m_range, n_fixed)

create_benchmark_plots(res_vars, n_range, m_fixed, res_cons, m_range, n_fixed)

println("\n=== 2D QP solution walk ===")
H2, g2, A2, b_l2, b_u2, x_l2, x_u2 = generate_test_problem_qp(2, 4)
plot_qp_walks(H2, g2, A2, b_l2, b_u2, x_l2, x_u2)