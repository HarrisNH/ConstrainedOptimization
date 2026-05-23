using LinearAlgebra, Plots, SparseArrays
using JuMP
using Ipopt
using Random, Distributions
using Plots

using Measures

include("test_problems.jl")
include("revised_simplex.jl")
include("helpers.jl")

function factorize_H(H)
    try
        F = cholesky(Symmetric(H + 1e-8 * I))
        #println("Using Cholesky")
        return F
    catch end

    try
        F = bunchkaufman(Symmetric(H))
        #println("Using Bunchkaufman")
        return F
    catch end

    try
        F = lu(H)
        #println("Using LU")
        return F
    catch end

    error("Cholesky, Bunch-Kaufman and LU all failed — matrix is singular")
end

# d 
function library_solver_lp(g, A, b_l, b_u, x_l, x_u)
    n = size(A)[1]
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    @variable(model, x[1:n])
    @constraint(model, b_l .<= A' * x .<= b_u)
    @constraint(model, x_l .<= x .<= x_u)
    @objective(model, Min, g' * x)
    optimize!(model)
    return value.(x)
end

# e and f 
function primal_dual_interior_LP(g, A, b, x0, n_orig, x_l)
    m, n = size(A)
    maxiter = 100
    tol = 1.0e-3

    lambda = ones(n, 1)
    mu = zeros(m, 1)
    x = x0

    rL = g - A' * mu - lambda 
    rA = A * x0 - b 
    rC = x0 .* lambda
    s = sum(rC) / n

    converged = (norm(rL, Inf) < tol) && (norm(rA, Inf) < tol) && (norm(s, Inf) < tol)
    k = 0
    history = [x0[1:n_orig] .+ x_l]  # store original-space iterates
    
    while !converged & (k < maxiter)
        k = k + 1
        
        xDivlambda = vec(x ./ lambda)
        H = A * diagm(xDivlambda) * A'

        F = factorize_H(H)
        
        tmp = (x .* rL + rC) ./ lambda
        rhs = -rA + A * tmp

        dmu = isnothing(F) ? H \ rhs : F \ rhs
        dx = xDivlambda .* (A' * dmu) - tmp 
        dlambda = -(rC + lambda .* dx) ./ x

        idx = findall(x -> x < 0.0, dx) 
        alpha = minimum([1.0; .-x[idx] ./ dx[idx]])
        idx = findall(x -> x < 0.0, dlambda)
        beta = minimum([1.0; .-lambda[idx] ./ dlambda[idx]])

        x_aff = x + alpha * dx 
        lambda_aff = lambda + beta * dlambda
        s_aff = sum(x_aff .* lambda_aff) / n 

        sigma = (s_aff / s)^3 
        tau = sigma * s 

        rC = rC + dx .* dlambda .- tau 

        tmp = (x .* rL + rC) ./ lambda
        rhs = -rA + A * tmp 

        dmu = isnothing(F) ? H \ rhs : F \ rhs
        dx = xDivlambda .* (A' * dmu) - tmp 
        dlambda = -(rC + lambda .* dx) ./ x
        
        idx = findall(x -> x < 0.0, dx)
        alpha = minimum([1.0; .-x[idx] ./ dx[idx]])
        idx = findall(x -> x < 0.0, dlambda)
        beta = minimum([1.0; .-lambda[idx] ./ dlambda[idx]])

        eta = 0.995
        x = x + (eta * alpha) * dx 
        mu = mu + (eta * beta) * dmu 
        lambda = lambda + (eta * beta) * dlambda

        rL = g - A' * mu - lambda
        rA = A * x - b 
        rC = x .* lambda
        s = sum(rC) / n 
        
        push!(history, x[1:n_orig] .+ x_l)
        converged = (norm(rL, Inf) < tol) && (norm(rA, Inf) < tol) && (norm(s, Inf) < tol) 
    end

    return x, mu, lambda, history
end



function benchmark_lp_vs_nvars(n_range, m_fixed)
    times_ip  = Float64[]
    times_sim = Float64[]
    times_lib = Float64[]
    iters_ip  = Int[]
    iters_sim = Int[]
    iters_lib = Int[]

    for n in n_range
        #println("  LP vars sweep: n=$n, m=$m_fixed")
        g, A, b_l, b_u, x_l, x_u, _ = generate_test_problem_lp(n, m_fixed)

        # Standard form (shared by IP and simplex)
        A_std, b_std, g_std = to_standard_form(g, A, b_l, b_u, x_l, x_u)
        n_std = length(g_std)
        A_fp, b_fp, g_fp, x0 = fp_standard_form(A_std, b_std)
        result_fp = revised_simplex(A_fp, b_fp, g_fp, x0)
        x0_std = max.(result_fp.x[1:n_std], 1e-4)

        # Interior point
        try
            t = @timed primal_dual_interior_LP(g_std, A_std, b_std, x0_std)
            push!(times_ip, t.time)
            _, _, _ = t.value
            # k is printed but not returned — rerun just for iter count
            # workaround: time includes k=100 for non-convergence
            push!(iters_ip, -1)  # placeholder; see note below
        catch e
            println("  IP failed: $e")
            push!(times_ip, NaN); push!(iters_ip, -1)
        end

        # Simplex
        try
            t = @timed revised_simplex(A_std, b_std, g_std, x0_std)
            push!(times_sim, t.time)
            push!(iters_sim, t.value.optimal ? -1 : -2)
        catch e
            println("  Simplex failed: $e")
            push!(times_sim, NaN); push!(iters_sim, -1)
        end

        # Commercial (Ipopt)
        try
            t = @timed library_solver_lp(g, A, b_l, b_u, x_l, x_u)
            push!(times_lib, t.time)
            push!(iters_lib, -1)
        catch e
            println("  Library failed: $e")
            push!(times_lib, NaN); push!(iters_lib, -1)
        end
    end

    return (times_ip=times_ip, times_sim=times_sim, times_lib=times_lib,
            iters_ip=iters_ip, iters_sim=iters_sim, iters_lib=iters_lib)
end

function benchmark_lp_vs_ncons(m_range, n_fixed)
    times_ip  = Float64[]
    times_sim = Float64[]
    times_lib = Float64[]

    for m in m_range
        println("  LP cons sweep: n=$n_fixed, m=$m")
        g, A, b_l, b_u, x_l, x_u, _ = generate_test_problem_lp(n_fixed, m)

        A_std, b_std, g_std = to_standard_form(g, A, b_l, b_u, x_l, x_u)
        n_std = length(g_std)
        A_fp, b_fp, g_fp, x0 = fp_standard_form(A_std, b_std)
        result_fp = revised_simplex(A_fp, b_fp, g_fp, x0)
        x0_std = max.(result_fp.x[1:n_std], 1e-4)

        try
            t = @timed primal_dual_interior_LP(g_std, A_std, b_std, x0_std)
            push!(times_ip, t.time)
        catch e
            println("  IP failed: $e"); push!(times_ip, NaN)
        end

        try
            t = @timed revised_simplex(A_std, b_std, g_std, x0_std)
            push!(times_sim, t.time)
        catch e
            println("  Simplex failed: $e"); push!(times_sim, NaN)
        end

        try
            t = @timed library_solver_lp(g, A, b_l, b_u, x_l, x_u)
            push!(times_lib, t.time)
        catch e
            println("  Library failed: $e"); push!(times_lib, NaN)
        end
    end

    return (times_ip=times_ip, times_sim=times_sim, times_lib=times_lib)
end

function create_lp_benchmark_plots(res_vars, n_range, m_fixed,
                                    res_cons, m_range, n_fixed)
    p1 = plot(n_range, res_vars.times_ip,
              label="Interior point", lw=2, marker=:diamond,
              xlabel="Number of variables (n)", ylabel="CPU time [s]",
              title="LP: CPU time vs variables\n(m=$m_fixed fixed)",
              yscale=:log10, left_margin=8mm)
    plot!(p1, n_range, res_vars.times_sim, label="Simplex",          lw=2, marker=:square)
    plot!(p1, n_range, res_vars.times_lib, label="Library (Ipopt)",  lw=2, marker=:circle)

    p2 = plot(m_range, res_cons.times_ip,
              label="Interior point", lw=2, marker=:diamond,
              xlabel="Number of constraints (m)", ylabel="CPU time [s]",
              title="LP: CPU time vs constraints\n(n=$n_fixed fixed)",
              yscale=:log10, left_margin=8mm)
    plot!(p2, m_range, res_cons.times_sim, label="Simplex",          lw=2, marker=:square)
    plot!(p2, m_range, res_cons.times_lib, label="Library (Ipopt)",  lw=2, marker=:circle)

    fig = plot(p1, p2, layout=(1, 2), size=(1100, 450), legend=:topleft)
    savefig(fig, "lp_benchmark_plot.png")
    println("Saved LP benchmark to lp_benchmark.png")
    return fig
end

# Run
n_range_lp = collect(5:5:50)
m_fixed_lp = 20
n_fixed_lp = 30
m_range_lp = collect(5:5:50)

println("=== LP: sweeping variables (m=$m_fixed_lp) ===")
res_vars_lp = benchmark_lp_vs_nvars(n_range_lp, m_fixed_lp)

println("=== LP: sweeping constraints (n=$n_fixed_lp) ===")
res_cons_lp = benchmark_lp_vs_ncons(m_range_lp, n_fixed_lp)

create_lp_benchmark_plots(res_vars_lp, n_range_lp, m_fixed_lp, res_cons_lp, m_range_lp, n_fixed_lp)
function plot_lp_solution_walks(g, A, b_l, b_u, x_l, x_u, n_dim)
    @assert n_dim == 2 "Solution walk plot only supported for 2D problems"

    A_std, b_std, g_std = to_standard_form(g, A, b_l, b_u, x_l, x_u)
    n_std = length(g_std)
    A_fp, b_fp, g_fp, x0_fp = fp_standard_form(A_std, b_std)
    result_fp = revised_simplex(A_fp, b_fp, g_fp, x0_fp)
    x0_std = max.(result_fp.x[1:n_std], 1e-4)

    x_ip, _, _, ip_history = primal_dual_interior_LP(g_std, A_std, b_std, x0_std, n_dim, x_l)
    x_ip_orig = x_ip[1:n_dim] .+ x_l

    result_sim = revised_simplex(A_std, b_std, g_std, x0_std)
    x_sim_orig = result_sim.x[1:n_dim] .+ x_l

    x_lib = library_solver_lp(g, A, b_l, b_u, x_l, x_u)

    # Collect all relevant points to determine axis range
    ip_x1 = [ip_history[i][1] for i in 1:length(ip_history)]
    ip_x2 = [ip_history[i][2] for i in 1:length(ip_history)]

    all_x1 = vcat(ip_x1, x_sim_orig[1], x_lib[1], x_l[1], x_u[1])
    all_x2 = vcat(ip_x2, x_sim_orig[2], x_lib[2], x_l[2], x_u[2])

    pad = 2.0
    x1_range = range(minimum(all_x1) - pad, maximum(all_x1) + pad, length=400)
    x2_range = range(minimum(all_x2) - pad, maximum(all_x2) + pad, length=400)

    Z = [g[1]*x1 + g[2]*x2 for x2 in x2_range, x1 in x1_range]

    plt = contour(x1_range, x2_range, Z,
                  levels=40, color=:viridis, linewidth=0.8,
                  xlabel="x₁", ylabel="x₂",
                  title="LP solution walks (n=2)",
                  colorbar=true, size=(900, 700))

    # Constraint boundaries — clip to plot range to avoid lines shooting off
    for i in 1:size(A, 2)
        a = A[:, i]
        if abs(a[2]) > 1e-8
            f_u(x1) = (b_u[i] - a[1]*x1) / a[2]
            f_l(x1) = (b_l[i] - a[1]*x1) / a[2]
            plot!(plt, x1_range, clamp.(f_u.(x1_range), minimum(x2_range), maximum(x2_range)),
                  color=:gray, alpha=0.5, label=false, lw=1.2)
            plot!(plt, x1_range, clamp.(f_l.(x1_range), minimum(x2_range), maximum(x2_range)),
                  color=:gray, alpha=0.5, label=false, lw=1.2)
        end
    end

    # Box bounds
    plot!(plt,
          [x_l[1], x_u[1], x_u[1], x_l[1], x_l[1]],
          [x_l[2], x_l[2], x_u[2], x_u[2], x_l[2]],
          color=:black, lw=2.0, linestyle=:dash, label="bounds")

    # Interior point walk
    plot!(plt, ip_x1, ip_x2,
          color=:red, lw=2.5, marker=:circle, markersize=5, label="Interior point")
    scatter!(plt, [ip_x1[1]], [ip_x2[1]],
             color=:red, markersize=7, marker=:diamond, label="IP start")
    scatter!(plt, [ip_x1[end]], [ip_x2[end]],
             color=:red, markersize=10, marker=:star5, label=false)

    # Simplex start → end
    plot!(plt, [x0_std[1] + x_l[1], x_sim_orig[1]],
               [x0_std[2] + x_l[2], x_sim_orig[2]],
          color=:blue, lw=2.5, marker=:square, markersize=5, label="Simplex")
    scatter!(plt, [x_sim_orig[1]], [x_sim_orig[2]],
             color=:blue, markersize=10, marker=:star5, label=false)

    # Commercial
    scatter!(plt, [x_lib[1]], [x_lib[2]],
             color=:green, markersize=12, marker=:star5, label="Ipopt")

    savefig(plt, "lp_solution_walks.png")
    println("Saved solution walk plot to lp_solution_walks.png")
    return plt
end

# --- Run 2D test ---
println("\n=== 2D solution walk test ===")
n_dim_2d = 2
n_con_2d = 4
g2, A2, b_l2, b_u2, x_l2, x_u2, _ = generate_test_problem_lp(n_dim_2d, n_con_2d)
plot_lp_solution_walks(g2, A2, b_l2, b_u2, x_l2, x_u2, n_dim_2d)
# println("Trying to solve problems with simplex")
# n_dim = 3
# n_con = 22
# println("Generate test prob with $(n_dim) x vars and $(n_con) constraints")
# g, A, b_l, b_u, x_l, x_u, x0 = generate_test_problem_lp(n_dim, n_con)
# I_matrix = Matrix{Float64}(I, n_dim, n_dim)


# #now rewrite to LP standard form:
# println("Rewriting to standard form")
# A_std, b_std, g_std = to_standard_form(g, A, b_l, b_u, x_l, x_u)
# #number of vars
# n_std = length(g_std)



# #LP problem to find feasible point
# println("Formulate sub-LP to find feasible point")
# A_fp, b_fp, g_fp, x0 = fp_standard_form(A_std, b_std)
# result_fp = revised_simplex(A_fp, b_fp, g_fp, x0)
# x0_std = result_fp.x[1:n_std]#actual feasible point  
# x0_std = max.(x0_std, 1e-4)
# if !result_fp.optimal
#     print(result_fp)
#     error("Phase 1 failed - problem may be infeasible")
# end
# x, mu, lambda = primal_dual_interior_LP(g_std, A_std, b_std, x0_std)
# println("test of primal_dual_LP")
# println("Interior point solution: $(x[1:n_dim] .+ x_l)")
# # display("x = $x")
# # display("mu = $mu")
# # display("lambda = $lambda")

# println("Feasible point found. Proceeding to solving orig. LP problem")

# result = revised_simplex(A_std, b_std, g_std, x0_std)
# if result.optimal
#     println("Optimal solution: $(result.x[1:n_dim] .+ x_l)")
# end

# lib_sol_lp = library_solver_lp(g, A, b_l, b_u, x_l, x_u)
# # print(lib_sol_lp)

# # Extract and compare solutions
# x_ip = x[1:n_dim] .+ x_l
# x_simplex = result.optimal ? result.x[1:n_dim] .+ x_l : fill(NaN, n_dim)
# x_ipopt = lib_sol_lp

# println("\n=== Solution Comparison ===")
# println("Interior Point:  $x_ip")
# println("Simplex:         $x_simplex")
# println("Ipopt:           $x_ipopt")