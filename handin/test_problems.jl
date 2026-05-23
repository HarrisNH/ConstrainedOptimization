using Measures
using Statistics

# ─── Commercial solver wrapper ────────────────────────────────────────────────
# Solves the same NLP using Ipopt directly via JuMP.
# Returns (time, iters) matching the format of our custom solvers.
# Ipopt iteration count is used for the iterations panel.

function run_commercial(A_mat, b_vec, x0; tol=1e-6)
    n      = length(x0)
    m_ineq = length(b_vec)

    model = Model(optimizer_with_attributes(
        Ipopt.Optimizer,
        "tol"         => tol,
        "print_level" => 0,
        "max_iter"    => 3000,
    ))

    @variable(model, x[1:n])
    for i in 1:n
        set_start_value(x[i], x0[i])
    end

    # Chained Rosenbrock objective — must be re-expressed symbolically for JuMP
    @objective(model, Min,
        sum((1 - x[i])^2 + 100*(x[i+1] - x[i]^2)^2 for i in 1:(n-1))
    )

    # Linear inequality constraints: A'x >= b
    for j in 1:m_ineq
        @constraint(model, sum(A_mat[i,j] * x[i] for i in 1:n) >= b_vec[j])
    end

    optimize!(model)

    iters = MOI.get(model, MOI.BarrierIterations())
    return termination_status(model), solve_time(model), iters
end


# ─── Test problem generator ───────────────────────────────────────────────────
# Chained Rosenbrock with random linear inequalities.
# Returns func, hess, ineq (for custom solvers) AND A_mat, b_vec (for JuMP).

function generate_sqp_test_problem(n, m_ineq; seed=nothing)
    seed !== nothing && Random.seed!(seed)

    function func(x)
        val  = 0.0
        grad = zeros(n)
        for i in 1:(n-1)
            val += (1 - x[i])^2 + 100*(x[i+1] - x[i]^2)^2
            grad[i]   += -2*(1 - x[i]) - 400*x[i]*(x[i+1] - x[i]^2)
            grad[i+1] +=  200*(x[i+1] - x[i]^2)
        end
        return val, grad
    end

    function hess(x)
        Hf = zeros(n, n)
        for i in 1:(n-1)
            Hf[i,   i]   += 1200*x[i]^2 - 400*x[i+1] + 2
            Hf[i,   i+1] += -400*x[i]
            Hf[i+1, i]   += -400*x[i]
            Hf[i+1, i+1] += 200.0
        end
        Hg = [zeros(n, n) for _ in 1:m_ineq]
        return Hf, Hg
    end

    x0_ref = 0.1 * ones(n)
    A_mat  = randn(n, m_ineq)
    slack  = rand(m_ineq) .* 0.9 .+ 0.1
    b_vec  = A_mat' * x0_ref .- slack

    function ineq(x)
        return A_mat' * x .- b_vec, A_mat
    end

    return func, hess, ineq, x0_ref, A_mat, b_vec
end


# ─── Single timed run (custom solvers) ───────────────────────────────────────
function run_one(solver_fn, args...; kwargs...)
    try
        t      = @timed solver_fn(args...; kwargs...)
        result = t.value
        niter  = result[4]
        conv   = result[5]
        return (time=t.time, iters=Float64(niter), ok=true)
    catch e
        @warn "Solver failed: $e"
        return (time=NaN, iters=NaN, ok=false)
    end
end


# ─── Sweep helpers ────────────────────────────────────────────────────────────
function _accum(vec, val)
    push!(vec, isempty(val) ? NaN : mean(val))
end

function _run_trial(n, m_ineq, trial, tol, MAX_ITER)
    func, hess, ineq, x0, A_mat, b_vec =
        generate_sqp_test_problem(n, m_ineq, seed=trial*1000+n*7+m_ineq)
    z0 = zeros(m_ineq)
    B0 = Matrix{Float64}(I, n, n)

    t_ls_b = run_one(SQP_line_search, func, ineq, x0, z0;
                     B0=B0, tol=tol, MAX_ITER=MAX_ITER)
    t_ls_a = run_one(SQP_line_search, func, ineq, x0, z0;
                     H0=hess, tol=tol, MAX_ITER=MAX_ITER)
    t_tr_b = run_one(SQP_trust_region, func, ineq, x0, z0, 1.0;
                     B0=B0, tol=tol, MAX_ITER=MAX_ITER)
    t_tr_a = run_one(SQP_trust_region, func, ineq, x0, z0, 1.0;
                     H0=hess, tol=tol, MAX_ITER=MAX_ITER)

    # Commercial (Ipopt via JuMP)
    com_ok = true
    com_time = NaN; com_iters = NaN
    try
        status, com_time, com_iters_int = run_commercial(A_mat, b_vec, x0; tol=tol)
        com_iters = Float64(com_iters_int)
    catch e
        @warn "Commercial solver failed: $e"
        com_ok = false
    end

    return t_ls_b, t_ls_a, t_tr_b, t_tr_a,
           (time=com_time, iters=com_iters, ok=com_ok)
end


# ─── Sweep vs n ───────────────────────────────────────────────────────────────
function benchmark_sqp_vs_n(n_range; m_ineq=3, n_trials=3, tol=1e-6, MAX_ITER=200)
    results = Dict(k => Float64[] for k in
        [:t_ls_b, :t_ls_a, :t_tr_b, :t_tr_a, :t_com,
         :k_ls_b, :k_ls_a, :k_tr_b, :k_tr_a, :k_com])

    for n in n_range
        println("  n=$n, m=$m_ineq")
        bufs = Dict(k => Float64[] for k in keys(results))

        for trial in 1:n_trials
            r_ls_b, r_ls_a, r_tr_b, r_tr_a, r_com =
                _run_trial(n, m_ineq, trial, tol, MAX_ITER)

            r_ls_b.ok && (push!(bufs[:t_ls_b], r_ls_b.time); push!(bufs[:k_ls_b], r_ls_b.iters))
            r_ls_a.ok && (push!(bufs[:t_ls_a], r_ls_a.time); push!(bufs[:k_ls_a], r_ls_a.iters))
            r_tr_b.ok && (push!(bufs[:t_tr_b], r_tr_b.time); push!(bufs[:k_tr_b], r_tr_b.iters))
            r_tr_a.ok && (push!(bufs[:t_tr_a], r_tr_a.time); push!(bufs[:k_tr_a], r_tr_a.iters))
            r_com.ok  && (push!(bufs[:t_com],  r_com.time);  push!(bufs[:k_com],  r_com.iters))
        end

        for k in keys(results)
            _accum(results[k], bufs[k])
        end
    end
    return results
end


# ─── Sweep vs m ───────────────────────────────────────────────────────────────
function benchmark_sqp_vs_m(m_range; n_fixed=5, n_trials=3, tol=1e-6, MAX_ITER=200)
    results = Dict(k => Float64[] for k in
        [:t_ls_b, :t_ls_a, :t_tr_b, :t_tr_a, :t_com,
         :k_ls_b, :k_ls_a, :k_tr_b, :k_tr_a, :k_com])

    for m in m_range
        println("  n=$n_fixed, m=$m")
        bufs = Dict(k => Float64[] for k in keys(results))

        for trial in 1:n_trials
            r_ls_b, r_ls_a, r_tr_b, r_tr_a, r_com =
                _run_trial(n_fixed, m, trial, tol, MAX_ITER)

            r_ls_b.ok && (push!(bufs[:t_ls_b], r_ls_b.time); push!(bufs[:k_ls_b], r_ls_b.iters))
            r_ls_a.ok && (push!(bufs[:t_ls_a], r_ls_a.time); push!(bufs[:k_ls_a], r_ls_a.iters))
            r_tr_b.ok && (push!(bufs[:t_tr_b], r_tr_b.time); push!(bufs[:k_tr_b], r_tr_b.iters))
            r_tr_a.ok && (push!(bufs[:t_tr_a], r_tr_a.time); push!(bufs[:k_tr_a], r_tr_a.iters))
            r_com.ok  && (push!(bufs[:t_com],  r_com.time);  push!(bufs[:k_com],  r_com.iters))
        end

        for k in keys(results)
            _accum(results[k], bufs[k])
        end
    end
    return results
end


# ─── Plots ────────────────────────────────────────────────────────────────────
function create_sqp_benchmark_plots(res_n, n_range, m_fixed,
                                     res_m, m_range, n_fixed;
                                     filename="sqp_benchmark.png")

    labels  = ["LS + BFGS", "LS + Analytical", "TR + BFGS", "TR + Analytical", "Ipopt (commercial)"]
    markers = [:circle, :square, :diamond, :utriangle, :star5]

    function make_panel(xdata, time_vecs, xlabel, title, ylabel_str)
        p = plot(xlabel=xlabel, ylabel=ylabel_str, title=title,
                 left_margin=10mm, bottom_margin=6mm, legend=:topleft)
        for (yd, lab, mk) in zip(time_vecs, labels, markers)
            plot!(p, xdata, yd, label=lab, lw=2, marker=mk, markersize=5)
        end
        return p
    end

    # vs n — time
    p1 = make_panel(n_range,
        [res_n[:t_ls_b], res_n[:t_ls_a], res_n[:t_tr_b], res_n[:t_tr_a], res_n[:t_com]],
        "Variables (n)", "CPU time vs n\n(m = $m_fixed)", "CPU time [s]")

    # vs n — iterations
    p2 = make_panel(n_range,
        [res_n[:k_ls_b], res_n[:k_ls_a], res_n[:k_tr_b], res_n[:k_tr_a], res_n[:k_com]],
        "Variables (n)", "Iterations vs n\n(m = $m_fixed)", "Iterations")

    # vs m — time
    p3 = make_panel(m_range,
        [res_m[:t_ls_b], res_m[:t_ls_a], res_m[:t_tr_b], res_m[:t_tr_a], res_m[:t_com]],
        "Constraints (m)", "CPU time vs m\n(n = $n_fixed)", "CPU time [s]")

    # vs m — iterations
    p4 = make_panel(m_range,
        [res_m[:k_ls_b], res_m[:k_ls_a], res_m[:k_tr_b], res_m[:k_tr_a], res_m[:k_com]],
        "Constraints (m)", "Iterations vs m\n(n = $n_fixed)", "Iterations")

    fig = plot(p1, p2, p3, p4, layout=(2,2), size=(1300, 950))
    savefig(fig, filename)
    println("Saved to $filename")
    return fig
end




function generate_test_problem_qp(n, m_a)
    """
    Generates H, g,A, b_l, b_u, x_l, x_u of size n. 

    H, g, A, b_l, b_u, x_l, x_u = generate_test_problem(n, m_a)
    """
    M = rand(Uniform(-1, 1), n, n)
    alpha = rand(Uniform(0.5, 2))
    H = M * M' + alpha * I  # this ensures that H > 0 

    g = rand(Uniform(-5, 5), n)
    
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

    y = A' * x
    diff_Ax = rand(Uniform(0, 5), m_a)
    b_l = y .- diff_Ax
    b_u = y .+ diff_Ax
    return H, g, A, b_l, b_u, x_l, x_u

end
 
 
function generate_test_problem_lp(n, m_a)
    """
    Generates g, A, b_l, b_u, x_l, x_u for an LP of size n.

    g, A, b_l, b_u, x_l, x_u, x = generate_test_problem_lp(n, m_a)
    """
    g = rand(Uniform(-5, 5), n)

    n_active = 2
    active_idx = randperm(n)[1:n_active]

    x = rand(Uniform(-3, 3), n)
    diff_x = rand(Uniform(0.5, 3), n)
    x_l = x .- diff_x
    x_u = x .+ diff_x
    x[active_idx] .= x_u[active_idx]

    A = rand(Uniform(-2, 2), n, m_a)
    println("rank(A) = $(rank(A))")
    y = A' * x
    diff_Ax = rand(Uniform(0, 5), m_a)
    b_l = y .- diff_Ax
    b_u = y .+ diff_Ax

    return g, A, b_l, b_u, x_l, x_u, x
end