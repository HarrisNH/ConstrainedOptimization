using LinearAlgebra, Plots, SparseArrays
using JuMP
using Ipopt
using Random, Distributions


include("problem2.jl")
include("helpers.jl")
# f
println("(f)")
f(x1, x2) = (x1^2 + x2 - 11)^2 + (x1 + x2^2 - 7)^2
c1(x1, x2) = (x1 + 2)^2 - x2  # >= 0
c2(x1, x2) = -4*x1 + 10*x2    # >= 0

model = Model(Ipopt.Optimizer)
@variable(model, x1)
@variable(model, x2)
@objective(model, Min, f(x1, x2))
@constraint(model, c1(x1, x2) >= 0)
@constraint(model, c2(x1, x2) >= 0)

optimize!(model)
println("solution x1, x2 = $(value(x1)), $(value(x2))")

# g
# helper functions for SQP
function HL(H_func, x, z)

    Hf, Hg = H_func(x)
    L = copy(Hf)

    for i in 1:length(z)
        L .-= z[i] .* Hg[i]
    end
    return L
end

function dL_func(x, z, df, dg)

    dgk = dg(x)
    L = df(x)

    if !isempty(z)
        L .-= dgk * z
    end
    return L
end

function solve_QP_ip(Hk, dfk, dgk, gk; xl=nothing, xu=nothing)
    n = length(dfk)
    m_ineq = length(gk)

    C = copy(dgk)       
    d = -copy(gk)  


    if xl !== nothing
        finite = .!isinf.(xl)
        if any(finite)
            C = [C  Matrix(I,n,n)[:, finite]]
            d = [d; xl[finite]]
        end
    end
    if xu !== nothing
        finite = .!isinf.(xu)
        if any(finite)
            C = [C  -Matrix(I,n,n)[:, finite]]
            d = [d; -xu[finite]]
        end
    end

    result = primal_dual_qp_ineq(Hk, dfk, C, d)

    zhat = result.z[1:m_ineq]

    return result.x, zhat   # p_opt, multipliers
end

function SQP_line_search(func, ineq, x0, z0; 
        xl=nothing, xu=nothing, H0=nothing, B0=nothing, 
        rho=0.5, eta=0.1, tau=0.9, l0=100.0, MAX_ITER=1000, tol=1e-6)
    """
    Computes a solution to the non-linear constrained optimisation problem:
        min_x f(x)  
        s.t. g(x) >= 0, xl <= x <= xu

    func is a vector consiting of f and df (the gradient of f), that is, 
        func = [f, df]
    and similarly
        ineq = [g, dg]
    """
    if H0 !== nothing
        H = x -> begin
            Hf_val, Hg_val = H0(x)
            return (Hf_val, Hg_val)
        end
    else
        H = nothing
    end

    f_eval(x) = func(x)[1]
    df_eval(x) = func(x)[2]
    g_eval(x) = ineq(x)[1]
    dg_eval(x) = ineq(x)[2]

    xk, zk = copy(x0), copy(z0) 
    Bk = B0 !== nothing ? copy(B0) : nothing
    niter = 0
    xs = [copy(xk)]
    conv = []

    while niter < MAX_ITER
        niter += 1
        if Bk === nothing
            Hk = HL(H, xk, zk)
            # regularization to ensure positive definitenes
            min_eig = minimum(eigvals(Hk))
            if min_eig < 1e-8
                Hk += (abs(min_eig) + 1e-6) * I
            end
        else
            Hk = Bk
        end

        fk, dfk = func(xk)
        gk, dgk = ineq(xk)

        xl_pk = xl === nothing ? nothing : xl .- xk
        xu_pk = xu === nothing ? nothing : xu .- xk

        pk, zhat = solve_QP_ip(Hk, dfk, dgk, gk, xl=xl_pk, xu=xu_pk)

        gkn = norm(min.(g_eval(xk), 0), 1)
        qp_val = abs(dfk' * pk + 0.5 * pk' * Hk * pk)
        lk = gkn > 1e-8 ? qp_val / ((1.0 - rho) * gkn) + l0  : l0
        
        D_phi = dfk' * pk - lk * gkn
        phi_0 = fk + lk * gkn
        
        if D_phi >= 0
            alpha = 1e-4
        else
            alpha = 1.0
            phi_alpha = f_eval(xk + alpha * pk) + lk * norm(min.(g_eval(xk + alpha * pk), 0), 1)
            while phi_alpha > phi_0 + alpha * eta * D_phi && alpha > 1e-8
                alpha *= tau
                phi_alpha = f_eval(xk + alpha * pk) + lk * norm(min.(g_eval(xk + alpha * pk), 0), 1)
            end
        end
        
        xk1 = xk + alpha * pk
        zk1 = zhat

        push!(xs, copy(xk1))

        res = [dL_func(xk1, zk1, df_eval, dg_eval); max.(-g_eval(xk1), 0)] 
        curr_conv = norm(res, Inf)
        push!(conv, curr_conv)

        if curr_conv < tol || norm(xk1 - xk) < tol
            println("Converged to xk = $xk1 in $niter iterations")
            xk, zk = xk1, zk1
            break
        end

        if H === nothing && niter < MAX_ITER
            sk = xk1 - xk
            qk = dL_func(xk1, zk1, df_eval, dg_eval) - dL_func(xk, zk, df_eval, dg_eval) 

            sBs = sk' * Bk * sk
            sqk = sk' * qk

            if !isapprox(sBs - sqk, 0, atol=1e-14)
                theta = 1.0
                if sqk < 0.2 * sBs
                    theta = (0.8 * sBs) / (sBs - sqk)
                end
                rk = theta .* qk + (1.0 - theta) .* (Bk * sk)
                Bk .+= (rk * rk') ./ (sk' * rk) .- (Bk * sk * (Bk * sk)') ./ sBs
            end
        end

        xk, zk = xk1, zk1
    end

    return xk, zk, xs, niter, conv
end

# Test problem for SQP_line_search
println("\n--- Testing SQP_line_search with Himmelblau function ---")

# x1_opt = 3.0, x2_opt = 2.0 is a minimum of Himmelblau
# The constraints are:
# c1: (x1 + 2)^2 - x2 >= 0
# c2: -4*x1 + 10*x2 >= 0

function test_func(x)
    x1, x2 = x[1], x[2]
    # Himmelblau's function
    val = (x1^2 + x2 - 11)^2 + (x1 + x2^2 - 7)^2
    grad = [
        4*x1*(x1^2 + x2 - 11) + 2*(x1 + x2^2 - 7),
        2*(x1^2 + x2 - 11) + 4*x2*(x1 + x2^2 - 7)
    ]
    return val, grad
end

function test_ineq(x)
    x1, x2 = x[1], x[2]
    val = [
        (x1 + 2)^2 - x2,
        -4*x1 + 10*x2
    ] 
    # each COLUMN is the gradient of one constraint
    jac = [
        2*(x1 + 2)  -4.0;
        -1.0         10.0
    ]
    return val, jac
end

function test_eq(x)
    return [0.0], reshape([0.0; 0.0], 2, 1) 
end

x0 = [0.0, 0.0]
z0 = [0.0, 0.0] # Initial multipliers for inequality constraints
B0 = Matrix{Float64}(I, 2, 2)

x_opt, z_opt, xs, niter, conv = SQP_line_search(
    test_func, test_ineq, x0, z0;
    B0=B0, tol=1e-9, MAX_ITER=50
)

# --- Rosenbrock Examples ---
println("\n" * "="^50)
println("EXAMPLE 2: Rosenbrock with Circular Inequality (x^2 + y^2 <= 1)")
println("="^50)

function rosenbrock_func(x)
    val = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
    grad = [
        -2 * (1 - x[1]) - 400 * x[1] * (x[2] - x[1]^2),
        200 * (x[2] - x[1]^2)
    ]
    return val, grad
end

function rosenbrock_hess(x)
    x1, x2 = x[1], x[2]
    Hf = [
        2 - 400*(x2 - x1^2) + 800*x1^2   -400*x1;
        -400*x1                          200.0
    ]
    Hg = [[-2.0 0.0; 0.0 -2.0]] # for 1 - x1^2 - x2^2
    return Hf, Hg
end

function circular_ineq(x)
    val = [1.0 - x[1]^2 - x[2]^2]
    jac = [-2*x[1]; -2*x[2]] # gradient as column vector
    return val, reshape(jac, 2, 1)
end

x0_ros = [0.0, 0.0]
z0_ros = [0.0]

println("\n> Running BFGS")
x_bfgs, _, _, n_bfgs, conv_bfgs = SQP_line_search(
    rosenbrock_func, circular_ineq, x0_ros, z0_ros; 
    B0=Matrix{Float64}(I, 2, 2), tol=1e-7, MAX_ITER=100
)
println("BFGS result: $x_bfgs (Iter: $n_bfgs, Norm: $(conv_bfgs[end]))")

println("\n> Running Analytical...")
x_ana, _, _, n_ana, conv_ana = SQP_line_search(
    rosenbrock_func, circular_ineq, x0_ros, z0_ros; 
    H0=rosenbrock_hess, tol=1e-7, MAX_ITER=100
)
println("Analytical result: $x_ana (Iter: $n_ana, Norm: $(conv_ana[end]))")
