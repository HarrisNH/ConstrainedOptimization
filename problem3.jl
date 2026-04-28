using LinearAlgebra, Plots, SparseArrays
using JuMP
using Ipopt
using Random, Distributions
using Plots

include("test_problems.jl")
include("revised_simplex.jl")


function to_standard_form(g, A, b_l, b_u, x_l, x_u)
    """
    Transform the constraints for lp problem from
    b_l <= A'x <= b_u
    x_l <= x <= x_u

    to

    A'x = b
    x >= 0
    to prepare for simplex algorithm
    """
    n = length(g) # number of orig. vars
    m_a = size(A, 2) # 1/2 of Ax constraints (bcs lowbound and upperbound)
    At  = A'
    
    # We need x>=0 so we must normalize x's by lowerbound x^~ = x-x_l

    b_l = b_l - A' * x_l
    b_u = b_u - A' * x_l
    x_u = x_u - x_l

    n_rows = n + 2 * m_a # number of constraints
    n_cols = n + n_rows #one s variable for each row/constraint + orig vars
    
    A_std = zeros(n_rows, n_cols) # constraints x vars
    A_std[1:m_a, 1: n] = At # orig A, LHS: Ax = b_l (transposed bcs standard form has Ax=b)
    A_std[m_a+1:2*m_a, 1:n] = At # orig A, LHS: Ax = b_u

    A_std[1:m_a, n+1: n + m_a] = -I(m_a) # slack vars for Ax = b_l
    A_std[m_a+1:2*m_a, n+m_a+1: n + 2* m_a] = I(m_a) # slack vars for Ax = b_u
    
    A_std[2 * m_a + 1 : end, 1 : n] = I(n) # x part of upper bound on vars: x + s = x_u (so identity)
    A_std[2 * m_a + 1 : end, n + 2 * m_a + 1 : end] = I(n)  # s part of upper bound on vars: x + s = x_u (so identity)
    
    b_std = [b_l; b_u; x_u]
    g_std = [g; zeros(n_rows)]

    return A_std, b_std, g_std

end 

function fp_standard_form(A, b)
    m, n = size(A) #constraints x variables
    e = ones(m)
    
    A_fp = [A e -I(m) zeros((m,m)); -A e zeros((m,m)) -I(m)]
    b_fp = [b; -b]
    g_fp = zeros(n + 2 * m + 1)
    g_fp[n + 1] = 1

    t = maximum(abs.(b))
    x0 = [zeros(n); t; t * e - b; t * e + b]
    return A_fp, b_fp, g_fp, x0
end

# d 
function library_solver_lp(g, A, b_l, b_u, x_l, x_u)
    """
    This solves the problem above using Ipopt. 
    #TODO transform to LP solver
    """
    n = size(A)[1]
    model = Model(Ipopt.Optimizer)
    @variable(model, x[1:n])
    @constraint(model, b_l .<= A' * x .<= b_u)
    @constraint(model, x_l .<= x .<= x_u)
    @objective(model, Min, g' * x)
    optimize!(model)

    return value.(x)
end

# e and f 
function primal_dual_interior_LP(g, A, b, x0) 
    """
    This solves 
    min_x g' x 
    s.t. A x = b, x >= 0 
    """

    m, n = size(A)
    maxiter = 1_000 
    tol = 1.0e-9

    lambda = ones(n, 1)
    mu = zeros(m, 1)

    # display(g)
    # display(A)
    # display(A')
    # display(mu)
    # display(lambda)
    # display(x0)
    # display(b)

    x = x0
    # display(x)
    # display(lambda)

    rL = g - A' * mu - lambda 
    rA = A * x0 - b 
    rC = x0 .* lambda
    s = sum(rC) / n

    converged = (norm(rL, Inf) < tol) & (norm(rA, Inf) < tol) & (norm(s, Inf) < tol)
    k = 0 
    
    while !converged & k < maxiter
        k = k + 1
        
        xDivlambda = vec(x ./ lambda) # use vec() to convert to vector, otherwise diagm crashes! 
        H = A * diagm(xDivlambda) * A' 
        L = cholesky(Symmetric(H))

        tmp = (x .* rL + rC) ./ lambda
        rhs = -rA + A * tmp

        dmu = L' \ (L \ rhs)
        dx = xDivlambda .* (A' * dmu) - tmp 
        dlambda = -(rC + lambda .* dx) ./ x

        idx = findall(x -> x < 0.0, dx) 
        alpha = minimum(hcat(1.0, .-x[idx] ./ dx[idx]))

        idx = findall(x -> x < 0.0, dlambda)
        beta = minimum(hcat(1.0, .-lambda[idx] ./ dlambda[idx]))

        x_aff = x + alpha * dx 
        lambda_aff = lambda + beta * dlambda
        s_aff = sum(x_aff .* lambda_aff) / n 

        sigma = (s_aff / s)^3 
        tau = sigma * s 

        rC = rC + dx .* dlambda .- tau 

        tmp = (x .* rL + rC) ./ lambda
        rhs = -rA + A * tmp 

        dmu = L' \ (L \ rhs) 
        dx = xDivlambda .* (A' * dmu) - tmp 
        dlambda = -(rC + lambda .* dx) ./ x
        
        idx = findall(x -> x < 0.0, dx)
        println(idx)
        println(x)
        println(dx)
        alpha = minimum(hcat(1.0, .-x[idx] ./ dx[idx]))

        idx = findall(x -> x < 0.0, dlambda)
        beta = minimum(hcat(1.0, .-lambda[idx] ./ dlambda[idx]))

        x = x + (eta * alpha) * dx 
        mu = mu + (eta * beta) * dmu 
        lambda = lambda + (eta * beta) * dlambda

        rL = g - A' * mu - lambda
        rA = A * x - b 
        rC = x .* lambda
        s = sum(rC) / n 
        

        converged = norm(rL, Inf) < tol & norm(rA, Inf) < tol & norm(s, Inf) < tol 
    end

    return x, mu, lambda
end

# simple test problem 
g = [1; 1]
A = [1 2; 2 3]
b = [2; 3]
x0 = [0; 2]

x, mu, lambda = primal_dual_interior_LP(g, A, b, x0)
println("test of primal_dual_LP")
println("x = $x")
println("mu = $mu")
println("lambda = $lambda")

g = [-2.0; -5.0]
A = [1.0 1.0; 1.0 2.0]'  # 2 constraints
b_l = [5.0; 5.0]
b_u = [10.0; 10.0]
x_l = [0.0; 0.0]
x_u = [100.0; 100.0]




#n_dim = 3
#n_con = 22
#g, A, b_l, b_u, x_l, x_u, x0 = generate_test_problem_lp(n_dim, n_con)
I_matrix = Matrix{Float64}(I, n_dim, n_dim)
lib_sol_lp = library_solver_lp(g, A, b_l, b_u, x_l, x_u)


#now rewrite to LP standard form:
# Ax=b
# x >= 0 
A_std, b_std, g_std = to_standard_form(g, A, b_l, b_u, x_l, x_u)
n_std = length(g_std)

A_fp, b_fp, g_fp, x0 = fp_standard_form(A_std, b_std)
result_fp = revised_simplex_chat(A_fp, b_fp, g_fp, x0)
if !result_fp.optimal
    print(result_fp)
    error("Phase 1 failed - problem may be infeasible")
end
x0_std = result_fp.x[1:n_std]     
result = revised_simplex_chat(A_std, b_std, g_std, x0_std)

x_sol, k = primal_dual_LP( g_std, A_std, b_std, x0)
print("From the implemented solver we get: \n")
print("x_sol : ")
display(x_sol)
print("Iterations: $k")


## using library solver

library_solver_lp(g, A, b_l, b_u, x_l, x_u)

