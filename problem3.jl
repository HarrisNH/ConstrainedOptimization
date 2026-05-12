using LinearAlgebra, Plots, SparseArrays
using JuMP
using Ipopt
using Random, Distributions
using Plots

include("test_problems.jl")
include("revised_simplex.jl")
include("helpers.jl")

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
lib_sol_lp = library_solver_lp(g, A, b_l, b_u, x_l, x_u)


#now rewrite to LP standard form:
# Ax=b
# x >= 0 
A_std, b_std, g_std, _ = to_standard_form(g, A, b_l, b_u, x_l, x_u)
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

