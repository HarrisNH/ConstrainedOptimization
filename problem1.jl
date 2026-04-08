using LinearAlgebra, Plots, SparseArrays
using JuMP
using Ipopt
using Random, Distributions

# 3. Test problem 
n = 100
alpha = 0.5
density = 0.15
beta = 0.75
flag = "sparse"

function percent_matrix(distribution, density, n, m)
    A = zeros(n,m)
    non_zeros = Int(round(density * n*m))
    A_idx = randperm(n*m)[1:non_zeros]
    A_ij = rand(distribution, non_zeros)
    A[A_idx] = A_ij 
    return A
end

function RandomEQP(n, alpha, density, beta, flag)
    """
    H, g, A, b = RandomEQP(n, alpha, density, beta, flag)
    n: size of x-vector
    alpha: factor to ensure Hessian is SPD 
    density: % of elements in A, M that is non-zero

    """
    N = Normal(0, 1)
    U = Uniform(-1, 1)
    m = Int(round(beta * n))

    A = percent_matrix(N, density, n, m)
    M = percent_matrix(N, density, n, m)
    H = M * M' + alpha * I 

    g = rand(U, n)
    b = rand(U, m)
    
    if lowercase(flag) == "dense"
        return H, g, A, b
    elseif lowercase(flag) == "sparse"
        return sparse(H), g, sparse(A), b
    end 
end 

# 4. 
function construct_KKT(H, g, A, b)
    # Check! See lecture 5, 5-1 in module 4
    m_a = size(A)[2]
	KKT_matrix = [
		hcat(H, -A); 
		hcat(-A', zeros(m_a, m_a))
	]
	rhs = [-g; -b]

	return KKT_matrix, rhs
end 

function LDL_solver(KKT_matrix, rhs)
    # TODO: ask next time about ldlt and bunchkaufman and ZeroPivotExcption
    F = issparse(KKT_matrix) && isposdef(KKT_matrix) ? ldlt(Symmetric(KKT_matrix)) : bunchkaufman(Matrix(KKT_matrix))
	return F \ rhs 
end 

function EqualityQPSolverLDLdense(H, g, A, b)
    """
    LDL dense solver
    x, lambda = EqualityQPSolverLDLdense(H, g, A, b)
    """
    n = size(H)[1]
    KKT, rhs = construct_KKT(H, g, A, b)
    sol = LDL_solver(KKT, rhs)

    x = sol[1:n]
    lambda = sol[n+1:end]

    return x, lambda
end

function EqualityQPSolverLDLsparse(H, g, A, b)
    """
    LDL sparse solver
    x, lambda = EqualityQPSolverLDLsparse(H, g, A, b)
    """
    n = size(H)[1]
    KKT, rhs = construct_KKT(H, g, A, b)
    sol = LDL_solver(KKT, rhs)

    x = sol[1:n]
    lambda = sol[n+1:end]

    return x, lambda 
end

function EqualityQPSolver(H, g, A, b, solver)
    """
    [x,lambda]=EqualityQPSolver(H,g,A,b,solver)
    """
    if lowercase(solver) == "dense"
        x, lambda = EqualityQPSolverLDLdense(H, g, A, b)
    elseif  lowercase(solver) == "sparse"
        x, lambda = EqualityQPSolverLDLsparse(H, g, A, b)
    end
    
    return x, lambda
end

H_sparse, g, A_sparse, b = RandomEQP(5, 0.5, 0.5, 0.5, "sparse")

KKT, rhs = construct_KKT(H_sparse, g, A_sparse, b)
println("eigvals = ", eigvals(Symmetric(Matrix(KKT))))
println("KKT matrix")
println(KKT)

println("computing sparse")
x_sparse, lambda_sparse = EqualityQPSolver(H_sparse, g, A_sparse, b, "sparse")
println("computing dense")
x_dense, lambda_dense = EqualityQPSolver(Matrix(H_sparse), g, Matrix(A_sparse), b, "dense")

println("comparing sparse and dense")
display(x_sparse)
display(x_dense)
println("")
display(lambda_sparse)
display(lambda_dense)

# H = [4.0 1.0 0.0;
#      1.0 2.0 0.0;
#      0.0 0.0 2.0]
# g = [8.0, 3.0, 3.0]
# A = [1.0, 1.0, 1.0]
# b = [1.0]

# x_sparse, lambda_sparse = EqualityQPSolver(sparse(H), g, sparse(A), b, "sparse")
# print(x_sparse)
# print(lambda_sparse)

# 5. 