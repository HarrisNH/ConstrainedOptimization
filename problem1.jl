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
alpha: factor to ensure  Hessian is SPD 
density: % of elements in A,M that is non-zero

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
    # Check!  
    n = size(A)[1]
	KKT_matrix = [
		hcat(H, -A); 
		hcat(-A', zeros(n, n))
	]
	rhs = [-g; -b]

	return KKT_matrix, rhs
end 

function LDL_solver(KKT_matrix, rhs)
    # Check! 
	F = ldlt(Symmetric(KKT_matrix))
	return F \ rhs 
end 

function EqualityQPSolverLDLdense(H,g,A,b)
    """
    LDL dense solver
    x, lambda = EqualityQPSolverLDLdense(H, g, A, b)
    """
end

function EqualityQPSolverLDLsparse(H,g,A,b)
    """
    LDL sparse solver
    x, lambda = EqualityQPSolverLDLsparse(H, g, A, b)
    """
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