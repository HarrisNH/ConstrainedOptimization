using LinearAlgebra, Plots, SparseArrays
using JuMP
using Ipopt
using Random, Distributions
using Plots

function EqualityQP(H, g, A, b)
    """
    Solves the problem 
    min 1/2 x' H x + g' x 
    s.t. A'x = b 

    return x, lambda 
    """
    n, m = size(A)
    KKT_matrix = [
        H   -A; 
        -A' zeros(n, n) 
    ]
    KKT_rhs = -[
        g; b
    ]

    sol = KKT_matrix \ KKT_rhs 
    x = sol[1:n]
    lambda = sol[n+1:end]

    return x, lambda 
end

function QP_active_set(G, g, A, b, x0)
    """
    Solves the problem 
    min 1/2 x' G x + g' x 
    s.t. A'x - b >= 0 

    x0 is an initial feasible point 
    """

    W_set = Array
end