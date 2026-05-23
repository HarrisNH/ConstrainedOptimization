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

function LDL_solver(KKT_matrix, rhs)
    """
    Solver for the KKT system.
    """
    if issparse(KKT_matrix)
        # this defaults to sparse LU and thus handles sparse indefinite systems
        return KKT_matrix \ rhs
    else
        # Bunch-Kaufman is for dense symmetric indefinite systems
        return bunchkaufman(Symmetric(KKT_matrix)) \ rhs
    end
end