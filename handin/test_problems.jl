function generate_test_problem_qp(n, m_a)
    """
    Generates H, g,A, b_l, b_u, x_l, x_u of size n. 

    H, g, A, b_l, b_u, x_l, x_u = generate_test_problem(n, m_a)
    """

    # TODO: maybe find better numbers 
    # and discuss the generation method - the idea should be good. 
    M = rand(Uniform(-1, 1), n, n)
    alpha = rand(Uniform(0.5, 2))
    H = M * M' + alpha * I  # this ensures that H > 0 

    g = rand(Uniform(-5, 5), n)
    
    # x = rand(n)
    # diff_x = rand(Uniform(0, 5), n)
    # x_l = x .- diff_x
    # x_u = x .+ diff_x

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
    println("rank(A) = $(rank(A)), constraints = $(m_a)")
    y = A' * x
    diff_Ax = rand(Uniform(0, 5), m_a)
    # print(size(diff_Ax))
    b_l = y .- diff_Ax
    b_u = y .+ diff_Ax
    
 
    return H, g, A, b_l, b_u, x_l, x_u, x
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