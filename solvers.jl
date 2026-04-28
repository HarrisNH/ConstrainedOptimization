function simplex(A, b,  g, x0)
    """ 
    This solves the general 
    min g' x 
     x
    s.t. A' x >= b. 
    """
    
    # find feasible point (initial point)

    tol = 1e-7
    err = 1 #?
    k = 1
    max_number_of_iterations = 10_000
    
    n_vars, m_const = size(A)
    x_list = [x0]

    # find initial working set
    # display(A)
    # display(x0)
    # display(b)
    # display(A' * x0 .== b)
    # println("n_vars, m_const = $n_vars, $m_const")
    # active_tolerance = tol 
    # W_set = Array(1:m_const)[A' * x0 - b .< active_tolerance] #index of working set
    # W_not_set = Array(1:m_const)[A' * x0 - b .> active_tolerance]
    W_set = Array(1:m_const)[A' * x0 .== b] #index of working set
    W_not_set = Array(1:m_const)[A' * x0 .!= b]

    converged = false 
    while !converged && k < max_number_of_iterations
        # solve the equality constraint
        A_W = A[:, W_set]

        x_k = x_list[end]
        k += 1

        n_W = size(A_W, 2)
        KKT_matrix = [
            G       -A_W; 
            -A_W'   zeros(n_W, n_W) # check A.size[2] 
        ]

        KKT_rhs = -[
            G * x_k + g; 
            zeros(n_W) 
        ] 

        res = KKT_matrix \ KKT_rhs

        p = res[1:n_vars]
        mu = res[n_vars+1:end]
        #display(mu)

        if norm(p, Inf) <= tol #||p^*|| = 0 
            if all(x -> x >= 0, mu)
                x_sol = x_k 
                mu_sol = mu # should only be those in W that are us rest should be zero
                push!(x_list, x_sol)
                converged = true
                break 
            else
                index_drop = argmin(mu)
                global_index = W_set[index_drop]
                
                W_set = deleteat!(W_set, index_drop)
                push!(x_list, x_k)

                push!(W_not_set, global_index)
                sort!(W_not_set) # just in case 
            end 
        else 
            # compute distance 
            new_set = A' * p .< 0 .&& in.(1:m_const, Ref(W_not_set))
            b_nw = b[new_set]
            A_nw = A[:, new_set]
            
            result = (b_nw - A_nw' * x_k) ./ (A_nw' * p)
            alpha = minimum(result)
            j = argmin(result) 
            
            if alpha < 1
                push!(x_list, x_k + alpha * p)
                # add constraint to working set
                new_constraint_index_global = findall(new_set)[j] #findall returns index of all 
                push!(W_set, new_constraint_index_global)
                sort!(W_set)

                filter!(x -> x != new_constraint_index_global, W_not_set) 
            else
                push!(x_list, x_k + p)
                # keep working set constant to what it currently is
            end
        end
        
        # update converged 
        rL = G * x_list[end] + g - A_W * mu
        rx = x_list[end] - x_list[end-1]
        
        converged = norm(rL, Inf) < tol && norm(rx, Inf) < tol && norm(mu, Inf) < tol 
    end
    
    return x_list[end], k#, mu_sol 
end
