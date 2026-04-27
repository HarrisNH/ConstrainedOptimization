function feasible_point(A, b, l, u)

end

function presolve(A, b, l, u)

    if any(l .> u)
        return nothing, nothing, nothing, nothing, true
    end

    while true
        m_before = size(A, 2)
        l_before = copy(l)
        u_before = copy(u)

        A, b, l, u, infeasible_flag = fixed_vars_check(A, b, l, u)
        infeasible_flag && return nothing, nothing, nothing, nothing, true

        A, b, infeasible_flag = redundancy_check(A, b, l, u)
        infeasible_flag && return nothing, nothing, nothing, nothing, true

        A, b, l, u, infeasible_flag = r_singleton_check(A, b, l, u)
        infeasible_flag && return nothing, nothing, nothing, nothing, true

        # stop if nothing changed
        no_constraints_dropped = size(A, 2) == m_before
        no_bounds_tightened    = l == l_before && u == u_before
        if no_constraints_dropped && no_bounds_tightened
            break
        end
    end

    return A, b, l, u, false
end

function fixed_vars_check(A, b, l, u)
    fixed_vars_arr = l .== u

    if !any(fixed_vars_arr)
        return A, b, l, u, false
    end

    x_fixed = l[fixed_vars_arr] #fixed vars
    A_fixed = A[:,fixed_vars_arr] #only get coeffs for fixed vars
    
    #A_free * x_free >= b - A_fixed*x_fixed
    residual = A_fixed * x_fixed # find contribution of fixed vars to rhs
    b_new = b - residual

    free_vars_arr = .!fixed_vars_arr
    A = A[:, free_vars_arr]
    l = l[free_vars_arr]
    u = u[free_vars_arr]

    return A, b_new, l, u, false
end


function redundancy_check(A, b, l ,u)
    m = size(A, 2)  # number of constraints (A is n x m, A'x >= b)
    
    rows_to_keep = trues(m)
    
    for i in 1:m
        a = A[:, i]
        
        min_act = sum(a[p] > 0 ? a[p] * l[p] : a[p] * u[p] for p in eachindex(a))
        max_act = sum(a[p] > 0 ? a[p] * u[p] : a[p] * l[p] for p in eachindex(a))
        
        if min_act >= b[i]
            rows_to_keep[i] = false  # redundant, drop
        elseif max_act < b[i]
            return nothing, nothing, true  # infeasible
        end
    end
    
    return A[:, rows_to_keep], b[rows_to_keep], false
end 


function r_singleton_check(A, b, l ,u)
    # row singleton in inequality a_kj * x_j >= b_k
    m = size(A, 2)  # number of constraints (A is n x m, A'x >= b)
    rows_to_keep = trues(m)   

    for i in 1:m
        a = A[:, i]

        is_singleton = sum(a .!= 0) == 1
        is_zero_row = all(a .== 0) 

        if is_zero_row
            if b[i] > 0
                return nothing, nothing, nothing, nothing, true
            else 
                rows_to_keep[i] = false
            end
        
        elseif is_singleton
            j = findfirst(a .!= 0)
            a_kj = a[j]

            if a_kj > 0
                l[j] = max(l[j], b[i] / a_kj)   # tighten lower bound Ax > b <=> x > b/a
            elseif a_kj < 0
                u[j] = min(u[j], b[i] / a_kj)   # tighten upper bound Ax >b <==> x < b/a  
            end
            if l[j] > u[j]
                return nothing, nothing, nothing, nothing, true
            end
            rows_to_keep[i] = false
        end
        # drop row k, check if l[j] > u[j] → infeasible
    end
    return A[:, rows_to_keep], b[rows_to_keep], l, u, false
end
 