using LinearAlgebra
function revised_simplex(A, b, c, x0)
    m, n = size(A) # constraints, vars

    # Derive basis from x0
    
    B = findall(x0 .> 1e-10) #basic = loose
    N = findall(x0 .<= 1e-10) #non-basic = tight = 0

    # Pad basis to size m if needed. So just pick indexes inactive and put in active.
    if length(B) < m
        extra = N[1:m-length(B)]
        append!(B, extra)
        N = setdiff(1:n, B)
    end

    max_iterations = 1000

    for iteration in 1:max_iterations
        A_B = A[:, B]
        A_N = A[:, N]
        c_B = c[B]
        c_N = c[N]

        mu       = A_B' \ c_B
        lambda_N = c_N - A_N' * mu #1 term impact on objective of c_j going from tight to loose, 2 term impact of tighten loose variable 

        if all(lambda_N .>= -1e-10)
            x = zeros(n)
            x[B] = A_B \ b
            return (optimal=true, x=x, objective=dot(c,x), basis=B, iterations=iteration)
        end

        s = argmin(lambda_N) #pivot rule
        i_s = N[s] #var to loosen
        a_is = A[:, i_s] #A col with loosened var
        h = A_B \ a_is # how other vars must change to keep Ax=b as loosened var increases

        x_B = A_B \ b #current values of original basic vars 

        ratios = [h[i] > 1e-10 ? x_B[i]/h[i] : Inf for i in 1:m] # ratio for each basic var
        all(isinf, ratios) && return (optimal=false, unbounded=true, iterations=iteration) # if all unbounded, can keep increase new basic var forever, so unbounded problem
        j    = argmin(ratios) # basic index of first var to become 0 as new basic var is loosened
        i_j  = B[j] #what is global index of this soon to be non-basic var

        B[j] = i_s #override old basic var index with new
        N[s] = i_j #override old non-basic var index with new
    end
    #returns best non-optimal solution found
    x = zeros(n)
    x[B] = A[:, B] \ b #basic vars are non-zero
    return (optimal=false, x=x, objective=dot(c,x), basis=B, iterations=max_iterations)
end

function example_problem()
    # min C'x
    #Ax=b
    #[x;s] >= 0

    A = [1.0  0.0  1.0  0.0 0.0;
        0.0  1.0  0.0  1.0 0.0;
        1.0 1.0 0.0 0.0 1.0]
    
    b = [3000.0; 4000.0 ; 5000.0]
    
    c = [-1.2 ; -1.7; 0 ; 0; 0]
    # we start origin so slacks = b
    x0 = [0; 0; b[1]; b[2]; b[3]]
    result = revised_simplex(A, b, c, x0)
    
    println("\n=== Results ===")
    if haskey(result, :unbounded)
        println("Problem is unbounded")
    else
        println("Solution: ", result.x)
        println("Objective value: ", result.objective)
        println("Basis indices: ", result.basis)
        println("Iterations: ", result.iterations)
    end
    
    return result
end

#example_problem()