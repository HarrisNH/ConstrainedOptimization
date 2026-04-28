using LinearAlgebra

function revised_simplex(A::Matrix{Float64}, b::Vector{Float64}, c::Vector{Float64}, x0)
    """
    min c'x 
    Ax = b
    x >= 0 

    with x0 as feasible starting point
    """
    m, n = size(A)

    B = collect(n-m+1:n) 
    N = collect(1:n-m)   
    
    max_iterations = 1000
    iteration = 0
    
    while iteration < max_iterations
        iteration += 1
        
        A_B = A[:, B]
        A_N = A[:, N]
        
        c_B = c[B]
        c_N = c[N]
        
        mu = A_B' \ c_B
        
        lambda_N = c_N - A_N' * mu
        
        if all(lambda_N .>= -1e-10) 
            println("Optimal solution found!")
            
            # Construct solution
            x = zeros(n)
            x_B = A_B \ b
            x[B] = x_B
            
            return (
                optimal = true,
                x = x,
                objective = dot(c, x),
                basis = B,
                non_basis = N,
                iterations = iteration
            )
        end
        
        s = findfirst(lambda_N .< 0)
        if s === nothing
            println("No entering variable found (optimality reached)")
            break
        end
        
        i_s = N[s]
        
        a_is = A[:, i_s]
        h = A_B \ a_is
        
        x_B = A_B \ b
        
        ratios = Float64[]
        ratio_indices = Int[]
        
        for i in 1:length(h)
            if h[i] > 1e-10  # h_i > 0
                push!(ratios, x_B[i] / h[i])
                push!(ratio_indices, i)
            end
        end
        
        if isempty(ratios)
            println("Unbounded problem, no solution")
            return (
                optimal = false,
                unbounded = true,
                iterations = iteration
            )
        end
        
        min_ratio_idx = argmin(ratios)
        j = ratio_indices[min_ratio_idx]
        alpha = ratios[min_ratio_idx]
    
        i_j = B[j]
        
        B[j] = i_s
        N[s] = i_j
        
        println("Iteration $iteration: Variable $i_s enters, variable $i_j leaves")
    end
    
    if iteration >= max_iterations
        println("Maximum iterations reached")
    end
    
    x = zeros(n)
    A_B = A[:, B]
    x_B = A_B \ b
    x[B] = x_B
    
    return (
        optimal = false,
        x = x,
        objective = dot(c, x),
        basis = B,
        non_basis = N,
        iterations = iteration
    )
end


function example_problem()
    
    A = [1.0  1.0  1.0  0.0;
         2.0  1.0  0.0  1.0]
    
    b = [4.0; 6.0]
    
    c = [-1.0; -2.0; 0.0; 0.0] 
    
    result = revised_simplex(A, b, c)
    
    println("\n=== Results ===")
    println("Optimal: ", result.optimal)
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


function revised_simplex_chat(A::Matrix{Float64}, b::Vector{Float64}, c::Vector{Float64}, x0::Vector{Float64})
    m, n = size(A)

    # Derive basis from x0
    B = findall(x0 .> 1e-10)
    N = findall(x0 .<= 1e-10)

    # Pad basis to size m if needed
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
        lambda_N = c_N - A_N' * mu

        if all(lambda_N .>= -1e-10)
            x = zeros(n)
            x[B] = A_B \ b
            return (optimal=true, x=x, objective=dot(c,x), basis=B, iterations=iteration)
        end

        s   = findfirst(lambda_N .< 0)
        i_s = N[s]
        h   = A_B \ A[:, i_s]
        x_B = A_B \ b

        ratios       = [(x_B[i]/h[i], i) for i in 1:m if h[i] > 1e-10]
        isempty(ratios) && return (optimal=false, unbounded=true, iterations=iteration)

        _, j = argmin(first, ratios) |> x -> ratios[findfirst(r -> r == x, ratios)]
        # simpler:
        j    = argmin([h[i] > 1e-10 ? x_B[i]/h[i] : Inf for i in 1:m])
        i_j  = B[j]

        B[j] = i_s
        N[s] = i_j
    end

    x = zeros(n)
    x[B] = A[:, B] \ b
    return (optimal=false, x=x, objective=dot(c,x), basis=B, iterations=max_iterations)
end