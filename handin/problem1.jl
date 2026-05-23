using LinearAlgebra, Plots, SparseArrays
using JuMP
using Ipopt
using Random, Distributions


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
    M = percent_matrix(N, density, n, n) # changed to n x n
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
		hcat(-A', spzeros(m_a, m_a))
	]
	rhs = [-g; -b]

	return KKT_matrix, rhs
end 

function LDL_solver(KKT_matrix, rhs)
    if issparse(KKT_matrix)
        # this defaults to sparse LU and thus handles sparse indefinite systems
        return KKT_matrix \ rhs
    else
        # Bunch-Kaufman is for dense symmetric indefinite systems
        return bunchkaufman(Symmetric(KKT_matrix)) \ rhs
    end
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

if abspath(PROGRAM_FILE) == @__FILE__
    # 3. Test problem 
    n = 100
    alpha = 0.5
    density = 0.15
    beta = 0.75
    flag = "sparse"
    println("density at $(density)")
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

    function timing_study(sizes, betas; alpha=0.5, density=0.15, n_runs=3)
        results = Dict{Float64, Tuple{Vector{Float64}, Vector{Float64}}}()

        for β in betas
            t_dense  = zeros(length(sizes))
            t_sparse = zeros(length(sizes))

            for (i, n) in enumerate(sizes)
                for _ in 1:n_runs
                    H, g_n, A, b_n = RandomEQP(n, alpha, density, β, "dense")

                    t_dense[i]  += @elapsed EqualityQPSolverLDLdense(H,         g_n, A,         b_n)
                    t_sparse[i] += @elapsed EqualityQPSolverLDLsparse(sparse(H), g_n, sparse(A), b_n)
                end
                t_dense[i]  /= n_runs
                t_sparse[i] /= n_runs

                println("  β=$β  n=$n  dense=$(round(t_dense[i]*1000, digits=2))ms  " *
                        "sparse=$(round(t_sparse[i]*1000, digits=2))ms")
            end

            results[β] = (t_dense, t_sparse)
        end

        return results
    end


    sizes = [50, 100, 200, 300, 400, 500,800,1200, 2000]
    betas = [0.25, 0.50, 0.75]

    println("\n=== H density check (density=$(density*100)%, α=$alpha) ===")
    for β in betas
        println("  β=$β:")
        for n_chk in [50, 200, 500, 1000]
            H_chk, _, A_chk, _ = RandomEQP(n_chk, alpha, density, β, "dense")
            d_H = count(!iszero, H_chk) / length(H_chk)
            d_A = count(!iszero, A_chk) / length(A_chk)
            println("    n=$n_chk  H density=$(round(d_H*100, digits=1))%  A density=$(round(d_A*100, digits=1))%")
        end
    end

    println("\nRunning timing study...")
    results = timing_study(sizes, betas)

    plots = []
    for β in betas
        t_dense, t_sparse = results[β]
        m_sizes = Int.(round.(β .* sizes))

        p = plot(sizes, [t_dense t_sparse],
                label      = ["Dense" "Sparse"],
                xlabel     = "n",
                ylabel     = "CPU time (s)",
                title      = "β = $β  (m = βn constraints)",
                marker     = :circle,
                linewidth  = 2,
                legend     = :topleft,
                yscale     = :log10)

        # annotate m values on x-axis for context
        annotate!(p, sizes[end], t_dense[end],
                text("m=$(m_sizes[end])", 8, :right))
        push!(plots, p)
    end

    fig = plot(plots...,
        layout = (1, 3),
        size   = (1100, 380),
        plot_title = "CPU time vs problem size n  (density=15%)",
        plot_titlefontsize = 12,
        titlefont=font(10,"Computer Modern")
    )
    savefig(fig, "handin/im/timing_exercise_15pct.png")
end