import numpy as np


def qpsolver_active_set(H, g, A, b, x0):
    """
    Solves the convex QP:
        min  0.5 x' H x + g' x
         x
        s.t. A' x + b >= 0

    using a primal active set method and a feasible initial point x0.

    Parameters
    ----------
    H : (n, n) array
    g : (n,) array
    A : (n, m) array
    b : (m,) array
    x0 : (n,) array  -- must be feasible

    Returns
    -------
    xopt      : (n,) array or None
    lambdaopt : (m,) array or None
    Wset      : list of int (active constraint indices)
    it        : int
    """
    tol = 1e-8

    n, m = A.shape
    x = x0.copy().astype(float)

    Wset  = []
    IWset = list(range(m))

    lambda_ = np.zeros(m)
    gk      = H @ x + g
    nabla_L = gk - A @ lambda_
    c       = A.T @ x + b      # c(x) = A'x + b >= 0

    kkt_ok = np.linalg.norm(nabla_L, np.inf) < tol

    maxit = 100 * (n + m)
    it = 0

    while not kkt_ok and it < maxit:
        it += 1

        # ── Solve equality-constrained QP subproblem ──────────────────────
        Aw = A[:, Wset] if Wset else np.zeros((n, 0))
        p, lambda_Wset = _equality_qp(H, gk, Aw)

        if np.linalg.norm(p, np.inf) > tol:          # p is non-zero
            alpha = 1.0
            idc   = -1

            for i, idx in enumerate(IWset):
                pA = A[:, idx] @ p
                if pA < 0.0:
                    alpha_cand = -c[idx] / pA
                    if alpha_cand < alpha:
                        alpha = alpha_cand
                        idc   = i

            x  += alpha * p
            gk  = H @ x + g
            c   = A.T @ x + b

            if idc >= 0:
                Wset.append(IWset[idc])
                IWset.pop(idc)

        else:                                          # p is zero
            idlambda    = -1
            min_lambda  =  0.0

            for i, lam in enumerate(lambda_Wset):
                if lam < min_lambda:
                    min_lambda = lam
                    idlambda   = i

            if idlambda >= 0:                          # drop constraint
                IWset.append(Wset[idlambda])
                Wset.pop(idlambda)
            else:                                      # optimal
                kkt_ok    = True
                xopt      = x.copy()
                lambdaopt = np.zeros(m)
                for i, idx in enumerate(Wset):
                    lambdaopt[idx] = lambda_Wset[i]
                return xopt, lambdaopt, Wset, it

    return None, None, [], it


def _equality_qp(H, gk, Aw):
    """
    Solve the equality-constrained QP subproblem:
        min  0.5 p' H p + gk' p
         p
        s.t. Aw' p = 0

    Returns p and the Lagrange multipliers mu.
    """
    n  = H.shape[0]
    nw = Aw.shape[1]

    if nw == 0:
        # no active constraints -> unconstrained step
        p = np.linalg.solve(H, -gk)
        return p, np.zeros(0)

    KKT = np.block([
        [H,        -Aw          ],
        [-Aw.T,    np.zeros((nw, nw))]
    ])
    rhs = np.concatenate([-gk, np.zeros(nw)])
    sol = np.linalg.solve(KKT, rhs)

    p   = sol[:n]
    mu  = sol[n:]
    return p, mu