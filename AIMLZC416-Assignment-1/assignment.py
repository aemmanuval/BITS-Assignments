"""
AIMLZC416 — Mathematical Foundations for Machine Learning
Assignment I — Complete solutions (Q1 and Q2)

Notes:
- Descriptive comment lines are included as required.
- Built-in linear-algebra functions are NOT used for REF, RREF, rank,
  covariance, or power method. numpy.random is used only for generating
  random data; numpy.linalg is used only in Q2(d) where the assignment
  explicitly allows "Python function".
- All random entities are decimal (n.dddddddd), not integer.
"""

import math
import numpy as np

np.random.seed(42)

# =====================================================================
# Utility helpers (pure Python, no numpy linear algebra)
# =====================================================================

def matrix_copy(A):
    """Deep copy of a 2-D list matrix."""
    return [row[:] for row in A]


def augment(A, b):
    """Construct augmented matrix [A | b] from A (m×n) and b (m×1)."""
    return [A[i][:] + [b[i][0]] for i in range(len(A))]


def print_matrix(M, title=""):
    """Pretty-print a matrix with optional title."""
    if title:
        print(title)
    for row in M:
        print("  [" + ", ".join(f"{v:12.8f}" for v in row) + "]")
    print()


def print_vector(v, title=""):
    """Pretty-print a column vector (list of [val] rows)."""
    if title:
        print(title)
    for row in v:
        print(f"  [{row[0]:12.8f}]")
    print()


# =====================================================================
# Q1.1  REF and RREF without built-in functions
# =====================================================================

def swap_rows(M, r1, r2):
    """Swap rows r1 and r2 in place."""
    M[r1], M[r2] = M[r2], M[r1]


def to_ref(aug, tol=1e-12):
    """
    Row Echelon Form via Gaussian elimination with partial pivoting.
    Raises ZeroDivisionError when a zero pivot is encountered so that the
    caller can choose a different A and/or b (as required by the assignment).
    """
    M = matrix_copy(aug)
    m = len(M)
    n_cols = len(M[0])
    pivot_row = 0

    for col in range(n_cols - 1):
        if pivot_row >= m:
            break

        # Partial pivoting: pick the row with the largest absolute value
        best_row = pivot_row
        best_val = abs(M[pivot_row][col])
        for r in range(pivot_row + 1, m):
            if abs(M[r][col]) > best_val:
                best_val = abs(M[r][col])
                best_row = r

        if best_val < tol:
            # All entries in this column are ~0 → skip (no pivot here)
            continue

        if best_row != pivot_row:
            swap_rows(M, pivot_row, best_row)

        pivot_val = M[pivot_row][col]
        # Guard: if pivot is zero despite the check above, signal the caller
        if abs(pivot_val) < tol:
            raise ZeroDivisionError(
                "Zero pivot encountered — choose a different A and/or b."
            )

        # Eliminate entries below the pivot
        for r in range(pivot_row + 1, m):
            if abs(M[r][col]) < tol:
                continue
            factor = M[r][col] / pivot_val
            for j in range(n_cols):
                M[r][j] -= factor * M[pivot_row][j]

        pivot_row += 1

    return M


def to_rref(aug, tol=1e-12):
    """
    Reduced Row Echelon Form: first produce REF, then back-substitute
    to make each pivot equal to 1 and eliminate all entries above pivots.
    """
    M = to_ref(aug, tol)
    m = len(M)
    n_cols = len(M[0])
    pivot_cols = []
    row = 0

    for col in range(n_cols - 1):
        if row >= m:
            break
        if abs(M[row][col]) < tol:
            continue

        pivot_cols.append(col)
        pivot_val = M[row][col]

        # Scale pivot row so pivot becomes 1
        for j in range(n_cols):
            M[row][j] /= pivot_val

        # Eliminate all other entries in this column
        for r in range(m):
            if r != row and abs(M[r][col]) > tol:
                factor = M[r][col]
                for j in range(n_cols):
                    M[r][j] -= factor * M[row][j]
        row += 1

    return M, pivot_cols


# =====================================================================
# Q1.2  Pivot / non-pivot columns, particular solution, null space
# =====================================================================

def mat_vec_mult(A, x):
    """A (m×n) times x (n×1 column vector stored as [[v],…])."""
    return [[sum(A[i][j] * x[j][0] for j in range(len(A[0])))]
            for i in range(len(A))]


def particular_and_nullspace(A, b, tol=1e-10):
    """
    From the RREF of [A|b], extract:
      - pivot and free (non-pivot) columns
      - one particular solution x_p  (free variables set to 0)
      - a basis for the null space of A  (one vector per free variable)
    """
    n = len(A[0])
    m = len(A)
    R, pivot_cols = to_rref(augment(A, b), tol)
    free_cols = [c for c in range(n) if c not in pivot_cols]

    # ── Particular solution (set free variables to 0) ──
    x_p = [[0.0] for _ in range(n)]
    for r in range(m):
        # Find the pivot column in this row
        pc = None
        for c in range(n):
            if abs(R[r][c]) > tol:
                pc = c
                break
        if pc is None:
            # Row of all zeros on the left — check consistency
            if abs(R[r][n]) > tol:
                raise ValueError("System is inconsistent (no solution).")
            continue
        x_p[pc][0] = R[r][n]  # since pivot is 1 and free vars are 0

    # ── Null-space basis (one vector per free column) ──
    basis = []
    for f in free_cols:
        v = [[0.0] for _ in range(n)]
        v[f][0] = 1.0  # free variable = 1
        for r in range(m):
            pc = None
            for c in range(n):
                if abs(R[r][c]) > tol:
                    pc = c
                    break
            if pc is None or pc == f:
                continue
            v[pc][0] = -R[r][f]  # from the RREF equation: x_pc + … + R[r][f]*x_f + … = 0
        basis.append(v)

    return x_p, basis, pivot_cols, free_cols


# =====================================================================
# Q1.3  Demonstration with a random 5×7 system
# =====================================================================

def run_q1():
    """Generate random 5×7 A and b, show REF, RREF, solutions, verification."""
    print("=" * 72)
    print("Q1) Finding solutions of linear systems")
    print("=" * 72)

    m, n = 5, 7

    # Retry loop: choose a different A / b if division by zero or inconsistency
    A = b = None
    for attempt in range(100):
        A_try = np.random.uniform(-5, 5, (m, n)).tolist()
        b_try = [[np.random.uniform(-5, 5)] for _ in range(m)]
        try:
            _ = to_ref(augment(A_try, b_try))
            particular_and_nullspace(A_try, b_try)
            A, b = A_try, b_try
            break
        except (ZeroDivisionError, ValueError):
            # Division by zero or inconsistency → choose different A and/or b
            continue

    if A is None:
        raise RuntimeError("Could not find a suitable random A, b after retries.")

    print(f"\nMatrix A ({m}×{n}):")
    print_matrix(A)
    print_vector(b, "Vector b:")

    # REF
    aug = augment(A, b)
    ref_M = to_ref(aug)
    print_matrix(ref_M, "REF of [A | b]:")

    # RREF
    rref_M, _ = to_rref(aug)
    print_matrix(rref_M, "RREF of [A | b]:")

    # Pivot / free columns, solutions
    x_p, basis, pivot_cols, free_cols = particular_and_nullspace(A, b)
    print(f"Pivot columns (0-indexed):      {pivot_cols}")
    print(f"Non-pivot (free) columns:       {free_cols}\n")
    print_vector(x_p, "Particular solution x_p (free vars = 0):")

    print(f"Null-space basis ({len(basis)} vector(s)):")
    for k, v in enumerate(basis):
        print_vector(v, f"  null vector v{k+1}:")

    # General solution: x = x_p + c1*v1 + c2*v2 + …
    coeffs = [1.23456789, -0.87654321, 2.34567890]
    x_gen = matrix_copy(x_p)
    coeff_str = []
    for k, v in enumerate(basis):
        ck = coeffs[k] if k < len(coeffs) else float(k + 1)
        coeff_str.append(f"{ck:+.8f}*v{k+1}")
        for i in range(n):
            x_gen[i][0] += ck * v[i][0]

    print(f"General solution x = x_p {' '.join(coeff_str)} :")
    print_vector(x_gen)

    # Verification: Ax_gen should equal b
    Ax = mat_vec_mult(A, x_gen)
    res = max(abs(Ax[i][0] - b[i][0]) for i in range(m))
    print(f"Verification  ||A x_gen - b||_inf = {res:.2e}  (should be ≈ 0)")

    for k, v in enumerate(basis):
        Av = mat_vec_mult(A, v)
        nrm = max(abs(Av[i][0]) for i in range(m))
        print(f"Verification  ||A v{k+1}||_inf     = {nrm:.2e}  (should be ≈ 0)")
    print()


# =====================================================================
# Q2  Dataset, rank, covariance, power method
# =====================================================================

# ── Q2.1  Generate X ∈ R^{500×6} ──

def generate_dataset(n_rows=500):
    """
    f1..f4 ~ N(0,1)  (standard normal, generated via numpy.random)
    f5 = 2*f1 + 3*f2
    f6 = f3 - 2*f4
    Returns a plain Python list-of-lists (n_rows × 6).
    """
    f1 = np.random.randn(n_rows)
    f2 = np.random.randn(n_rows)
    f3 = np.random.randn(n_rows)
    f4 = np.random.randn(n_rows)
    f5 = 2.0 * f1 + 3.0 * f2
    f6 = f3 - 2.0 * f4

    X = []
    for i in range(n_rows):
        X.append([f1[i], f2[i], f3[i], f4[i], f5[i], f6[i]])
    return X


# ── Q2.2  Rank (via our REF — no numpy.linalg) ──

def matrix_rank(M, tol=1e-10):
    """Rank = number of non-zero rows in REF."""
    m = len(M)
    n = len(M[0])
    aug = [row[:] + [0.0] for row in M]   # dummy augmented column
    R = to_ref(aug, tol)
    rank = 0
    for r in range(m):
        if any(abs(R[r][c]) > tol for c in range(n)):
            rank += 1
    return rank


# ── Q2.3(a)  Covariance C = (1/n) X^T X ──

def transpose(M):
    """Transpose of list-of-lists matrix."""
    return [[M[r][c] for r in range(len(M))] for c in range(len(M[0]))]


def mat_mult(A, B):
    """Matrix multiply A (m×k) by B (k×n) — pure Python."""
    m, k, n = len(A), len(A[0]), len(B[0])
    C = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for t in range(k):
                s += A[i][t] * B[t][j]
            C[i][j] = s
    return C


def covariance_matrix(X):
    """Compute C = (1/n) X^T X without numpy."""
    n = len(X)
    p = len(X[0])
    XT = transpose(X)
    C = mat_mult(XT, X)
    for i in range(p):
        for j in range(p):
            C[i][j] /= n
    return C


# ── Q2.3(b)  Power method ──

def vec_normalize(v):
    """Normalize a list-vector to unit length."""
    norm = math.sqrt(sum(x * x for x in v))
    if norm < 1e-15:
        return v[:]
    return [x / norm for x in v]


def mat_vec(C, v):
    """Multiply square matrix C (list-of-lists) by vector v (list)."""
    return [sum(C[i][j] * v[j] for j in range(len(v))) for i in range(len(C))]


def power_method(C, tol=1e-7, max_iter=100000):
    """
    Power method: iteratively compute  v ← C v / ||C v||  and track the
    Rayleigh quotient  λ = v^T C v  until successive estimates differ by < tol.
    Returns (eigenvalue, eigenvector, iteration_count).
    """
    n = len(C)
    v = vec_normalize([np.random.randn() for _ in range(n)])

    lam_old = 0.0
    for it in range(1, max_iter + 1):
        w = mat_vec(C, v)
        lam = sum(v[i] * w[i] for i in range(n))    # Rayleigh quotient
        v = vec_normalize(w)
        if abs(lam - lam_old) < tol:
            return lam, v, it
        lam_old = lam
    return lam_old, v, max_iter


# ── Q2.3(c)  Deflation: C_deflated = C − Σ_j  v_j v_j^T C ──

def deflate(C, eigenvectors):
    """
    Hotelling deflation: subtract the projections of C along the already-found
    eigenvectors so the power method converges to the next eigenvalue.
    """
    D = matrix_copy(C)
    p = len(C)
    for v in eigenvectors:
        # Compute v^T C  (row vector)
        vtC = [sum(v[k] * C[k][j] for k in range(p)) for j in range(p)]
        # Subtract v (v^T C)
        for i in range(p):
            for j in range(p):
                D[i][j] -= v[i] * vtC[j]
    return D


def run_q2():
    """Execute all Q2 sub-parts."""
    print("=" * 72)
    print("Q2) Dataset X, rank, covariance, power method")
    print("=" * 72)

    # ── Q2.1 ──
    n = 500
    X = generate_dataset(n_rows=n)
    p = len(X[0])
    print(f"\nQ2.1  Generated X: {n} × {p}")
    print("  First row:")
    print("  " + ", ".join(f"{x:.8f}" for x in X[0]))
    print("  Last row:")
    print("  " + ", ".join(f"{x:.8f}" for x in X[-1]))

    # ── Q2.2 ──
    rank_X = matrix_rank(X)
    print(f"\nQ2.2  rank(X) = {rank_X}")
    print("  (Expected 4 because f5, f6 are linear combinations of f1..f4)")

    # ── Q2.3(a) ──
    C = covariance_matrix(X)
    print(f"\nQ2.3(a)  Covariance matrix C = (1/{n}) X^T X   ({p}×{p}):")
    print_matrix(C)

    # ── Q2.3(b) ──
    lam1, v1, it1 = power_method(C)
    print(f"Q2.3(b)  Largest eigenvalue via power method:")
    print(f"  λ_1 = {lam1:.8f}   (converged in {it1} iterations)")
    print(f"  v_1 = [" + ", ".join(f"{x:.8f}" for x in v1) + "]\n")

    # ── Q2.3(c) — find ALL eigenvalues by successive deflation ──
    eigenvalues = []
    eigenvectors = []
    C_work = matrix_copy(C)

    print("Q2.3(c)  Successive deflation (C ← C − Σ v_j v_j^T C):\n")
    for k in range(p):
        lam_k, v_k, it_k = power_method(C_work)
        # C = (1/n) X^T X is positive semi-definite, so every true eigenvalue
        # is >= 0.  A negative result or a value negligibly small compared to
        # λ_1 is a numerical artifact of deflation → treat as zero.
        if lam_k < 0 or (eigenvalues and abs(lam_k) < 1e-6 * abs(eigenvalues[0])):
            lam_k = 0.0
        eigenvalues.append(lam_k)
        eigenvectors.append(v_k)
        C_work = deflate(C, eigenvectors)
        print(f"  Step {k+1}:  λ_{k+1} = {lam_k:12.8f}   iters = {it_k}")
        print(f"          v_{k+1} = [" + ", ".join(f"{x:.8f}" for x in v_k) + "]")

    print(f"\n  All eigenvalues (power method + deflation):")
    for k in range(p):
        print(f"    λ_{k+1} = {eigenvalues[k]:.8f}")

    # ── Q2.3(d) — numpy built-in (explicitly allowed) ──
    C_np = np.array(C)
    evals_np, evecs_np = np.linalg.eigh(C_np)
    idx = np.argsort(evals_np)[::-1]
    evals_np = evals_np[idx]
    evecs_np = evecs_np[:, idx]

    print(f"\nQ2.3(d)  Eigenvalues via numpy.linalg.eigh (descending):")
    for k in range(p):
        print(f"    λ_{k+1} = {evals_np[k]:.8f}")

    print(f"\n  Eigenvectors (columns of V, matching eigenvalue order):")
    for k in range(p):
        vec = evecs_np[:, k]
        print(f"    v_{k+1} = [" + ", ".join(f"{x:.8f}" for x in vec) + "]")

    print(f"\n  Comparison (power-method vs numpy):")
    for k in range(p):
        diff = abs(eigenvalues[k] - evals_np[k])
        print(f"    |λ_{k+1}^PM − λ_{k+1}^NP| = {diff:.2e}")

    # ── Q2.3(e) — iteration count to 10^-7 accuracy (reference = numpy) ──
    print(f"\nQ2.3(e)  Iterations for |λ_PM − λ_true| < 10^-7  "
          f"(λ_true = numpy eigenvalues):\n")
    vecs_e = []
    for k in range(p):
        lam_true = evals_np[k]
        C_def = deflate(C, vecs_e) if vecs_e else matrix_copy(C)

        # If the true eigenvalue is essentially zero, a few iterations suffice
        if abs(lam_true) < 1e-12:
            print(f"    Eigenpair {k+1}:  λ_true ≈ 0  →  "
                  f"eigenvalue is zero (rank(X) = {rank_X}), no iteration needed")
            vecs_e.append([0.0] * p)
            continue

        # Run power method, measuring when we first get within 1e-7 of truth
        nn = len(C_def)
        v = vec_normalize([np.random.randn() for _ in range(nn)])
        lam_old = 0.0
        it_needed = None
        for it in range(1, 200001):
            w = mat_vec(C_def, v)
            lam = sum(v[i] * w[i] for i in range(nn))
            v = vec_normalize(w)
            if abs(lam - lam_true) < 1e-7:
                it_needed = it
                break
            lam_old = lam

        if it_needed is None:
            it_needed = "> 200000"
        print(f"    Eigenpair {k+1}:  λ_true = {lam_true:.8f},  "
              f"iterations = {it_needed}")
        vecs_e.append(v)
    print()


# =====================================================================
# Main
# =====================================================================

def main():
    run_q1()
    run_q2()


if __name__ == "__main__":
    main()
