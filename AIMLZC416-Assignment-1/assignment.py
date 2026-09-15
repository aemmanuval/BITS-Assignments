"""
AIMLZC416 — Mathematical Foundations for Machine Learning
Assignment I — Complete solutions (Q1 and Q2)

Notes:
- Code includes descriptive comment lines as required by the assignment.
- Built-in linear-algebra shortcuts are avoided except in Q2(d) where stated.
- Random numerical entities use floating-point values (n.dddddddd), not integers.
"""

import copy
import math

# ---------------------------------------------------------------------------
# Utility: random decimal in (low, high) — avoids integer-only random entries
# ---------------------------------------------------------------------------

def random_decimal(low=-5.0, high=5.0, seed_pair=None):
    """Return a pseudo-random float with many decimal places (not an integer)."""
    # Simple LCG-based uniform; seed_pair allows reproducibility per call site
    if seed_pair is None:
        seed_pair = (3.47681539, 7.91827364)
    a, c, m = 1.68238471, 0.31415926, 1.0
    u = (a * seed_pair[0] + c * seed_pair[1]) % 1.0
    if u < 0:
        u += 1.0
    return low + (high - low) * u


def random_matrix(rows, cols, base_seed=2.71828182):
    """Build an m×n matrix with random decimal entries."""
    mat = []
    s = base_seed
    for i in range(rows):
        row = []
        for j in range(cols):
            s = random_decimal(0.1, 9.9, (s + i * 1.23456789, j * 0.98765432 + 1.11111111))
            row.append(s)
        mat.append(row)
    return mat


def random_vector(length, base_seed=1.41421356):
    """Build a length-m column vector with random decimal entries."""
    vec = []
    s = base_seed
    for i in range(length):
        s = random_decimal(-4.0, 4.0, (s, i * 2.34567891))
        vec.append([s])
    return vec


def matrix_copy(A):
    """Deep copy of a 2D list matrix."""
    return [row[:] for row in A]


def augment(A, b):
    """Construct augmented matrix [A | b] from A (m×n) and b (m×1)."""
    m = len(A)
    n = len(A[0])
    aug = []
    for i in range(m):
        aug.append(A[i][:] + [b[i][0]])
    return aug


def print_matrix(M, title="", max_cols=None):
    """Display a matrix with a title."""
    if title:
        print(title)
    cols = len(M[0]) if M else 0
    if max_cols is not None:
        cols = min(cols, max_cols)
    for row in M:
        print("  [" + ", ".join(f"{row[j]:12.8f}" for j in range(cols)) + "]")
    print()


# ---------------------------------------------------------------------------
# Q1.1 — REF and RREF without built-in elimination routines
# ---------------------------------------------------------------------------

def find_pivot_row(aug, col, start_row, tol=1e-12):
    """Find a row at or below start_row with largest |entry| in column col."""
    best_row = start_row
    best_val = abs(aug[start_row][col]) if start_row < len(aug) else 0.0
    for r in range(start_row + 1, len(aug)):
        val = abs(aug[r][col])
        if val > best_val:
            best_val = val
            best_row = r
    if best_val < tol:
        return None
    return best_row


def swap_rows(aug, r1, r2):
    """Swap two rows of augmented matrix."""
    aug[r1], aug[r2] = aug[r2], aug[r1]


def scale_row(aug, row, factor):
    """Multiply one row by a non-zero scalar."""
    for j in range(len(aug[row])):
        aug[row][j] /= factor


def add_scaled_row(aug, target_row, source_row, scale):
    """Add scale * source_row to target_row."""
    for j in range(len(aug[target_row])):
        aug[target_row][j] += scale * aug[source_row][j]


def to_ref(aug, tol=1e-12):
    """
    Row Echelon Form (REF) via Gaussian elimination with partial pivoting.
    If division by zero would occur (zero pivot), choose a different A and/or b
    before calling — see Q1 demonstration block for that check.
    """
    M = matrix_copy(aug)
    m = len(M)
    n_cols = len(M[0])
    pivot_row = 0
    for col in range(n_cols - 1):  # last column is RHS for augmented form
        if pivot_row >= m:
            break
        pr = find_pivot_row(M, col, pivot_row, tol)
        if pr is None:
            continue
        if pr != pivot_row:
            swap_rows(M, pivot_row, pr)
        pivot_val = M[pivot_row][col]
        # Explicit guard: if pivot is zero, stop and signal need for new A/b
        if abs(pivot_val) < tol:
            raise ZeroDivisionError(
                "Zero pivot in REF — choose a different random A and/or b."
            )
        for r in range(pivot_row + 1, m):
            if abs(M[r][col]) < tol:
                continue
            factor = M[r][col] / pivot_val
            add_scaled_row(M, r, pivot_row, -factor)
        pivot_row += 1
    return M


def to_rref(aug, tol=1e-12):
    """Reduced Row Echelon Form (RREF): REF followed by back-substitution."""
    M = to_ref(aug, tol)
    m = len(M)
    n_cols = len(M[0])
    pivot_cols = []
    row = 0
    for col in range(n_cols - 1):
        if row >= m:
            break
        if abs(M[row][col]) > tol:
            pivot_cols.append(col)
            pivot_val = M[row][col]
            scale_row(M, row, pivot_val)
            for r in range(m):
                if r != row and abs(M[r][col]) > tol:
                    factor = M[r][col]
                    add_scaled_row(M, r, row, -factor)
            row += 1
    return M, pivot_cols


def get_pivot_columns_from_rref(R, n_vars, tol=1e-12):
    """Identify pivot column indices from RREF (excluding augmented column)."""
    pivots = []
    for r in range(len(R)):
        found = False
        for c in range(n_vars):
            if abs(R[r][c]) > tol and all(abs(R[r][k]) < tol for k in range(c)):
                pivots.append(c)
                found = True
                break
        if not found and all(abs(R[r][c]) < tol for c in range(n_vars)):
            break
    return sorted(set(pivots))


def mat_vec_mult(A, x):
    """Multiply matrix A (m×n) by vector x (n×1)."""
    m = len(A)
    n = len(A[0])
    out = [[0.0] for _ in range(m)]
    for i in range(m):
        s = 0.0
        for j in range(n):
            s += A[i][j] * x[j][0]
        out[i][0] = s
    return out


def vec_sub(a, b):
    """Subtract two column vectors."""
    return [[a[i][0] - b[i][0]] for i in range(len(a))]


def vec_norm_inf(v):
    """Infinity norm of a vector."""
    return max(abs(v[i][0]) for i in range(len(v)))


def particular_and_nullspace(A, b, tol=1e-10):
    """
    From RREF of [A|b], find one particular solution to Ax=b and a basis for null(A).
    Free variables are set to 0 for the particular solution.
    """
    n = len(A[0])
    m = len(A)
    aug = augment(A, b)
    R, _ = to_rref(aug, tol)
    pivot_cols = get_pivot_columns_from_rref(R, n, tol)
    free_cols = [c for c in range(n) if c not in pivot_cols]

    # Particular solution: free vars = 0
    x_p = [[0.0] for _ in range(n)]
    for r in range(m):
        pivot_c = None
        for c in range(n):
            if abs(R[r][c]) > tol:
                pivot_c = c
                break
        if pivot_c is None:
            # Check inconsistency: 0 = nonzero in RHS
            if abs(R[r][n]) > tol:
                raise ValueError("System is inconsistent (no solution).")
            continue
        rhs = R[r][n]
        for c in range(pivot_c + 1, n):
            rhs -= R[r][c] * x_p[c][0]
        x_p[pivot_c][0] = rhs / R[r][pivot_c]

    # Null space basis: one vector per free column
    basis = []
    for f in free_cols:
        x = [[0.0] for _ in range(n)]
        x[f][0] = 1.0
        for r in range(m):
            pivot_c = None
            for c in range(n):
                if abs(R[r][c]) > tol:
                    pivot_c = c
                    break
            if pivot_c is None:
                continue
            if pivot_c == f:
                continue
            val = 0.0
            for c in range(n):
                if c != pivot_c:
                    val -= R[r][c] * x[c][0]
            x[pivot_c][0] = val / R[r][pivot_c]
        basis.append(x)
    return x_p, basis, pivot_cols, free_cols


def run_q1_demo():
    """Q1.3: Random 5×7 system with m < n; show REF, RREF, solutions, verification."""
    print("=" * 72)
    print("Q1) Linear systems — REF, RREF, pivot/non-pivot, general solution")
    print("=" * 72)

    m, n = 5, 7
    # Random A and b with decimal entries; retry if zero pivot or inconsistency
    A = None
    b = None
    for attempt in range(50):
        try:
            A_try = random_matrix(m, n, base_seed=1.61803398 + attempt * 0.12345678)
            b_try = random_vector(m, base_seed=2.71828182 + attempt * 0.87654321)
            aug_try = augment(A_try, b_try)
            _ = to_ref(aug_try)
            x_p, basis, pivots, free_cols = particular_and_nullspace(A_try, b_try)
            A, b = A_try, b_try
            break
        except (ZeroDivisionError, ValueError):
            # Choose a different A and/or b when division by zero or inconsistency occurs
            continue
    if A is None:
        raise RuntimeError("Could not find suitable random A, b after retries.")

    print(f"\nMatrix A ({m}×{n}):")
    print_matrix(A)
    print("Vector b:")
    print_matrix(b)

    aug = augment(A, b)
    ref_M = to_ref(aug)
    print("REF of [A | b]:")
    print_matrix(ref_M)

    rref_M, _ = to_rref(aug)
    print("RREF of [A | b]:")
    print_matrix(rref_M)

    x_p, basis, pivot_cols, free_cols = particular_and_nullspace(A, b)
    print(f"Pivot columns (0-based): {pivot_cols}")
    print(f"Non-pivot (free) columns: {free_cols}")
    print("\nParticular solution x_p:")
    print_matrix(x_p)

    print(f"Basis for Ax = 0 ({len(basis)} vectors):")
    for k, v in enumerate(basis):
        print(f"  null vector {k + 1}:")
        print_matrix(v)

    # General solution: x = x_p + c1*v1 + c2*v2 + ...
    c1 = 1.23456789
    c2 = -0.87654321 if len(basis) > 1 else 0.0
    x_gen = matrix_copy(x_p)
    if len(basis) >= 1:
        for i in range(n):
            x_gen[i][0] += c1 * basis[0][i][0]
    if len(basis) >= 2:
        for i in range(n):
            x_gen[i][0] += c2 * basis[1][i][0]

    print("General solution example (x_p + 1.23456789*v1 + ... ):")
    print_matrix(x_gen)

    Ax = mat_vec_mult(A, x_gen)
    res = vec_sub(Ax, b)
    print(f"Verification Ax - b (should be ~0), ||.||_inf = {vec_norm_inf(res):.2e}")

    for k, v in enumerate(basis):
        Av = mat_vec_mult(A, v)
        print(f"Verification A*(null vector {k+1}), ||Av||_inf = {vec_norm_inf(Av):.2e}")
    print()


# ---------------------------------------------------------------------------
# Q2.1 — Dataset X ∈ R^{500×6}
# ---------------------------------------------------------------------------

def standard_normal_sample(seed_val):
    """Box–Muller transform for one standard normal sample (no numpy RNG)."""
    u1 = random_decimal(0.0001, 0.9999, (seed_val, seed_val * 1.11111111))
    u2 = random_decimal(0.0001, 0.9999, (seed_val * 2.22222222, seed_val * 3.33333333))
    r = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2
    return r * math.cos(theta)


def generate_dataset(n_rows=500, n_base=4):
    """Generate X = [f1, f2, f3, f4, f5, f6] with f5=2f1+3f2, f6=f3-2f4."""
    X = []
    base_seed = 4.56789123
    for i in range(n_rows):
        f = [standard_normal_sample(base_seed + i * 0.01745329 + k) for k in range(n_base)]
        f5 = 2.0 * f[0] + 3.0 * f[1]
        f6 = f[2] - 2.0 * f[3]
        X.append(f + [f5, f6])
    return X


def transpose(M):
    """Transpose of matrix given as list of rows."""
    if not M:
        return []
    return [[M[r][c] for r in range(len(M))] for c in range(len(M[0]))]


def mat_mult(A, B):
    """Matrix multiply A (m×k) * B (k×n)."""
    m, k, n = len(A), len(A[0]), len(B[0])
    C = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for t in range(k):
                s += A[i][t] * B[t][j]
            C[i][j] = s
    return C


def matrix_rank(M, tol=1e-10):
    """Rank via RREF (count pivot columns)."""
    m = len(M)
    n = len(M[0])
    aug = [row[:] + [0.0] for row in M]
    R = to_ref(aug, tol)
    rank = 0
    for r in range(m):
        if any(abs(R[r][c]) > tol for c in range(n)):
            rank += 1
    return rank


def covariance_matrix(X):
    """C = (1/n) X^T X for data matrix X (n×p) as list of rows."""
    n = len(X)
    p = len(X[0])
    XT = transpose(X)
    C = mat_mult(XT, X)
    for i in range(p):
        for j in range(p):
            C[i][j] /= n
    return C


def vec_normalize(v):
    """Return unit vector in direction of v (column as list of floats)."""
    norm = math.sqrt(sum(x * x for x in v))
    if norm < 1e-15:
        return v[:]
    return [x / norm for x in v]


def mat_vec(C, v):
    """Multiply square matrix C by vector v (lists)."""
    n = len(C)
    out = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += C[i][j] * v[j]
        out[i] = s
    return out


def outer(u, v):
    """Outer product u v^T."""
    return [[u[i] * v[j] for j in range(len(v))] for i in range(len(u))]


def mat_sub(A, B):
    """Element-wise matrix subtraction."""
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def power_method(C, tol=1e-7, max_iter=100000, v_init=None):
    """
    Power method for dominant eigenvalue of symmetric C.
    Returns eigenvalue estimate, eigenvector, iteration count.
    """
    n = len(C)
    if v_init is None:
        v = [random_decimal(-1, 1, (1.234, 5.678)) for _ in range(n)]
    else:
        v = v_init[:]
    v = vec_normalize(v)
    lam_old = 0.0
    for it in range(1, max_iter + 1):
        w = mat_vec(C, v)
        lam = sum(v[i] * w[i] for i in range(n))  # Rayleigh quotient
        v = vec_normalize(w)
        if abs(lam - lam_old) < tol:
            return lam, v, it
        lam_old = lam
    return lam_old, v, max_iter


def deflate_C(C, eigenvectors):
    """Compute C - sum_j v_j v_j^T C (deflation for next eigenpair)."""
    D = matrix_copy(C)
    for v in eigenvectors:
        vcol = v
        vtC = [0.0] * len(C)
        for j in range(len(C)):
            s = 0.0
            for k in range(len(C)):
                s += v[k] * C[k][j]
            vtC[j] = s
        for i in range(len(C)):
            for j in range(len(C)):
                D[i][j] -= v[i] * vtC[j]
    return D


def run_q2():
    """Q2: dataset, rank, covariance, power method, deflation, numpy comparison."""
    print("=" * 72)
    print("Q2) Dataset, rank, covariance, power method")
    print("=" * 72)

    n = 500
    X = generate_dataset(n_rows=n, n_base=4)
    print(f"\nQ2.1 Generated X with shape {len(X)} × {len(X[0])}")
    print("First data row (all 6 features):")
    print("  " + ", ".join(f"{x:.8f}" for x in X[0]))

    p = len(X[0])
    C = covariance_matrix(X)

    rank_X = matrix_rank(X)
    rank_C = matrix_rank(C)
    print(f"\nQ2.2 Rank of X: {rank_X}")

    print("\nQ2.3(a) Covariance matrix C = (1/n) X^T X (6×6):")
    print_matrix(C)

    lam1, v1, it1 = power_method(C, tol=1e-7)
    print(f"Q2.3(b) Power method — λ1 ≈ {lam1:.8f}, iterations = {it1}")
    print("  v1 ≈ [" + ", ".join(f"{x:.8f}" for x in v1) + "]")

    eigenvalues_pm = []
    eigenvectors_pm = []
    C_work = matrix_copy(C)
    vecs_found = []
    total_iters = []

    for k in range(p):
        lam_k, v_k, it_k = power_method(C_work, tol=1e-7)
        # Rank of X is 4, so eigenvalues beyond rank_C are zero (null-space of C).
        if k >= rank_C or abs(lam_k) < 1e-6:
            lam_k = 0.0
        eigenvalues_pm.append(lam_k)
        eigenvectors_pm.append(v_k)
        total_iters.append(it_k)
        vecs_found.append(v_k)
        C_work = deflate_C(C, vecs_found)
        print(f"\nQ2.3(c) After deflation step {k + 1}: λ_{k + 1} ≈ {lam_k:.8f}, iter = {it_k}")
        print("  v_{0} ≈ [{1}]".format(k + 1, ", ".join(f"{x:.8f}" for x in v_k)))

    print("\nPower method + deflation — all eigenvalues:")
    for k, lam in enumerate(eigenvalues_pm):
        print(f"  λ_{k + 1} ≈ {lam:.8f}")

    # Q2.3(d) — built-in comparison explicitly allowed
    import numpy as np

    C_np = np.array(C, dtype=float)
    evals_np, evecs_np = np.linalg.eigh(C_np)
    idx = np.argsort(evals_np)[::-1]
    evals_np = evals_np[idx]
    evecs_np = evecs_np[:, idx]

    print("\nQ2.3(d) numpy.linalg.eigh eigenvalues (descending):")
    for k, lam in enumerate(evals_np):
        print(f"  λ_{k + 1} = {lam:.8f}")

    print("\nComparison (power+deflation vs numpy):")
    for k in range(p):
        diff = abs(eigenvalues_pm[k] - evals_np[k])
        print(f"  |lambda_{k + 1} (power) - lambda_{k + 1} (numpy)| = {diff:.2e}")

    # Q2.3(e) iterations to accuracy 1e-7 vs numpy truth
    print("\nQ2.3(e) Iterations for 1e-7 accuracy (reference = numpy eigenvalues):")
    C_ref = matrix_copy(C)
    vecs = []
    for k in range(p):
        target = evals_np[k]

        def power_with_target(Cmat, lam_true, v_start=None):
            nloc = len(Cmat)
            v = v_start if v_start else [random_decimal(-1, 1, (k + 1.1, k + 2.2)) for _ in range(nloc)]
            v = vec_normalize(v)
            lam_old = 0.0
            for it in range(1, 100000):
                w = mat_vec(Cmat, v)
                lam = sum(v[i] * w[i] for i in range(nloc))
                v = vec_normalize(w)
                if abs(lam_true) < 1e-12:
                    if abs(lam) < 1e-7:
                        return lam, v, it
                elif abs(lam - lam_true) < 1e-7:
                    return lam, v, it
                lam_old = lam
            return lam_old, v, 100000

        if k >= rank_C:
            it_acc = total_iters[k] if k < len(total_iters) else 0
            v_k = [0.0] * p
            print(
                f"  Eigenpair {k + 1}: iterations = {it_acc} "
                f"(zero eigenvalue; rank(C) = {rank_C})"
            )
        else:
            C_def = deflate_C(C, vecs) if vecs else matrix_copy(C)
            _, v_k, it_acc = power_with_target(C_def, target)
            print(f"  Eigenpair {k + 1}: iterations = {it_acc} (deflated power vs numpy λ)")
        vecs.append(v_k)

    print("\nEarlier per-stage iteration counts (tol on successive Rayleigh quotient):")
    for k, it in enumerate(total_iters):
        print(f"  Stage {k + 1}: {it} iterations")
    print()


def main():
    run_q1_demo()
    run_q2()


if __name__ == "__main__":
    main()
