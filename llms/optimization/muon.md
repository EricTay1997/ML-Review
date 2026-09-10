  - Newton–Schulz pushes every singular value toward 1 **without ever forming an SVD** — 5 steps of a tuned quintic, all matmuls, so it is cheap and GPU-friendly:

    ```python
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G / (G.norm() + eps)
    for _ in range(5):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    ```

    - Those coefficients are tuned for *fast* convergence, not exact convergence — they leave $`\sigma_i`$ near 1 rather than exactly 1, which is deliberate since nothing downstream needs exactness.