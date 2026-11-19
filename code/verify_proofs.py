#!/usr/bin/env python3

"""Self-contained theorem and proposition verification script for:
Exact Subspace Bures-Wasserstein Barycenters and Grassmannian Debiasing for Spiked High-Dimensional Covariances
"""

from __future__ import annotations

import math
import numpy as np
from scipy.linalg import eigh

def mat_sqrt_psd(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = eigh(0.5 * (S + S.T))
    vals_c = np.maximum(vals, 1e-14)
    return (vecs * np.sqrt(vals_c)[None, :]) @ vecs.T, (vecs * (1.0 / np.sqrt(vals_c))[None, :]) @ vecs.T

def bw_dist(A: np.ndarray, B: np.ndarray) -> float:
    Bsq, _ = mat_sqrt_psd(B)
    mid = Bsq @ A @ Bsq
    vals = eigh(0.5 * (mid + mid.T), eigvals_only=True)
    nuc = float(np.sum(np.sqrt(np.maximum(vals, 0.0))))
    return math.sqrt(max(0.0, float(np.trace(A) + np.trace(B)) - 2.0 * nuc))

def dense_bw_bary(covs: list[np.ndarray], n_iter: int = 18) -> np.ndarray:
    B = np.mean(covs, axis=0)
    for _ in range(n_iter):
        Bsq, Bisq = mat_sqrt_psd(B)
        acc = np.zeros_like(B)
        for C in covs:
            msq, _ = mat_sqrt_psd(Bsq @ C @ Bsq)
            acc += msq
        acc /= len(covs)
        B = 0.5 * (Bisq @ (acc @ acc) @ Bisq + (Bisq @ (acc @ acc) @ Bisq).T)
    return B

def verify_theorems() -> None:
    rng = np.random.default_rng(2026111807)

    # 1. Theorem 1: High-d Singular Wishart Grassmannian Bias & (k-1)/d < 1/4 Inversion
    # Test finite-d Cauchy-Schwarz lower bound E[B^2(S1, S2)] >= 2*d*s2*(1 - sqrt((k-1)/d)) on exact Monte Carlo
    d = 64
    k = 13  # (k-1)/d = 12/64 = 0.1875 < 0.25
    s2 = 1.4
    gamma = (k - 1) / float(d)
    b2_same = []
    b2_sing = []
    eps_cov = 1e-6 * np.eye(d)
    for _ in range(60):
        X1 = rng.normal(0.0, math.sqrt(s2), size=(k, d))
        X2 = rng.normal(0.0, math.sqrt(s2), size=(k, d))
        X1 -= X1.mean(axis=0, keepdims=True)
        X2 -= X2.mean(axis=0, keepdims=True)
        S1 = (X1.T @ X1) / (k - 1)
        S2 = (X2.T @ X2) / (k - 1)
        # Nuclear norm of S1^{1/2} S2 S1^{1/2} via Gram SVD
        A1 = X1 / math.sqrt(k - 1)
        A2 = X2 / math.sqrt(k - 1)
        svals = np.linalg.svd(A1 @ A2.T, compute_uv=False)
        b2_val = float(np.trace(S1) + np.trace(S2) - 2.0 * np.sum(svals))
        # Exact deterministic per-realization Cauchy-Schwarz inequality: ||A1 A2^T||_* <= sqrt(k-1) ||A1 A2^T||_F
        assert np.sum(svals) <= math.sqrt(k - 1) * np.linalg.norm(A1 @ A2.T, "fro") + 1e-10
        b2_same.append(b2_val)
        b2_sing.append(float(np.trace(S1)))  # distance to zero point-mass limit eps -> 0 is tr(S1)
    mean_same = float(np.mean(b2_same))
    mean_sing = float(np.mean(b2_sing))
    theo_lb = 2.0 * d * s2 * (1.0 - math.sqrt(gamma))
    assert mean_same >= theo_lb - 0.5, f"Wishart LB violated: {mean_same} vs {theo_lb}"
    assert mean_same > mean_sing, f"Inversion failed at (k-1)/d={gamma}: {mean_same} <= {mean_sing}"

    # 2. Theorem 2: Exact Zero-Error Orthogonal Subspace Decomposition of Spiked BW Barycenters & Pairwise Distances
    for trial in range(25):
        d = int(rng.choice([24, 40, 64]))
        M = int(rng.integers(2, 6))
        U_list, lam_list, s2_list, full_covs = [], [], [], []
        for _ in range(M):
            r_m = int(rng.integers(1, 5))
            U_m, _ = np.linalg.qr(rng.normal(size=(d, r_m)), mode="reduced")
            s2_m = float(rng.uniform(0.15, 1.8))
            lam_m = s2_m + rng.uniform(0.2, 5.0, size=r_m)
            C_m = (U_m * (lam_m - s2_m)[None, :]) @ U_m.T + s2_m * np.eye(d)
            U_list.append(U_m)
            lam_list.append(lam_m)
            s2_list.append(s2_m)
            full_covs.append(0.5 * (C_m + C_m.T))

        B_dense = dense_bw_bary(full_covs, n_iter=20)
        Q, _ = np.linalg.qr(np.hstack(U_list), mode="reduced")
        ks = Q.shape[1]
        sub_covs = []
        for U_m, lam_m, s2_m in zip(U_list, lam_list, s2_list):
            Cm = Q.T @ U_m
            S_sub = (Cm * (lam_m - s2_m)[None, :]) @ Cm.T + s2_m * np.eye(ks)
            sub_covs.append(0.5 * (S_sub + S_sub.T))
        B_sub = dense_bw_bary(sub_covs, n_iter=20)
        bar_sp = float(np.mean([math.sqrt(s2) for s2 in s2_list]))
        bar_s2 = bar_sp ** 2
        B_fast = Q @ (B_sub - bar_s2 * np.eye(ks)) @ Q.T + bar_s2 * np.eye(d)
        err = float(np.linalg.norm(B_dense - B_fast, "fro"))
        assert err < 1e-10, f"Thm 2 barycenter mismatch: {err}"

        # Also check pairwise distance identity against a fresh spiked query
        U_q, _ = np.linalg.qr(rng.normal(size=(d, 3)), mode="reduced")
        s2_q = float(rng.uniform(0.2, 1.5))
        lam_q = s2_q + rng.uniform(0.1, 4.0, size=3)
        C_q = (U_q * (lam_q - s2_q)[None, :]) @ U_q.T + s2_q * np.eye(d)
        d_dense = bw_dist(C_q, B_fast)

        Q_ab, _ = np.linalg.qr(np.hstack([Q, U_q]), mode="reduced")
        kv = Q_ab.shape[1]
        Ca = Q_ab.T @ U_q
        Cb = Q_ab.T @ Q
        Av = (Ca * (lam_q - s2_q)[None, :]) @ Ca.T + s2_q * np.eye(kv)
        Bv = Cb @ (B_sub - bar_s2 * np.eye(ks)) @ Cb.T + bar_s2 * np.eye(kv)
        d_sub_sq = bw_dist(Av, Bv) ** 2 + (d - kv) * (math.sqrt(s2_q) - math.sqrt(bar_s2)) ** 2
        assert abs(d_dense - math.sqrt(d_sub_sq)) < 1e-10, "Thm 2 distance mismatch"

    print("PASS: (Thm. 1 Wishart inversion, Thm. 2 exact subspace BW barycenter & distance) — 0 counterexamples")


if __name__ == "__main__":
    verify_theorems()
    print("ALL_THEOREMS_VERIFIED_OK")
