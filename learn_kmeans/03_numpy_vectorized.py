"""第 3 步：改成 numpy 向量化。逻辑一模一样，速度快几十倍。

关键点有两个：
  1. 用矩阵化距离代替双重 for 循环；
  2. 用距离展开式 ||x-c||^2 = ||x||^2 - 2x·c + ||c||^2，避免生成 (n,k,d) 三维张量。

运行： python 03_numpy_vectorized.py
"""

import importlib.util
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pure = load_module("pure", "02_pure_python.py")


# --------------------------------------------------------------------------
# 距离矩阵的两种写法
# --------------------------------------------------------------------------
def dist_naive(X, C):
    """直观写法：广播成 (n, k, d)。n 大或 d 大时直接爆内存。"""
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)


def dist_fast(X, C, x_sq=None):
    """展开式写法：先算 (n, k)，再按需 sqrt。"""
    if x_sq is None:
        x_sq = (X ** 2).sum(axis=1)
    return x_sq[:, None] - 2.0 * X @ C.T + (C ** 2).sum(axis=1)[None, :]


# --------------------------------------------------------------------------
# numpy 版 k-means
# --------------------------------------------------------------------------
def kmeans_np(X, k, max_iter=300, tol=1e-9, seed=0, init=None, naive=False):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    rng = np.random.default_rng(seed)

    C = (X[rng.choice(n, size=k, replace=False)].copy() if init is None
         else np.asarray(init, dtype=float).copy())
    x_sq = (X ** 2).sum(axis=1)

    labels = np.zeros(n, dtype=int)
    inertia = np.inf

    for _ in range(max_iter):
        # ---- E 步 ----
        d2 = dist_naive(X, C) if naive else dist_fast(X, C, x_sq)
        labels = d2.argmin(axis=1)

        # ---- M 步 ----
        new_C = np.stack([X[labels == j].mean(axis=0) if (labels == j).any() else C[j]
                          for j in range(k)])

        inertia = float(((X - new_C[labels]) ** 2).sum())
        shift = float(((new_C - C) ** 2).sum())
        C = new_C
        if shift <= tol:
            break

    return labels, C, inertia


if __name__ == "__main__":
    # ---- 1) 与纯 Python 版结果对齐 ----
    X1 = np.array([[1], [2], [3], [4], [10], [11], [12], [13]], dtype=float)
    init1 = X1[:2]
    lab_np, C_np, J_np = kmeans_np(X1, 2, init=init1)
    lab_py, C_py, J_py, _ = pure.kmeans(X1.tolist(), 2, init=init1.tolist())

    print("【结果对齐】")
    print(f"  numpy   : 中心 {C_np.ravel().round(3).tolist()}, J = {J_np:.4f}")
    print(f"  pure py : 中心 {[round(c[0], 3) for c in C_py]}, J = {J_py:.4f}")
    assert np.allclose(C_np.ravel(), [c[0] for c in C_py])

    # ---- 2) 两种距离写法等价，但内存/速度差很多 ----
    rng = np.random.default_rng(0)
    n, d, k = 20000, 32, 8
    Xb = rng.normal(size=(n, d)) + np.repeat(rng.normal(scale=6, size=(k, d)), n // k, axis=0)
    Cb = Xb[rng.choice(n, k, replace=False)]

    t0 = time.perf_counter(); dist_fast(Xb, Cb); t_fast = time.perf_counter() - t0
    t0 = time.perf_counter(); dist_naive(Xb, Cb); t_naive = time.perf_counter() - t0
    print("\n【距离矩阵：n=20000, k=8, d=32】")
    print(f"  展开式 (n,k)  : {t_fast * 1000:7.1f} ms")
    print(f"  广播 (n,k,d)  : {t_naive * 1000:7.1f} ms   "
          f"({t_naive / t_fast:.1f}x 慢，且中间张量占 "
          f"{n * k * d * 8 / 1e6:.0f} MB)")

    # ---- 3) 整体速度：numpy vs 纯 Python ----
    n2, d2, k2 = 4000, 10, 6
    Xs = rng.normal(size=(n2, d2))
    t0 = time.perf_counter(); kmeans_np(Xs, k2, max_iter=20, seed=1); t_np = time.perf_counter() - t0
    t0 = time.perf_counter(); pure.kmeans(Xs.tolist(), k2, max_iter=20); t_py = time.perf_counter() - t0
    print(f"\n【整体耗时：n={n2}, k={k2}, d={d2}, 最多 20 轮】")
    print(f"  numpy      : {t_np:6.2f} s")
    print(f"  纯 Python  : {t_py:6.2f} s   ({t_py / t_np:.0f}x 慢)")
