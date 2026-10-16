"""第 4 步：加固 —— k-means++ 初始化、多次重启、空簇处理。

前三步的版本能跑，但会踩三个坑：
  坑 1：随机初始化可能极差 → k-means++ + n_init
  坑 2：某个簇一个点都没分到 → mean 出 nan → 抢最远点
  坑 3：只看一次结果就交差 → 取 inertia 最小的那次

运行： python 04_kmeans_plusplus.py
"""

import numpy as np


# --------------------------------------------------------------------------
# k-means++ 初始化
# --------------------------------------------------------------------------
def kmeans_plusplus(X, k, rng):
    n = X.shape[0]
    centers = [X[rng.integers(n)]]                       # 第一个中心随便选
    d2 = ((X - centers[0]) ** 2).sum(axis=1)             # 到最近中心的距离平方
    for _ in range(k - 1):
        probs = d2 / d2.sum()                            # 越远越可能被选中
        centers.append(X[rng.choice(n, p=probs)])
        d2 = np.minimum(d2, ((X - centers[-1]) ** 2).sum(axis=1))   # 增量更新
    return np.asarray(centers, dtype=float)


# --------------------------------------------------------------------------
# 完整版
# --------------------------------------------------------------------------
def kmeans(X, k, n_init=10, max_iter=300, tol=1e-8, seed=0,
           init="kmeans++", return_history=False, verbose=False):
    """init: "kmeans++" | "random" | 显式指定的初始中心数组"""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if k > n:
        raise ValueError("k 不能大于样本数")

    rng = np.random.default_rng(seed)
    x_sq = (X ** 2).sum(axis=1)

    fixed_init = None
    if not isinstance(init, str):
        fixed_init = np.asarray(init, dtype=float)
        n_init = 1

    best_inertia, best_labels, best_C, best_hist = np.inf, None, None, None

    for run in range(n_init):
        # ---- 初始化 ----
        if fixed_init is not None:
            C = fixed_init.copy()
        elif init == "kmeans++":
            C = kmeans_plusplus(X, k, rng)
        elif init == "random":
            C = X[rng.choice(n, size=k, replace=False)].copy()
        else:
            raise ValueError(f"未知 init: {init}")

        history = []
        inertia = np.inf

        for it in range(max_iter):
            # ---------- E 步：分配 ----------
            d2 = x_sq[:, None] - 2.0 * X @ C.T + (C ** 2).sum(axis=1)[None, :]
            labels = d2.argmin(axis=1)
            history.append((C.copy(), labels.copy()))

            # ---------- 处理空簇：把最远的点抢过来 ----------
            counts = np.bincount(labels, minlength=k)
            if (counts == 0).any():
                excluded = np.zeros(n, dtype=bool)
                for j in np.flatnonzero(counts == 0):
                    cand = np.where(excluded, -1.0, d2[np.arange(n), labels])
                    far = int(cand.argmax())
                    excluded[far] = True
                    labels[far] = j
                if verbose:
                    print(f"    [警告] 第 {it} 轮出现空簇，已用最远点补位")

            # ---------- M 步：更新中心 ----------
            new_C = C.copy()
            for j in range(k):
                mask = labels == j
                if mask.any():
                    new_C[j] = X[mask].mean(axis=0)

            inertia = float(((X - new_C[labels]) ** 2).sum())
            shift = float(((new_C - C) ** 2).sum())
            C = new_C
            if verbose:
                print(f"  run={run} iter={it:3d} inertia={inertia:.6f}")
            if shift <= tol:
                break

        if inertia < best_inertia:
            best_inertia, best_labels, best_C, best_hist = inertia, labels, C, history

    if return_history:
        return best_labels, best_C, best_inertia, best_hist
    return best_labels, best_C, best_inertia


# --------------------------------------------------------------------------
# 演示
# --------------------------------------------------------------------------
def make_data(seed=0, n_per=120):
    rng = np.random.default_rng(seed)
    true_c = np.array([[0.0, 0.0], [6.0, 5.0], [0.0, 7.0], [7.0, 0.0]])
    return np.vstack([rng.normal(c, 0.75, size=(n_per, 2)) for c in true_c]), true_c


if __name__ == "__main__":
    X, true_c = make_data()

    # ---- 演示 1：随机初始化的方差有多大 ----
    print("【演示 1：初始化方式对比（k=4, 每簇 120 点）】")
    rand = [kmeans(X, 4, n_init=1, init="random", seed=s)[2] for s in range(20)]
    kpp = [kmeans(X, 4, n_init=1, init="kmeans++", seed=s)[2] for s in range(20)]
    print(f"  纯随机初始化 20 次 : J 最小 {min(rand):9.2f} | 最大 {max(rand):9.2f} | "
          f"均值 {np.mean(rand):9.2f} | 标准差 {np.std(rand):8.2f}")
    print(f"  k-means++ 20 次   : J 最小 {min(kpp):9.2f} | 最大 {max(kpp):9.2f} | "
          f"均值 {np.mean(kpp):9.2f} | 标准差 {np.std(kpp):8.2f}")
    bad_rate = np.mean([j > min(rand) * 1.5 for j in rand])
    print(f"  → 随机初始化有 {bad_rate * 100:.0f}% 的概率掉进明显更差的局部最优")

    # 稳稳拿到好解的办法
    _, C, J = kmeans(X, 4, n_init=10, init="kmeans++", seed=0)
    print(f"\n  k-means++ + n_init=10 最终 J = {J:.2f}")

    # ---- 演示 2：空簇不再产生 nan ----
    print("\n【演示 2：空簇处理】")
    dup_init = np.array([X[0], X[0], X[120], X[240]])   # 前两个中心完全重合
    labels, C, J = kmeans(X, 4, init=dup_init, verbose=True)
    print(f"  结果 J = {J:.2f}，中心数 = {len(C)}，labels 是否有 nan: "
          f"{np.isnan(labels.astype(float)).any()}")
    print(f"  各簇样本数 = {np.bincount(labels, minlength=4).tolist()}")

    # ---- 演示 3：n_init 越大越稳 ----
    print("\n【演示 3：n_init 的影响（各跑 20 个 seed，看最差 J）】")
    for n_init in (1, 3, 10):
        worst = max(kmeans(X, 4, n_init=n_init, init="kmeans++", seed=s)[2]
                    for s in range(20))
        print(f"  n_init={n_init:2d} → 20 次里最差的 J = {worst:9.2f}")
