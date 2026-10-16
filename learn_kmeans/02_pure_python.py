"""第 2 步：把第 1 步的手算流程写成通用函数 —— 仍然完全不用 numpy。

支持任意维度、任意 k，这才是能交差的「手撕」版本（白板题基本就考这个）。

运行： python 02_pure_python.py
"""


def dist2(a, b):
    """两点欧氏距离的平方（省掉 sqrt，比较大小用不着）。"""
    return sum((ai - bi) ** 2 for ai, bi in zip(a, b))


def kmeans(X, k, max_iter=100, tol=1e-9, init=None):
    """X: [[x1..xd], ...]，返回 (labels, centers, inertia, iters)。"""
    n, d = len(X), len(X[0])
    if k > n:
        raise ValueError("k 不能大于样本数")

    # ---- 初始化：默认取前 k 个样本（烂但诚实，第 4 步再换成 k-means++）----
    C = [list(map(float, p)) for p in (X[:k] if init is None else init)]

    labels, inertia, iters = [0] * n, float("inf"), 0

    for iters in range(1, max_iter + 1):
        # ---- E 步：每个点找最近的中心 ----
        labels = [min(range(k), key=lambda j: dist2(X[i], C[j])) for i in range(n)]

        # ---- M 步：每个簇取均值当新中心 ----
        new_C = []
        for j in range(k):
            pts = [X[i] for i in range(n) if labels[i] == j]
            if pts:
                new_C.append([sum(p[t] for p in pts) / len(pts) for t in range(d)])
            else:
                new_C.append(C[j])          # 空簇：先原地不动（第 4 步再优化）

        inertia = sum(dist2(X[i], new_C[labels[i]]) for i in range(n))

        # ---- 收敛判定：中心移动量足够小就停 ----
        shift = max(dist2(a, b) for a, b in zip(new_C, C))
        C = new_C
        if shift <= tol:
            break

    return labels, C, inertia, iters


def farthest_point_init(X, k):
    """确定性的分散初始化：第一个中心取第 0 个点，之后每次选「离已有中心最远」的点。
    纯 Python 几行就能写，比随机初始化稳得多（k-means++ 的简化版）。"""
    centers = [X[0]]
    while len(centers) < k:
        nxt = max(X, key=lambda p: min(dist2(p, c) for c in centers))
        centers.append(nxt)
    return centers


if __name__ == "__main__":
    # 1) 用第 1 步的同一份数据，结果必须完全一致
    X1 = [[1], [2], [3], [4], [10], [11], [12], [13]]
    labels, C, J, iters = kmeans(X1, k=2)          # 默认初始化恰好就是 [1, 2]
    print("【复现第 1 步】")
    print(f"  迭代 {iters} 轮后收敛")
    print(f"  中心   = {[round(c[0], 3) for c in C]}")
    print(f"  划分   = {labels}")
    print(f"  inertia= {J:.4f}")
    assert [round(c[0], 3) for c in C] == [2.5, 11.5], "与手算结果不一致！"

    # 2) 换成二维、4 个簇，证明它是通用的
    import random

    random.seed(0)
    true_centers = [(0, 0), (6, 6), (12, 0), (0, 12)]
    X2 = [[c[0] + random.gauss(0, 0.6), c[1] + random.gauss(0, 0.6)]
          for c in true_centers for _ in range(60)]

    # 2a) 默认初始化（取前 k 个点）—— 运气不好就会翻车
    labels_n, Cn, Jn, itn = kmeans(X2, k=4, max_iter=200)
    print("\n【二维 4 簇 · 默认初始化 X[:k]】")
    print(f"  迭代 {itn} 轮，inertia = {Jn:.2f}   <- 明显偏大，簇被合并了")
    print(f"  各簇样本数: {[labels_n.count(j) for j in range(4)]}  (每簇应有 60)")

    # 2b) 换成最远点初始化 —— 稳稳还原
    labels2, C2, J2, iters2 = kmeans(X2, k=4, max_iter=200,
                                    init=farthest_point_init(X2, 4))
    print("\n【二维 4 簇 · 最远点初始化】")
    print(f"  迭代 {iters2} 轮，inertia = {J2:.2f}")
    print("  学到的中心（排序后）:")
    for c in sorted(C2, key=lambda p: (round(p[0]), round(p[1]))):
        print(f"    ({c[0]:7.3f}, {c[1]:7.3f})")
    print("  真实中心:", true_centers)
    print(f"  各簇样本数: {[labels2.count(j) for j in range(4)]}  (每簇应有 60)")
