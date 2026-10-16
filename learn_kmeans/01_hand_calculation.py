"""第 1 步：先别写算法，用纸笔把 k-means 走一遍。

数据：数轴上的 8 个点，肉眼一看就知道该分成 {1,2,3,4} 和 {10,11,12,13}。
但我们故意把两个初始中心放在 1 和 2 —— 挤在左边簇里，看算法怎么爬出来。

运行： python 01_hand_calculation.py
"""

X = [1, 2, 3, 4, 10, 11, 12, 13]
K = 2
centers = [1.0, 2.0]        # 很烂的初始化，故意的


def assign(X, centers):
    """E 步：每个点分给最近的中心，返回每个点的簇编号。"""
    labels = []
    for x in X:
        dists = [abs(x - c) for c in centers]
        labels.append(dists.index(min(dists)))
    return labels


def update(X, labels, k):
    """M 步：新中心 = 该簇所有点的均值。"""
    new_centers = []
    for j in range(k):
        pts = [x for x, l in zip(X, labels) if l == j]
        new_centers.append(sum(pts) / len(pts) if pts else None)
    return new_centers


def inertia(X, labels, centers):
    """目标函数 J = 每个点到所属中心的距离平方和。"""
    return sum((x - centers[l]) ** 2 for x, l in zip(X, labels))


def fmt(vals):
    return "[" + ", ".join(f"{v:g}" for v in vals) + "]"


print(f"数据      X = {fmt(X)}")
print(f"初始中心  c = {fmt(centers)}   <- 两个中心都挤在左簇里\n")

for it in range(1, 11):
    labels = assign(X, centers)
    groups = [[x for x, l in zip(X, labels) if l == j] for j in range(K)]
    j_now = inertia(X, labels, centers)

    print(f"第 {it} 轮")
    print("  [E 步] 按最近中心分配：")
    for j, g in enumerate(groups):
        print(f"         簇{j + 1} = {fmt(g):<22} (当前中心 {centers[j]:g})")
    print(f"         此刻 J = {j_now:.4f}")

    new_centers = update(X, labels, K)
    j_new = inertia(X, labels, new_centers)
    print(f"  [M 步] 新中心 = 各簇均值 → {fmt(new_centers)}，J = {j_new:.4f}")

    if new_centers == centers:
        print("\n  中心不再变化 → 收敛，结束。")
        break

    centers = new_centers
    print()

print(f"\n最终中心   = {fmt(centers)}")
print(f"最终划分   = {fmt(labels)}   (0 表示第 1 簇，1 表示第 2 簇)")
print(f"最终 J     = {inertia(X, labels, centers):.4f}")
