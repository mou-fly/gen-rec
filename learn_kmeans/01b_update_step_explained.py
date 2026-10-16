"""01b：把 M 步那一行拆开看 —— 每一句到底在做什么。

配套 01_hand_calculation.py。

运行： python 01b_update_step_explained.py
"""

X = [1, 2, 3, 4, 10, 11, 12, 13]
labels = [0, 1, 1, 1, 1, 1, 1, 1]      # 第 1 轮 E 步的结果
k = 2


# --------------------------------------------------------------------------
# 原始写法（一行 list comprehension）
# --------------------------------------------------------------------------
def update(X, labels, k):
    new_centers = []
    for j in range(k):
        pts = [x for x, l in zip(X, labels) if l == j]
        new_centers.append(sum(pts) / len(pts) if pts else None)
    return new_centers


# --------------------------------------------------------------------------
# 等价写法（展开成最笨的循环，方便对照）
# --------------------------------------------------------------------------
def update_verbose(X, labels, k):
    new_centers = []
    for j in range(k):
        print(f"  处理簇 {j}：")
        pts = []
        for i in range(len(X)):
            print(f"    点 X[{i}]={X[i]:<3} 的标签是 {labels[i]}  "
                  f"→ {'属于' if labels[i] == j else '不属于'}簇 {j}")
            if labels[i] == j:
                pts.append(X[i])
        print(f"    收集到 pts = {pts}")
        if pts:
            s, n = sum(pts), len(pts)
            print(f"    求和 sum(pts) = {s}，个数 len(pts) = {n}")
            print(f"    新中心 = {s} / {n} = {s / n}")
            new_centers.append(s / n)
        else:
            print("    没有点 → 返回 None（危险！见文末）")
            new_centers.append(None)
    return new_centers


print("=" * 62)
print("【第 1 轮】输入：labels =", labels)
print("=" * 62)
c1 = update_verbose(X, labels, k)
print("  → 原始写法结果:", update(X, labels, k), "（必须和上面一致）")
assert c1 == update(X, labels, k)

print()
print("=" * 62)
print("【第 2 轮】E 步重算 labels，再跑一次 M 步")
print("=" * 62)
# 用第 1 轮的新中心重新分配
new_labels = [min(range(k), key=lambda j: abs(x - c1[j])) for x in X]
print("  E 步得到 labels =", new_labels)
c2 = update_verbose(X, new_labels, k)
print("  → 新中心 =", c2)

print()
print("=" * 62)
print("【为什么 M 步用均值？在数轴上扫一遍看谁让 J 最小】")
print("=" * 62)
pts = [2, 3, 4, 10, 11, 12, 13]
sq = lambda c: sum((x - c) ** 2 for x in pts)     # k-means 的目标（平方误差）
ab = lambda c: sum(abs(x - c) for x in pts)       # 绝对值误差（不是 k-means 用的）
grid = [i / 1000 for i in range(0, 15001)]

best_sq = min(grid, key=sq)
best_ab = min(grid, key=ab)
srt = sorted(pts)
median = srt[len(srt) // 2]                        # 7 个点，取中间那个
print(f"  簇内点 = {pts}")
print(f"  让「平方误差和」最小的点 = {best_sq:.3f}   ← 正好是均值 "
      f"sum/len = {sum(pts) / len(pts):.3f}")
print(f"  让「绝对误差和」最小的点 = {best_ab:.3f}   ← 正好是中位数 {median}，"
      f"但 k-means 不用它")
print(f"  在均值处 J = {sq(best_sq):.3f}；若把中心硬设为中位数 {median}，"
      f"J = {sq(median):.3f}（更大，所以 M 步必须用均值）")
print("  → J = Σ(x-μ)² 对 μ 求导置零：-2Σ(x-μ)=0 → μ = mean。这就是 M 步的全部道理")
print("  → 二阶导 = 2n > 0，所以是极小值，不是极大值")

print()
print("=" * 62)
print("【危险：空簇返回 None 会怎样】")
print("=" * 62)
bad = update([1, 2, 3], [0, 0, 0], 2)      # 簇 1 一个点都没有
print("  update 返回:", bad)
try:
    [abs(x - c) for x in [1, 2, 3] for c in bad]     # 下一轮 E 步要拿中心算距离
except TypeError as e:
    print(f"  下一轮 E 步立刻爆炸：TypeError: {e}")
print("  → 所以第 4 步要处理空簇（把最远的点抢过来当新中心），见 04_kmeans_plusplus.py")
