"""01c：pts 到底是什么？—— 把它的生命周期打印出来。

配套 01_hand_calculation.py 和 01b_update_step_explained.py。

运行： python 01c_what_is_pts.py
"""

print("=" * 66)
print("场景 A：一维数据（第 1 步的数轴例子）d=1")
print("=" * 66)

X_1d = [1, 2, 3, 4, 10, 11, 12, 13]        # 8 个点，每个点只有 1 个数
labels = [0, 0, 0, 0, 1, 1, 1, 1]          # 每个点的簇归属
k = 2

for j in range(k):
    pts = [x for x, l in zip(X_1d, labels) if l == j]
    print(f"j = {j}:  pts = {pts}")
    print(f"         类型是 list，长度 {len(pts)}，里面装的是【标量】")
    print(f"         均值 = sum(pts)/len(pts) = {sum(pts)}/{len(pts)} = "
          f"{sum(pts) / len(pts)}")
print("→ 注意：每循环一次，pts 这个名字就被重新赋值一次，旧内容被丢掉")

print()
print("=" * 66)
print("场景 B：二维数据 d=2（每个点是一个坐标对）")
print("=" * 66)

X_2d = [[1, 1], [1.5, 1.2], [8, 8], [8.4, 7.8]]   # 4 个点，每个点 2 个数
labels_2d = [0, 0, 1, 1]

for j in range(2):
    pts = [x for x, l in zip(X_2d, labels_2d) if l == j]
    print(f"j = {j}:  pts = {pts}")
    print(f"         里面装的是【向量】，每个点 2 个数")
    # 逐维求均值：第 t 维 = 所有点的第 t 维之和 / 点数
    d = len(pts[0])
    center = [sum(p[t] for p in pts) / len(pts) for t in range(d)]
    print(f"         逐维求均值 → 新中心 = {center}")
print("→ 所以 d 维数据不能写 sum(pts)/len(pts)，必须逐维求和（02_pure_python.py:34）")

print()
print("=" * 66)
print("关键不变量：各个簇的 pts 拼起来，正好是全部 n 个点")
print("=" * 66)

all_pts = []
for j in range(k):
    all_pts += [x for x, l in zip(X_1d, labels) if l == j]
print(f"  X                    = {X_1d}")
print(f"  各簇 pts 拼起来       = {all_pts}")
print(f"  完全相等？            = {all_pts == X_1d}")
print(f"  长度之和 = {sum(len([x for x, l in zip(X_1d, labels) if l == j]) for j in range(k))}"
      f" = n = {len(X_1d)}")
print("→ 每个点只被数到一次，所以 M 步总共只遍历 n 个点 → 复杂度 O(nd)，跟 k 无关")
print("→ 而 E 步是「每个点 × 每个中心」，所以才冒出 k → O(nkd)")

print()
print("=" * 66
      )
print("命名对照表")
print("=" * 66)
for name, meaning in [
    ("X", "全部数据，n 个点"),
    ("labels", "每个点的簇编号，长度 n"),
    ("j", "当前正在处理的簇编号，取值 0..k-1"),
    ("pts", "簇 j 内的点，临时变量，长度 = 该簇点数"),
    ("C / centers", "k 个中心"),
    ("d", "每个点的维度"),
]:
    print(f"  {name:<12} {meaning}")
