"""01d：inertia 这个词从哪来，以及它的三个坑。

运行： python 01d_what_is_inertia.py
"""

import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("km04", HERE / "04_kmeans_plusplus.py")
km = importlib.util.module_from_spec(spec)
spec.loader.exec_module(km)

print("=" * 68)
print("① inertia 的定义：每个点到【它所属中心】的距离平方和")
print("=" * 68)

X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])


def inertia_of(X, labels, centers):
    """手动实现：不用 sklearn，也不用 04 里的函数，看看它到底是什么。"""
    total = 0.0
    for x, l in zip(X, labels):
        total += float(((x - centers[l]) ** 2).sum())   # 到【自己那个】中心的距离平方
    return total


# 划分 A：分对了
labels_good = [0, 0, 0, 1, 1, 1]
centers_good = np.array([[2.0], [11.0]])
# 划分 B：分歪了
labels_bad = [0, 0, 1, 1, 1, 1]
centers_bad = np.array([[1.5], [9.0]])

print(f"  划分 A（对）: labels={labels_good}, 中心={centers_good.ravel().tolist()}"
      f"  → inertia = {inertia_of(X, labels_good, centers_good):.2f}")
print(f"  划分 B（歪）: labels={labels_bad}, 中心={centers_bad.ravel().tolist()}"
      f"  → inertia = {inertia_of(X, labels_bad, centers_bad):.2f}")
print("  → inertia 越小说明簇内越紧凑，所以 k-means 就是在最小化它")

print()
print("=" * 68)
print("② 它有一堆同义名字（面试/论文里都会遇到）")
print("=" * 68)
for abbr, full in [
    ("inertia", "sklearn 的属性名，KMeans.inertia_"),
    ("SSE", "Sum of Squared Errors，误差平方和"),
    ("WCSS", "Within-Cluster Sum of Squares，簇内平方和"),
    ("distortion", "有些教材叫「失真度」"),
    ("J", "数学记法，目标函数"),
    ("簇内平方和", "中文教材最常见的说法"),
]:
    print(f"  {abbr:<12} {full}")
print("  → 它们指的是同一个东西：Σ ‖x − μ_所属簇‖²")

print()
print("=" * 68)
print("③ 坑一：inertia 永远随 k 增大而单调下降，不能直接拿来选 k")
print("=" * 68)

rng = np.random.default_rng(3)
Xb = np.vstack([rng.normal(c, 0.7, size=(40, 2))
                for c in ([0, 0], [6, 6], [12, 0])])

print("   k    inertia")
for k in range(1, 10):
    J = km.kmeans(Xb, k, n_init=5, seed=0)[2]
    bar = "#" * max(1, int(40 * J / 900))
    print(f"  {k:2d}   {J:9.2f}  {bar}")
J_full = km.kmeans(Xb, len(Xb), n_init=1, seed=0)[2]
print(f"  k=n={len(Xb)} → inertia = {J_full:.2f}"
      f"   （每个点自成一簇，距离必然为 0）")
print("  → 所以 k 越大 inertia 一定越小，最小值出现在 k=n，毫无意义")
print("  → 只能用「肘部法看拐点」或「轮廓系数」，不能直接取最小值")

print()
print("=" * 68)
print("④ 坑二：inertia 的量纲是「距离的平方」，对特征尺度极其敏感")
print("=" * 68)

X_small = Xb.copy()
X_big = Xb * 1000.0                       # 同一个数据集，只是把单位从「米」换成「毫米」
J1 = km.kmeans(X_small, 3, n_init=5, seed=0)[2]
J2 = km.kmeans(X_big, 3, n_init=5, seed=0)[2]
print(f"  原始尺度      : inertia = {J1:,.2f}")
print(f"  整体 ×1000    : inertia = {J2:,.2f}")
print(f"  倍数          : {J2 / J1:,.0f}   （= 1000²，因为 inertia 是平方量纲）")
print("  → 所以不同数据集、不同单位之间的 inertia 完全不可比")
print("  → 这也是「聚类前必须先标准化」的原因（StandardScaler）")

print()
print("=" * 68)
print("⑤ 坑三：inertia 只能用来「同 k 下比不同初始化」，这正是 n_init 的依据")
print("=" * 68)

vals = [km.kmeans(Xb, 3, n_init=1, init="random", seed=s)[2] for s in range(15)]
print(f"  k 固定为 3，15 次随机初始化: 最好 {min(vals):.2f} / 最差 {max(vals):.2f}")
print("  → 同一个 k，inertia 越小说明这次初始化越成功，取最小的那次即可")
