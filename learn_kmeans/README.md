# 手撕 k-means · 五步渐进讲义

> 配套代码全部可直接运行，每一步都能看到输出。推荐**按顺序**跑，别跳。

## 0. 一句话版

k-means = 反复做两件事：**把每个点分给最近的中心（E 步）** → **把中心挪到簇的均值（M 步）**，直到中心不动。

```
随机/++初始化 k 个中心
repeat:
    labels = 每个点找最近的中心        # E 步，O(nkd)
    centers = 每个簇的均值            # M 步，O(nd)
until 中心不再移动 或 达到 max_iter
```

## 30 分钟学习路线

| 时间 | 做什么 | 文件 |
|---|---|---|
| 0–5 min | 手算一遍，理解两个步骤在算什么 | `01_hand_calculation.py` |
| 5–12 min | 自己默写通用版（不用 numpy），对答案 | `02_pure_python.py` |
| 12–18 min | 换成 numpy 向量化，理解距离展开式 | `03_numpy_vectorized.py` |
| 18–25 min | 加 k-means++、n_init、空簇处理 | `04_kmeans_plusplus.py` |
| 25–30 min | 看图复盘，把过程讲给自己听 | `05_visualize.py` → `kmeans_iterations.png` |

运行方式（在 `learn_kmeans/` 目录下）：

```powershell
python 01_hand_calculation.py
python 02_pure_python.py
python 03_numpy_vectorized.py
python 04_kmeans_plusplus.py
python 05_visualize.py
```

---

## 第 1 步：先手算，别写代码

数据：数轴上 8 个点 `[1,2,3,4,10,11,12,13]`，`k=2`，初始中心故意给成 `[1, 2]`（两个都挤在左簇里）。

关键结论：**J 单调不增**，这是算法一定收敛的原因。

| 轮次 | E 步分配 | 分配后 J | M 步新中心 | 更新后 J |
|---|---|---|---|---|
| 1 | `[1]` \| `[2..13]` | 371.00 | `[1, 7.857]` | 130.86 |
| 2 | `[1,2,3,4]` \| `[10..13]` | 72.08 | `[2.5, 11.5]` | 10.00 |
| 3 | 同上 | 10.00 | `[2.5, 11.5]` | 10.00 → 收敛 |

要理解的三件事：

1. 初始中心很烂也没关系，**均值会把它拉回簇心**；
2. 每一步 J 都在降：`371 → 130.86 → 72.08 → 10`；
3. 「中心不再变化」= 收敛条件。

## 第 2 步：纯 Python 通用版（白板就能写）

四个函数，20 多行：

```python
def dist2(a, b):                       # 距离平方，省掉 sqrt
    return sum((ai - bi) ** 2 for ai, bi in zip(a, b))

def kmeans(X, k, max_iter=100, tol=1e-9, init=None):
    n, d = len(X), len(X[0])
    C = [list(map(float, p)) for p in (X[:k] if init is None else init)]

    for _ in range(max_iter):
        # E 步：每个点分给最近的中心
        labels = [min(range(k), key=lambda j: dist2(X[i], C[j])) for i in range(n)]

        # M 步：每个簇的新中心 = 簇内均值
        new_C = []
        for j in range(k):
            pts = [X[i] for i in range(n) if labels[i] == j]
            if pts:
                new_C.append([sum(p[t] for p in pts) / len(pts) for t in range(d)])
            else:
                new_C.append(C[j])                  # 空簇先原地不动

        # 判停：中心移动量足够小
        if max(dist2(a, b) for a, b in zip(new_C, C)) <= tol:
            break
        C = new_C

    return labels, C, new_C
```

**这一步最重要的收获**：初始化方式决定成败。同一份二维 4 簇数据：

| 初始化 | 迭代轮数 | inertia | 各簇样本数（应各 60） |
|---|---|---|---|
| `X[:k]`（取前 k 个点） | 7 | **2206.16** | `[60, 120, 30, 30]` ← 簇被合并 |
| 最远点初始化 | 2 | **179.00** | `[60, 60, 60, 60]` ✅ |

J 差了 12 倍。最远点初始化只要几行：

```python
def farthest_point_init(X, k):
    centers = [X[0]]
    while len(centers) < k:
        centers.append(max(X, key=lambda p: min(dist2(p, c) for c in centers)))
    return centers
```

## 第 3 步：numpy 向量化（工程版必写）

两个关键改动：

```python
# 1) E 步一次算完所有点到所有中心的距离：不要写 (X[:,None]-C[None])**2（会生成 n*k*d 张量）
x_sq = (X ** 2).sum(axis=1)
d2 = x_sq[:, None] - 2.0 * X @ C.T + (C ** 2).sum(axis=1)[None, :]
labels = d2.argmin(axis=1)

# 2) M 步用布尔索引
new_C = np.stack([X[labels == j].mean(axis=0) for j in range(k)])
```

实测收益（本机，n=20000 / k=8 / d=32）：

| 写法 | 耗时 | 中间张量 |
|---|---|---|
| 广播 `(n,k,d)` | 21.4 ms | 41 MB |
| 展开式 `(n,k)` | **5.1 ms** | 1.3 MB |

整体（n=4000, k=6, d=10, 最多 20 轮）：numpy **0.01 s** vs 纯 Python **0.66 s** → **约 79 倍**。

## 第 4 步：加固三个坑

### 坑 1：初始化差 → k-means++ + n_init

```python
def kmeans_plusplus(X, k, rng):
    centers = [X[rng.integers(len(X))]]
    d2 = ((X - centers[0]) ** 2).sum(axis=1)
    for _ in range(k - 1):
        centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])   # 越远越可能被选中
        d2 = np.minimum(d2, ((X - centers[-1]) ** 2).sum(axis=1))
    return np.asarray(centers)
```

实测（k=4，每簇 120 点，各跑 20 个 seed）：

| 初始化 | J 最小 | J 最大 | J 均值 | 标准差 |
|---|---|---|---|---|
| 纯随机 | 504.24 | 3406.26 | 1195.88 | 1101.75 |
| k-means++ | 504.24 | 2066.97 | 660.51 | 468.81 |

随机初始化有 **30%** 的概率掉进明显更差的局部最优；k-means++ 把方差砍掉一半，但仍不保险，所以还要 `n_init`：

| n_init | 20 个 seed 里最差的 J |
|---|---|
| 1 | 2066.97 |
| 3 | **504.24** |
| 10 | **504.24** |

### 坑 2：空簇 → 抢最远的点

某个中心一个点都没分到，`mean` 会变 `nan`。处理办法：把离自己中心最远的那个点抢过来当新中心。演示中故意给两个完全重合的初始中心，立刻触发：

```
[警告] 第 0 轮出现空簇，已用最远点补位
iter=0 inertia=2121.23  →  iter=4 inertia=504.24   # 最终照样收敛到最优
```

### 坑 3：收敛判定

用**中心位移** `((new_C - C) ** 2).sum() <= tol` 或**目标函数下降量** `prev_inertia - inertia <= tol`，别用浮点相等。

## 第 5 步：看图复盘

`05_visualize.py` 生成 `kmeans_iterations.png`：上排看一轮怎么走，下排看选 k 和初始化的影响。

**能对着这张图把 E 步、M 步、收敛、n_init 讲一遍，手撕就过关了。**

---

## 白板默写口诀

> 初始化 → **分配** → **均值** → **判停** → 取最好的一次

代码骨架（面试直接写这个）：

```python
def kmeans(X, k, n_init=10, max_iter=300, tol=1e-8):
    best = (np.inf, None, None)
    for _ in range(n_init):                          # ① 多次重启
        C = kmeans_plusplus(X, k)                    # ② k-means++ 初始化
        for _ in range(max_iter):
            labels = dist2(X, C).argmin(1)           # ③ E 步
            newC = np.stack([X[labels == j].mean(0)  # ④ M 步
                             for j in range(k)])
            if ((newC - C) ** 2).sum() <= tol:       # ⑤ 判停
                C = newC; break
            C = newC
        J = ((X - C[labels]) ** 2).sum()
        if J < best[0]:                              # ⑥ 留最好的
            best = (J, labels, C)
    return best[1], best[2], best[0]
```

## 复杂度（必背）

- 单轮：`O(nkd)`，其中 E 步 `O(nkd)`、M 步 `O(nd)`
- 整体：`O(nkdt)`，`t` 为迭代轮数（通常 < 50）
- 空间：`O(n + kd)`（不算 `(n,k)` 距离矩阵）

## 高频追问速答

| 追问 | 一句话答案 |
|---|---|
| 为什么一定收敛？ | E 步、M 步都不增大 J，且划分组合有限，必在有限步停下 |
| 是全局最优吗？ | 不是，NP-hard，只能多次初始化取最好 |
| 为什么用均值？ | J 对中心求导置零 → 解就是簇内均值 |
| 一定能发现任意形状的簇？ | 不能，隐含假设是球状、同尺度（欧氏距离 + 硬分配） |
| 数据量大怎么办？ | MiniBatch K-means、Elkan/Hamerly 三角不等式加速、KD-tree |
| 哪些特征不能直接用？ | 类别特征（用 k-modes/k-prototypes）、量纲差异大的（先标准化） |
| 怎么选 k？ | 肘部法、轮廓系数、Gap statistic、业务先验 |

## 文件清单

| 文件 | 说明 |
|---|---|
| `01_hand_calculation.py` | 手算过程，纯 Python，打印每轮分配/均值/J |
| `02_pure_python.py` | 通用纯 Python 版 + 最远点初始化对照 |
| `03_numpy_vectorized.py` | numpy 向量化 + 两种距离写法性能对比 |
| `04_kmeans_plusplus.py` | k-means++ / n_init / 空簇处理 + 三个实测演示 |
| `05_visualize.py` | 生成 `kmeans_iterations.png` 六格图 |
| `kmeans_iterations.png` | 迭代过程、肘部曲线、初始化对比 |
