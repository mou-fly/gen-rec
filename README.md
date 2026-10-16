# gen-rec
生成式推荐学习

## 目录

- [`learn_kmeans/`](learn_kmeans/README.md) —— 手撕 k-means 五步渐进讲义
  - `01_hand_calculation.py` 手算一遍：E 步分配 → M 步取均值 → 收敛
  - `01b`–`01e` 逐词拆解：M 步怎么写、`pts` 是什么、`inertia` 是什么、`fmt` 是什么
  - `02_pure_python.py` 纯 Python 通用版（含最远点初始化对照）
  - `03_numpy_vectorized.py` numpy 向量化 + 距离展开式性能对比
  - `04_kmeans_plusplus.py` k-means++ / n_init / 空簇处理
  - `05_visualize.py` → `kmeans_iterations.png` 迭代过程图解
