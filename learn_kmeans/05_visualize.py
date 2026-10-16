"""第 5 步：把过程画出来。看懂图，才算真的懂 k-means。

产出 kmeans_iterations.png，一张图六格：
  上排：第 1 轮 → 第 2 轮 → 收敛（点和中心怎么一步步挪）
  下排：选 k 的肘部曲线 | 随机初始化的坏结果 | k-means++ 的好结果

运行： python 05_visualize.py
"""

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE / "kmeans_iterations.png"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


km = load_module("km04", "04_kmeans_plusplus.py")

COLORS = ["#e74c3c", "#2980b9", "#27ae60", "#8e44ad", "#e67e22", "#16a085"]


def get_font(size, bold=False):
    for p in (r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
              r"C:\Windows\Fonts\msyh.ttc",
              r"C:\Windows\Fonts\simhei.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default(size=size)


F_TITLE = get_font(17, bold=True)
F_TEXT = get_font(14)
F_SMALL = get_font(13)

# ---------------------------------------------------------------- 画布参数
PANEL_W, PANEL_H = 372, 296
GAP_X, GAP_Y = 22, 34
MARGIN_X, MARGIN_Y = 26, 74


def mapper(box, xlim, ylim):
    x0, y0, x1, y1 = box
    pad = 12

    def to_px(p):
        px = x0 + pad + (p[0] - xlim[0]) / (xlim[1] - xlim[0]) * (x1 - x0 - 2 * pad)
        py = y1 - pad - (p[1] - ylim[0]) / (ylim[1] - ylim[0]) * (y1 - y0 - 2 * pad)
        return px, py

    return to_px


def scatter_panel(dr, box, X, labels, centers, title, sub, xlim, ylim):
    x0, y0, x1, y1 = box
    dr.rectangle(box, fill="#fcfcfc", outline="#dcdcdc")
    to_px = mapper(box, xlim, ylim)

    for p, l in zip(X, labels):
        px, py = to_px(p)
        r = 3.0
        dr.ellipse([px - r, py - r, px + r, py + r], fill=COLORS[int(l) % len(COLORS)])

    for c in centers:                                  # 中心画成粗黑十字
        px, py = to_px(c)
        dr.ellipse([px - 8, py - 8, px + 8, py + 8], outline="black", width=3)
        dr.line([px - 10, py, px + 10, py], fill="black", width=2)
        dr.line([px, py - 10, px, py + 10], fill="black", width=2)

    dr.text((x0 + 10, y0 + 8), title, fill="#1a1a1a", font=F_TITLE)
    dr.text((x0 + 10, y1 - 24), sub, fill="#777777", font=F_SMALL)


def line_panel(dr, box, xs, ys, title, sub, xlabel):
    x0, y0, x1, y1 = box
    dr.rectangle(box, fill="#fcfcfc", outline="#dcdcdc")
    pad_l, pad_r, pad_t, pad_b = 46, 16, 40, 40
    ax0, ax1 = x0 + pad_l, x1 - pad_r
    ay0, ay1 = y1 - pad_b, y0 + pad_t                   # ay0 下, ay1 上
    lo, hi = min(ys) * 0.9, max(ys) * 1.05

    def px(x):
        return ax0 + (x - min(xs)) / (max(xs) - min(xs)) * (ax1 - ax0)

    def py(y):
        return ay0 - (y - lo) / (hi - lo) * (ay0 - ay1)

    dr.line([ax0, ay0, ax1, ay0], fill="#bbbbbb", width=1)   # x 轴
    dr.line([ax0, ay0, ax0, ay1], fill="#bbbbbb", width=1)   # y 轴
    pts = [(px(x), py(y)) for x, y in zip(xs, ys)]
    dr.line(pts, fill="#2980b9", width=2)
    for (cx, cy), x, y in zip(pts, xs, ys):
        dr.ellipse([cx - 3.5, cy - 3.5, cx + 3.5, cy + 3.5], fill="#2980b9")
        dr.text((cx - 4, ay0 + 5), str(x), fill="#555555", font=F_SMALL)
    for frac in (0.0, 0.5, 1.0):                            # y 轴刻度
        y = lo + frac * (hi - lo)
        dr.line([ax0 - 4, py(y), ax0, py(y)], fill="#bbbbbb", width=1)
        dr.text((x0 + 6, py(y) - 7), f"{y:.0f}", fill="#999999", font=F_SMALL)
    dr.text((ax1 - 34, ay0 + 20), xlabel, fill="#555555", font=F_SMALL)
    dr.text((x0 + 10, y0 + 8), title, fill="#1a1a1a", font=F_TITLE)
    dr.text((x0 + 10, y1 - 22), sub, fill="#777777", font=F_SMALL)


def main():
    rng = np.random.default_rng(11)
    true_c = np.array([[0.0, 0.0], [6.5, 5.5], [0.0, 7.5], [7.5, 0.5]])
    X = np.vstack([rng.normal(c, 0.8, size=(70, 2)) for c in true_c])
    k = 4

    xlim = (X[:, 0].min() - 0.8, X[:, 0].max() + 0.8)
    ylim = (X[:, 1].min() - 0.8, X[:, 1].max() + 0.8)

    # 故意给一个很烂的初始化：前两个中心塞在同一个簇里
    bad_init = np.array([[0.0, 0.0], [0.3, 0.2], [6.5, 5.5], [0.0, 7.5]])
    labels, C, J, hist = km.kmeans(X, k, init=bad_init, max_iter=50,
                                   return_history=True)

    # 肘部曲线
    ks = list(range(1, 9))
    elbows = [km.kmeans(X, kk, n_init=5, init="kmeans++", seed=0)[2] for kk in ks]

    # 随机初始化 vs k-means++：挑一个随机会掉坑的 seed
    J_rand, seed_bad = None, 0
    j_kpp = km.kmeans(X, k, n_init=1, init="kmeans++", seed=0)[2]
    for s in range(60):
        j = km.kmeans(X, k, n_init=1, init="random", seed=s)[2]
        if J_rand is None or j > J_rand:
            J_rand, seed_bad = j, s
    lab_r, C_r, J_r = km.kmeans(X, k, n_init=1, init="random", seed=seed_bad)
    lab_p, C_p, J_p = km.kmeans(X, k, n_init=1, init="kmeans++", seed=0)

    # ------------------------------------------------------------ 开始绘制
    cols, rows = 3, 2
    W = MARGIN_X * 2 + cols * PANEL_W + (cols - 1) * GAP_X
    H = MARGIN_Y + rows * PANEL_H + (rows - 1) * GAP_Y + 46
    img = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(img)
    dr.text((MARGIN_X, 22), "k-means 手撕图解：E 步分配 → M 步挪中心 → 收敛",
            fill="#111111", font=get_font(24, bold=True))
    dr.text((MARGIN_X, 50), "上排：一轮是怎么走的    下排：选 k、以及初始化为什么重要",
            fill="#888888", font=F_TEXT)

    def box_of(i):
        r, c = divmod(i, cols)
        x0 = MARGIN_X + c * (PANEL_W + GAP_X)
        y0 = MARGIN_Y + r * (PANEL_H + GAP_Y)
        return (x0, y0, x0 + PANEL_W, y0 + PANEL_H)

    # 上排：把前 3 轮画出来（不够就复用最后几轮）
    picks = [0, 1, len(hist) - 1]
    titles = ["第 1 轮：E 步按最近中心分配", "第 2 轮：中心已挪到簇心附近",
              f"第 {len(hist)} 轮：收敛（中心不再移动）"]
    for i, (idx, title) in enumerate(zip(picks, titles)):
        idx = min(idx, len(hist) - 1)
        Ci, li = hist[idx]
        J_i = float(((X - Ci[li]) ** 2).sum())
        scatter_panel(dr, box_of(i), X, li, Ci, title, f"此刻 J = {J_i:.0f}", xlim, ylim)

    line_panel(dr, box_of(3), ks, elbows,
               "肘部法选 k", "k=4 之后下降明显变缓 → 选 4", "k")

    scatter_panel(dr, box_of(4), X, lab_r, C_r,
                  f"随机初始化（seed={seed_bad}）", f"掉进局部最优   J = {J_r:.0f}", xlim, ylim)
    scatter_panel(dr, box_of(5), X, lab_p, C_p,
                  "k-means++ 初始化", f"正确还原 4 个簇   J = {J_p:.0f}", xlim, ylim)

    dr.text((MARGIN_X, H - 32),
            f"共 {len(hist)} 轮收敛（bad init）；k-means++ 一次就到了 J={J_p:.0f}，"
            f"随机初始化最差能到 J={J_r:.0f} —— 这就是 n_init 存在的理由。",
            fill="#444444", font=F_TEXT)

    img.save(OUT)
    print(f"已保存: {OUT}")
    print(f"迭代轮数 = {len(hist)}，最终 J = {J:.0f}")
    print(f"k-means++ J = {J_p:.0f} vs 随机初始化最差 J = {J_r:.0f}")


if __name__ == "__main__":
    main()
