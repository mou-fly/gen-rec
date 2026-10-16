"""01e：fmt 是什么，以及它依赖的那套 f-string 格式语法。

01_hand_calculation.py 里的辅助函数，与聚类算法本身无关。

运行： python 01e_what_is_fmt.py
"""

print("=" * 70)
print("① fmt 的定义（01_hand_calculation.py 里原样抄过来的）")
print("=" * 70)


def fmt(vals):
    return "[" + ", ".join(f"{v:g}" for v in vals) + "]"


print("  def fmt(vals):")
print('      return "[" + ", ".join(f"{v:g}" for v in vals) + "]"')
print()
print("  作用：把一串数字变成带方括号的字符串，方便 print")
print("  输入: [1, 2, 3, 4, 10, 11, 12, 13]")
print(f"  输出: {fmt([1, 2, 3, 4, 10, 11, 12, 13])}   ← 就是一个字符串")
print(f"  类型: {type(fmt([1, 2]))}")

print()
print("=" * 70)
print("② 拆开看：join 是把字符串「缝」起来")
print("=" * 70)

pieces = ["1", "2", "3"]
print(f'  ", ".join(["1", "2", "3"])  →  "{", ".join(pieces)}"')
print("  原理: 用「, 」当胶水，把列表里的字符串按顺序粘起来")
print("  ⚠️ 列表里必须全是字符串，混进数字会报错：")
try:
    ", ".join([1, 2, 3])
except TypeError as e:
    print(f"     TypeError: {e}")
print('  → 所以要先 f"{v:g}" 把每个数字转成字符串')

print()
print("=" * 70)
print("③ f-string 里的 :g 到底是什么：通用格式 (general)")
print("=" * 70)

samples = [1.0, 2.5, 7.857142857142857, 0.000123456, 1234567.0, 100.0]
print(f"  {'原值':<22} {'str() 直接打印':<24} {'f\"{{v:g}}\"':<16} {'f\"{{v:.2f}}\"'}")
for v in samples:
    print(f"  {v!r:<22} {str(v):<24} {v:<16g} {v:.2f}")
    # ↑ {v!r} 是 repr（能看到真实精度），{v:<16g} 是左对齐+通用格式
print()
print("  :g 的规则 = 「最多 6 位有效数字 + 去掉无意义的尾随零」")
print("    1.0                    → 1         （.0 被去掉）")
print("    7.857142857142857      → 7.85714   （截到 6 位有效数字）")
print("    1234567.0              → 1.23457e+06（超过 6 位就转科学计数法）")
print("  → 这就是为什么第 1 步的输出是 [1, 7.85714] 而不是 7.857142857142857")

print()
print("=" * 70)
print("④ 顺便看懂同文件里的 :<22（对齐填充）")
print("=" * 70)

groups = [[1], [2, 3, 4, 10, 11, 12, 13]]
for j, g in enumerate(groups):
    print(f"  不加填充: 簇{j + 1} = {fmt(g)}")
for j, g in enumerate(groups):
    print(f"  加 <22  : 簇{j + 1} = {fmt(g):<22} (当前中心 ...)")
print()
print("  <22 表示：左对齐，不足 22 个字符就用空格补齐")
print("  目的：让下一列的「(当前中心 ...)」竖直对齐，像表格一样")

print()
print("=" * 70)
print("⑤ 常用格式符速查")
print("=" * 70)
rows = [
    ("{v:g}", "通用格式，最多 6 位有效数字，去掉尾零", f"{7.857142857142857:g}"),
    ("{v:.4f}", "保留 4 位小数", f"{7.857142857142857:.4f}"),
    ("{v:.2%}", "百分比，保留 2 位小数", f"{0.3:.2%}"),
    ("{v:>8.2f}", "右对齐、总宽 8、2 位小数", f"{3.14159:>8.2f}"),
    ("{v:<8.2f}", "左对齐、总宽 8、2 位小数", f"{3.14159:<8.2f}"),
    ("{v:6d}", "整数、总宽 6（右对齐）", f"{42:6d}"),
    ("{v:,}", "千分位分隔", f"{118977711.2:,}"),
    ("{v:+.3f}", "强制显示正负号", f"{0.5:+.3f}"),
    ("{v!r}", "repr，看真实精度（调试用）", repr(0.1 + 0.2)),
]
for spec, desc, out in rows:
    print(f"  {spec:<10} {desc:<34} → {out}")

print()
print("=" * 70)
print("⑥ 一句话总结")
print("=" * 70)
print("  fmt = format 的缩写，是我写的打印助手，与算法无关")
print("  它 = 「方括号」+「用 ', ' 缝起来的数字」")
print("  :g = 通用格式，6 位有效数字、去尾零 —— 这也是输出里 7.85714 的来源")
