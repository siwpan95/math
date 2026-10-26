"""math（细胞自动机公式实验室）无头自检 —— 不打开 pygame 窗口。

只 import models 包（GridState / FormulaEngine 纯数据与算法层），
不 import controllers / views（会初始化 pygame + pygame_gui 界面）。

验证点：
1. GridState 的初始状态、[0,1] 裁剪、迭代计数推进、形状校验；
2. 默认 Gray-Scott 公式：形状保持、值域裁剪、多步迭代真有状态推进；
3. generate_template 生成的公式模板可被独立加载并正常参与迭代（模板往返）。

运行：python selfcheck.py
"""

import importlib.util
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import FormulaEngine, GridState

# Gray-Scott 常用参数
DU, DV, F, K, DT, DX = 0.16, 0.08, 0.035, 0.06, 1.0, 1.0


def check_grid_state():
    """功能点 1：网格状态的初始化 / 裁剪 / 计数推进 / 形状校验。"""
    width, height = 40, 30
    grid = GridState(width=width, height=height)
    if grid.u.shape != (height, width) or grid.v.shape != (height, width):
        return "reset 后网格形状为 %r / %r，期望 (%d, %d)" % (
            grid.u.shape, grid.v.shape, height, width)
    if not np.allclose(grid.u, 1.0):
        return "reset 后 u 应全为 1"
    if int(np.count_nonzero(grid.v)) != 20 * 20:
        return "reset 后 v 的扰动方块应为 20x20，实际非零 %d 个" % int(
            np.count_nonzero(grid.v))
    if grid.iteration != 0:
        return "reset 后 iteration 应为 0，实际 %r" % (grid.iteration,)

    grid.update(np.full((height, width), 2.0), np.full((height, width), -1.0))
    if not np.allclose(grid.u, 1.0) or not np.allclose(grid.v, 0.0):
        return "update 未把状态裁剪回 [0, 1]：u=%r..%r v=%r..%r" % (
            grid.u.min(), grid.u.max(), grid.v.min(), grid.v.max())
    if grid.iteration != 1:
        return "update 后 iteration 应为 1，实际 %r" % (grid.iteration,)

    grid.increment_iteration()
    if grid.iteration != 2:
        return "increment_iteration 后 iteration 应为 2，实际 %r" % (grid.iteration,)

    try:
        grid.set_state(np.zeros((4, 4)), np.zeros((4, 5)))
    except ValueError:
        pass
    else:
        return "set_state 未对 u/v 形状不匹配抛 ValueError"

    u0 = np.full((height, width), 0.5)
    v0 = np.full((height, width), 3.0)
    grid.set_state(u0, v0)
    if grid.iteration != 0:
        return "set_state 后 iteration 应重置为 0"
    if not np.allclose(grid.v, 1.0):
        return "set_state 未把 v 裁剪到 [0, 1]"
    return ""


def check_gray_scott():
    """功能点 2：默认公式的数值往返（形状/值域/状态推进）。"""
    engine = FormulaEngine()
    if engine.formula_name != "Gray-Scott":
        return "默认公式名应为 'Gray-Scott'，实际 %r" % (engine.formula_name,)

    size = 32
    u = np.ones((size, size))
    v = np.zeros((size, size))
    v[14:18, 14:18] = 1.0
    u_start, v_start = u.copy(), v.copy()

    for step in range(10):
        u, v = engine.execute_update(u, v, DU, DV, F, K, DT, DX)
        if u.shape != (size, size) or v.shape != (size, size):
            return "第 %d 步输出形状被改变：%r / %r" % (step, u.shape, v.shape)
        if not (np.isfinite(u).all() and np.isfinite(v).all()):
            return "第 %d 步出现 NaN/Inf" % step
        if u.min() < 0.0 or u.max() > 1.0 or v.min() < 0.0 or v.max() > 1.0:
            return "第 %d 步输出越出 [0, 1]：u=%r..%r v=%r..%r" % (
                step, u.min(), u.max(), v.min(), v.max())

    if np.allclose(u, u_start) and np.allclose(v, v_start):
        return "10 步迭代后状态毫无变化，公式没有真正推进"
    if not np.any(np.abs(v - v_start) > 1e-6):
        return "v 场没有发生扩散"
    if not np.any(np.abs(u - u_start) > 1e-9):
        return "u 场没有对 v 场作出反应"

    displayed = engine.get_display_array(v)
    if displayed is not v and not np.array_equal(displayed, v):
        return "get_display_array 在无自定义显示函数时未原样返回 v"
    if engine.get_init_state(size, size) != (None, None):
        return "默认公式不应提供自定义初始状态"
    return ""


def check_template_roundtrip():
    """功能点 3：生成的公式模板文件可独立 import 并正常迭代。"""
    engine = FormulaEngine()
    tmpdir = tempfile.mkdtemp(prefix="math_selfcheck_")
    try:
        path = os.path.join(tmpdir, "formula_tpl.py")
        if not engine.generate_template(path):
            return "generate_template 返回 False"
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return "generate_template 未写出非空模板文件"

        spec = importlib.util.spec_from_file_location("math_selfcheck_tpl", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "update"):
            return "模板文件缺少 update 函数"

        half = np.full((8, 8), 0.5)
        nu, nv = module.update(half, half, DU, DV, F, K, DT, DX)
        if nu.shape != (8, 8) or nv.shape != (8, 8):
            return "模板公式输出形状异常：%r / %r" % (nu.shape, nv.shape)
        if nu.min() < 0.0 or nu.max() > 1.0 or nv.min() < 0.0 or nv.max() > 1.0:
            return "模板公式输出越出 [0, 1]"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    engine.formula_name = "broken"
    engine.update_func = None
    engine.restore_default()
    if engine.formula_name != "Gray-Scott" or engine.update_func is None:
        return "restore_default 未恢复默认公式"
    return ""


def main():
    checks = [
        ("GridState init/clip/iteration", check_grid_state),
        ("Gray-Scott default formula iteration", check_gray_scott),
        ("generate_template round-trip", check_template_roundtrip),
    ]
    for label, func in checks:
        try:
            err = func()
        except Exception:
            print("FAIL: %s raised" % label)
            traceback.print_exc()
            return 1
        if err:
            print("FAIL: %s: %s" % (label, err))
            return 1

    print("selfcheck ok: validated %d things -- %s" % (
        len(checks), "; ".join(label for label, _ in checks)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
