"""从零实现经典连续 PSO + 离散服务器编号映射。

运行：python handwritten_pso.py
前提：pip install numpy simpy
"""
import numpy as np
from mec_sim import (N_TASKS, N_SERVERS, cost_of_position, decode,
                     print_baselines, print_report)


def handwritten_pso(seed=7, n_particles=30, n_iterations=39):
    """最简 Global-best PSO：30 个粒子、共 40 轮评价（初始化+39 次更新）。"""
    rng = np.random.default_rng(seed)
    dims = N_TASKS                  # 每维代表一个任务
    lower, upper = 0.0, float(N_SERVERS)  # 连续潜在空间：[0,3)
    w, c1, c2 = 0.7, 1.5, 1.5

    # X: shape=(30, 18)，每行是一只粒子的一套候选卸载方案。
    x = rng.uniform(lower, np.nextafter(upper, lower), size=(n_particles, dims))
    v = rng.uniform(-0.5, 0.5, size=(n_particles, dims))

    # pbest: 每只粒子曾经找到的最优连续潜在位置
    pbest_x = x.copy()
    pbest_f = np.array([cost_of_position(row) for row in x])
    calls = n_particles

    # gbest: 整个粒子群迄今发现的最优连续潜在位置
    idx = int(np.argmin(pbest_f))
    gbest_x = pbest_x[idx].copy()
    gbest_f = float(pbest_f[idx])
    print(f"迭代  0 | 最优平均响应时延 {gbest_f:.4f} s")

    for iteration in range(1, n_iterations + 1):
        # 按元素生成随机向量，分别控制个体经验和群体经验。
        r1 = rng.random(size=x.shape)
        r2 = rng.random(size=x.shape)
        v = w * v + c1 * r1 * (pbest_x - x) + c2 * r2 * (gbest_x - x)
        v = np.clip(v, -0.6, 0.6)  # 速度截断，避免一次跨越过多离散区间
        # 返回浮点数 upper 沿着指向 lower 的方向，紧邻的下一个可表示的浮点数，避免因浮点精度问题导致越界或取到错误的值。
        x = np.clip(x + v, lower, np.nextafter(upper, lower))

        costs = np.array([cost_of_position(row) for row in x])
        calls += n_particles
        improved = costs < pbest_f
        pbest_x[improved] = x[improved]
        pbest_f[improved] = costs[improved]

        idx = int(np.argmin(pbest_f))
        if pbest_f[idx] < gbest_f:
            gbest_f = float(pbest_f[idx])
            gbest_x = pbest_x[idx].copy()

        if iteration % 10 == 0 or iteration == n_iterations:
            print(f"迭代 {iteration:2d} | 最优平均响应时延 {gbest_f:.4f} s")

    return decode(gbest_x), calls


if __name__ == "__main__":
    print_baselines()
    assignment, calls = handwritten_pso()
    print_report("手写 PSO", assignment, calls=calls, show_tasks=True)



