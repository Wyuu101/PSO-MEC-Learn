"""使用 pymoo 标准 PSO，复用和 handwritten_pso.py 完全相同的 SimPy 环境。

运行：python pymoo_pso.py
前提：pip install numpy simpy pymoo==0.6.2
"""
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from mec_sim import (N_TASKS, N_SERVERS, cost_of_position, decode,
                     print_baselines, print_report)


class OffloadingProblem(ElementwiseProblem):
    """ ※pymoo 规定：每个候选解 x 调用一次 _evaluate，并写入 out['F']。"""
    def __init__(self):
        # n_var: 维度 = 任务数量
        # n_obj: 只有一个优化目标（平均任务响应时延）
        # xl/xu: 潜在连续位置的搜索范围
        # 用略低于3的浮点上界，避免索引越界；decode 本身也会兜底处理。
        super().__init__(n_var=N_TASKS, n_obj=1,
                         xl=np.zeros(N_TASKS),
                         xu=np.full(N_TASKS, np.nextafter(float(N_SERVERS), 0.0)))

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = cost_of_position(x)


def library_pso(seed=7, n_particles=30, generations=40):
    problem = OffloadingProblem()

    algorithm = PSO(
        pop_size=n_particles,
        w=0.7,
        c1=1.5,
        c2=1.5,
        # ※在学习过程中发现的差别较大的地方
        adaptive=False,      # 不自动修改三个核心超参数
        pertube_best=False,   # 关闭 pymoo 的额外最优粒子扰动机制
    )

    result = minimize(
        problem, algorithm,
        # ※配置终止条件为迭代次数满
        termination=("n_gen", generations),
        seed=seed, verbose=False,
    )

    if result.X is None:
        raise RuntimeError("pymoo 没有返回可用解，请检查问题和算法参数")

    # result.X 为连续潜在位置，必须用同一 decode 映射为三个服务器编号。
    assignment = decode(result.X)
    return assignment, int(result.algorithm.evaluator.n_eval)


if __name__ == "__main__":
    print_baselines()
    assignment, calls = library_pso()
    print_report("pymoo PSO", assignment, calls=calls, show_tasks=True)
