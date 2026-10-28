"""MEC 教学仿真：SimPy 为任务建立排队、传输与计算事件。

单位统一：
- 时间：秒 (s)
- 数据大小：Mbit
- 上行速率：Mbit/s
- CPU 工作量：Gcycles
- CPU 处理速率：Gcycles/s

计算任务 != 数据包；本示例只模拟一次任务传输所需的总时间。
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import simpy


SERVER_SPEEDS = (2.0, 3.0, 4.0)  # Gcycles/s
SERVER_UPLOAD_RATES = (8.0, 10.0, 12.0)  # Mbit/s
N_SERVERS = len(SERVER_SPEEDS)
N_TASKS = 18


@dataclass(frozen=True)  # ※frozen=True表示该数据类型修饰的值一旦初始化后就禁止修改
class Task:
    """到达时间、待上传的数据量和任务所需 CPU 工作量。"""
    task_id: int
    arrival: float       # s
    data_mbit: float     # Mbit
    work_gcycles: float  # Gcycles


def generate_tasks(seed=2026):
    """只生成一次任务集，让不同算法面对完全相同的仿真场景。"""
    rng = np.random.default_rng(seed)
    # ※指数到达间隔是泊松到达过程的常用入门模型；并非突发流量模型。
    gaps = rng.exponential(scale=0.25, size=N_TASKS)
    # ※累计求和，获取每个任务的实际到达时刻
    arrival_times = np.cumsum(gaps)
    # ※从均匀分布中随机取值，取N_TASKS个样本
    sizes = rng.uniform(1.0, 5.0, N_TASKS)
    works = rng.uniform(2.0, 8.0, N_TASKS)
    return tuple(Task(i, float(arrival_times[i]), float(sizes[i]), float(works[i])) for i in range(N_TASKS))


TASKS = generate_tasks()


def decode(position):
    """把连续 PSO 位置映射为服务器编号 0、1、2。

    例如 0.2->0、1.4->1、2.9->2。此处并非经典二进制 PSO。
    注意：多个不同实数位置可能映射到同一个离散方案。
    """
    z = np.asarray(position, dtype=float)
    # 截断函数，低于a_min的取a_min，反之取a_max
    return np.clip(np.floor(z).astype(int), 0, N_SERVERS - 1)


def simulate(assignment, tasks=TASKS, detailed=False):
    """为一份完整的任务分配方案运行一次全新的 SimPy 仿真。

    assignment[i] 是任务 i 的服务器编号。
    同一任务先上传，再请求其目标服务器的单核 CPU。
    每台服务器使用 simpy.Resource(capacity=1)，即串行、非抢占、FCFS。
    同一服务器的多个上传可并行：本教程暂不考虑无线带宽争抢。

    返回字典，含平均响应时延、平均等待、各服务器任务数、记录等。
    """
    assignment = np.asarray(assignment, dtype=int)
    if len(assignment) != len(tasks) or np.any(assignment < 0) or np.any(assignment >= N_SERVERS):
        raise ValueError("assignment 必须为每个任务给出 0/1/2 的服务器编号")

    env = simpy.Environment()  # 创建一个独立的模拟时间轴，从 0 开始
    cpus = [simpy.Resource(env, capacity=1) for _ in range(N_SERVERS)]
    records = []

    def run_one_task(task, server_id):
        """SimPy 进程：到达 -> 上传 -> 进入 CPU 队列 -> 执行 -> 完成。"""
        # yield timeout 是等待模拟时间，不会让 Python 真实睡眠。
        yield env.timeout(task.arrival)
        actual_arrival = env.now

        upload_duration = task.data_mbit / SERVER_UPLOAD_RATES[server_id]
        yield env.timeout(upload_duration)
        cpu_queue_arrival = env.now

        # request() 返回事件；资源忙时 yield request 将进程挂起排队。
        # with 会在执行结束后自动释放 CPU 资源。
        with cpus[server_id].request() as req:
            yield req
            cpu_start = env.now
            waiting = cpu_start - cpu_queue_arrival
            computing = task.work_gcycles / SERVER_SPEEDS[server_id]
            yield env.timeout(computing)
            finished = env.now

        records.append({
            "task": task.task_id,
            "server": server_id,
            "arrival": actual_arrival,
            "upload": upload_duration,
            "waiting": waiting,
            "computing": computing,
            "finish": finished,
            "response": finished - actual_arrival,
        })

    # 一次性注册全部任务进程；由到达事件决定何时执行。
    for task, server_id in zip(tasks, assignment):
        env.process(run_one_task(task, int(server_id)))
    env.run()  # 自动运行直到全部事件完成

    records.sort(key=lambda r: r["task"])
    result = {
        "mean_response": float(np.mean([r["response"] for r in records])),
        "mean_wait": float(np.mean([r["waiting"] for r in records])),
        "max_response": float(max(r["response"] for r in records)),
        "makespan": float(max(r["finish"] for r in records)),
        "server_counts": [int(np.sum(assignment == s)) for s in range(N_SERVERS)],
        "records": records if detailed else [],
    }
    return result


# 相同离散分配方案可以复用结果，避免重复运行完全相同的模拟。
# 固定 TASKS、SERVER_SPEEDS 等配置之后才能这么缓存；若改变场景须清缓存。
@lru_cache(maxsize=200_000)
def cached_cost(assignment_tuple):
    # ※缓存：只记录输入的形参和return的算式结果，本质上是字典，所以输入形参必须可哈希，所以才需要经过cost_of_position转为元组
    return simulate(assignment_tuple)["mean_response"]

# ※计算每个粒子的花销，即优化函数的值
def cost_of_position(position):
    """两种 PSO 调用同一目标函数，使比较条件一致。"""
    return cached_cost(tuple(int(x) for x in decode(position)))


def print_report(name, assignment, calls=None, show_tasks=False):
    """统一打印所有策略的模拟结果。"""
    result = simulate(assignment, detailed=show_tasks)
    print(f"\n--- {name} ---")
    print("任务分配:    ", list(map(int, assignment)))
    print("各服务器任务数:", result["server_counts"])
    print(f"平均响应时延: {result['mean_response']:.4f} s")
    print(f"平均CPU排队:   {result['mean_wait']:.4f} s")
    print(f"最慢任务响应:  {result['max_response']:.4f} s")
    print(f"整体完成时刻:  {result['makespan']:.4f} s")
    if calls is not None:
        print(f"适应度求值次数: {calls}")
    if show_tasks:
        print("任务  节点  到达(s) 上传(s) 排队(s) 执行(s) 响应(s)")
        for r in result["records"]:
            print(f"{r['task']:>3} {r['server']:>5} {r['arrival']:>8.3f} "
                  f"{r['upload']:>7.3f} {r['waiting']:>7.3f} "
                  f"{r['computing']:>7.3f} {r['response']:>7.3f}")
    return result


def print_baselines():
    print("任务数:", len(TASKS), "服务器数:", N_SERVERS)
    print("服务器速率 (Gcycles/s):", SERVER_SPEEDS)
    print("上传速率 (Mbit/s):", SERVER_UPLOAD_RATES)
    print("任务集随机种子: 2026；各算法使用同一批任务\n")
    print_report("轮询基线", [i % N_SERVERS for i in range(N_TASKS)])
    print_report("全部选最快服务器", [N_SERVERS - 1] * N_TASKS)
