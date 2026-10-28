# MEC × PSO × SimPy：科研入门实验

## 环境
建议 Python 3.10–3.12，使用虚拟环境。

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
pip install numpy simpy 'pymoo==0.6.2'
```

## 运行

```bash
python handwritten_pso.py
python pymoo_pso.py
```

两段算法都调用 `mec_sim.py` 中同一个 `simulate()` 和 `cost_of_position()`。
输出包括轮询、全选最快服务器的基线，以及两版 PSO 的任务分配和统计指标。

## 目录
- `mec_sim.py`：任务生成、SimPy 上传及串行 CPU 队列仿真、适应度、打印结果
- `handwritten_pso.py`：手写 Global-best PSO + floor 离散映射
- `pymoo_pso.py`：pymoo 内置 PSO + 同一离散映射

## 假设和局限
1. 为教学明确规定 18 个任务信息事先已知，优化器离线联合分配整批任务；不属于真正在线调度。
2. 任务随机到达，各服务器计算速率固定，CPU 采用单核串行 FCFS 队列。
3. 上传到边缘服务器占用固定延迟，各任务上传之间没有无线资源竞争。
4. 不考虑本地计算、结果下载、容器冷启动、任务截止时间和节点故障。
5. PSO 使用连续潜在位置加 `floor` 映射，不是经典 BPSO；每种编码的优化效果需要专门验证。
6. 本代码属于教学模型，单次运行结果不能用于宣称某算法优于另一算法。需增加多随机种子、多任务集与统计对比。
7. 两种 PSO 的初始化与内部算子不同，即使迭代数和粒子数相近也不保证公平同轨迹比较；应进一步对齐适应度调用预算和多次重复。

## 推荐阅读
- SimPy https://simpy.readthedocs.io/en/latest/simpy_intro/index.html
- SimPy Shared Resources https://simpy.readthedocs.io/en/latest/topical_guides/resources.html
- pymoo PSO https://pymoo.org/algorithms/soo/pso.html
- pymoo 算法列表 https://pymoo.org/algorithms/list.html
