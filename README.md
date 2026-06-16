## 📋 项目简介

SA-VNS-Hybrid-Solver 是一个高效的元启发式混合优化算法求解器，融合了模拟退火算法（SA, Simulated Annealing） 的全局搜索能力与变邻域搜索（VNS, Variable Neighborhood Search） 的局部挖掘能力，专门用于求解大规模 NP 难组合优化问题。

本求解器针对选址路径问题（LRP）、旅行商问题（TSP）、车辆路径问题（VRP）等运筹优化经典问题设计，通过算法融合优势互补，在解的质量与收敛速度之间取得良好平衡。

## ✨ 核心特性

- 🔗 **算法融合优势**：SA 全局探索 + VNS 局部开发，跳出局部最优
- 🎯 **多邻域结构**：支持多种邻域操作算子（swap、insert、2-opt、3-opt 等）
- ⚡ **自适应策略**：温度衰减与邻域切换自适应调整
- 📊 **收敛监控**：实时追踪优化过程与收敛曲线
- 🎨 **结果可视化**：迭代曲线、最优解路径可视化
- 🔌 **模块化设计**：易于扩展新问题与新算子
- 💯 **基准测试**：内置标准测试集与性能对比

## 🏗️ 项目架构
```
SA-VNS-Hybrid-Solver/
├── SA/                         # 模拟退火算法模块
│   ├── sa_core.py              # SA核心算法
│   ├── temperature.py          # 温度调度策略
│   ├── acceptance.py           # 接受准则
│   └── sa_solver.py            # SA求解器封装
├── VNS/                        # 变邻域搜索模块
│   ├── vns_core.py             # VNS核心算法
│   ├── neighborhoods.py        # 邻域结构定义
│   ├── shake.py                # 扰动操作
│   ├── local_search.py         # 局部搜索
│   └── vns_solver.py           # VNS求解器封装
├── hybrid/                     # 混合算法模块
│   ├── sa_vns_hybrid.py        # SA-VNS混合主算法
│   ├── scheduler.py            # 算法调度策略
│   └── adaptive.py             # 自适应参数调整
├── problems/                   # 问题定义模块
│   ├── base_problem.py         # 问题基类
│   ├── tsp.py                  # TSP问题
│   ├── vrp.py                  # VRP问题
│   └── lrp.py                  # LRP选址路径问题
├── utils/                      # 工具模块
│   ├── loader.py               # 数据加载
│   ├── metrics.py              # 性能指标
│   ├── visualization.py        # 结果可视化
│   └── logger.py               # 日志记录
├── examples/                   # 使用示例
│   ├── tsp_demo.py             # TSP求解示例
│   ├── vrp_demo.py             # VRP求解示例
│   └── lrp_demo.py             # LRP求解示例
├── benchmarks/                 # 基准测试
├── configs/                    # 配置文件
└── main.py                     # 主程序入口

```
