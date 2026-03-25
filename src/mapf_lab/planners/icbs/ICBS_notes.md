# ICBS 算法完整讲解

## 一、ICBS 总体架构

ICBS (Improved Conflict-Based Search) 是 CBS 的升级版，核心思想是用**冲突的"基数"（cardinality）** 来更智能地指导搜索。相比盲目的CBS（任意选冲突），ICBS 优先处理那些必然会导致成本增加的冲突（cardinal conflicts），从而减少搜索树展开。

```
ICBS = CBS 的核心框架 + 三大增强
│
├─ 增强1：MDD（Multi-value Decision Diagram）
│  └─ 表示单个agent在约束下的所有最优路径
│
├─ 增强2：冲突分类（Conflict Cardinality）
│  └─ cardinal/semi-cardinal/non-cardinal
│
└─ 增强3：Bypass 机制
   └─ 非cardinal冲突可原地优化，不产生分支
```

---

## 二、核心概念详解

### 2.1 MDD（Multi-value Decision Diagram）

**定义**：对于给定 agent、约束集合、最优成本，所有仍然最优的路径的集合表示。

```python
# MDD 的三个关键属性
MDD = {
    "cost": int,           # 从起点到终点的最优路径长度
    "levels": list[set],   # levels[t] = agent在时间 t 可能到达的所有单元格
    "edges": dict,         # edges[t] = 时间 t 从一个单元格到另一个的所有最优移动
}
```

**例子**：agent 从 (0,0) → (3,3)，最优成本 6，约束集合为 {不能在 t=2 时进入 (1,1)}

```
Level 0:  {(0,0)}                    <- 只能从起点开始
Level 1:  {(1,0), (0,1)}             <- 两种选择，都在最优路径上
Level 2:  {(2,0), (1,1), (0,2)}      <- 冲突约束会被拦截
         （实际应用会排除 (1,1)）
...
Level 6:  {(3,3)}                    <- 只能到终点
```

**代码中的实现**（`mdd.py`）：
```python
def build_mdd(world, robot, constraints, optimal_cost, heuristic_type):
    # 1. BFS 前向扫描：从起点，沿着所有最优移动展开
    # 2. 约束检查：遇到 vertex/edge 约束就跳过
    # 3. DFS 后向剪枝：只保留能到达终点的状态
    return MDD(cost=optimal_cost, levels=..., edges=...)
```

**缓存优化**：
```python
self._mdd_cache: dict[tuple[agent_id, constraints_set, cost], MDD]
# 同一个 agent，同一个约束集合，同一个最优成本 → 复用同一个 MDD
# 测试结果：6616 次缓存命中，220 次 miss（命中率 96.8%）
```

---

### 2.2 冲突分类（Cardinality Analysis）

**动机**：两个 agent 在某处碰撞。这个碰撞有多"严重"？

---

#### **Cardinal Conflict**（必然冲突）
- **定义**：两个 agent 的 MDD 在冲突时刻都只有一条最优路径，且都必须经过碰撞点
- **例子**：
  ```
  Agent 0 的 MDD: t=4 时只能在 (3,3)（别无选择）
  Agent 1 的 MDD: t=4 时只能在 (3,3)（别无选择）
  → Cardinal: 无论怎么约束，都会增加某一方的成本
  ```
- **代码判定**：
  ```python
  forced_0 = mdd_0.width(t=4) == 1 and mdd_0.has_vertex((3,3), t=4)
  forced_1 = mdd_1.width(t=4) == 1 and mdd_1.has_vertex((3,3), t=4)
  if forced_0 and forced_1:
      cardinality = "cardinal"  # 必须分支
  ```

---

#### **Semi-cardinal Conflict**（单边必然）
- **定义**：只有一方被迫在冲突点
- **例子**：
  ```
  Agent 0 的 MDD: t=4 只能在 (3,3)
  Agent 1 的 MDD: t=4 可以在 (3,3)、(2,4)、(4,2)
  → Semi-cardinal: Agent 0 无选择，Agent 1 可避免
  ```

---

#### **Non-cardinal Conflict**（不必然）
- **定义**：两方都有多个选择，冲突发生是因为当前的选择组合不巧合
- **例子**：
  ```
  Agent 0 的 MDD: t=4 可在 (3,3) 或 (2,4)
  Agent 1 的 MDD: t=4 可在 (3,3) 或 (4,2)
  → Non-cardinal: 可能换条路就无冲突了
  ```
- **代码** (`cardinal.py`)：
  ```python
  def _classify_vertex_conflict(conflict, mdd_i, mdd_j):
      t = conflict.time
      cell = conflict.cell
      
      forced_i = mdd_i.width(t) == 1 and mdd_i.has_vertex(cell, t)
      forced_j = mdd_j.width(t) == 1 and mdd_j.has_vertex(cell, t)
      
      if forced_i and forced_j:
          return "cardinal"
      elif forced_i or forced_j:
          return "semi-cardinal"
      else:
          return "non-cardinal"
  ```

---

### 2.3 冲突选择策略（Conflict Selection）

**核心原则**：**优先处理 cardinal，其次 semi-cardinal，最后再碰 non-cardinal**

```python
def select_classified_conflict(classified_conflicts):
    # 排序键：(cardinality_priority, conflict.time)
    # cardinality_priority: cardinal=0, semi-cardinal=1, non-cardinal=2
    
    best = min(
        classified_conflicts,
        key=lambda cc: (_priority(cc.cardinality), cc.conflict.time),
    )
    return best.conflict
```

**效果**：
- **Cardinal first** 保证分支必然减少搜索空间（不会浪费分支）
- **同优先级按时间早到晚** 确定性方向（可复现结果）

---

### 2.4 Bypass 机制（ICBS 特色）

**问题背景**：
- CBS 每次遇到冲突就硬分支（2 个子节点）
- 某些 non-cardinal 冲突分支后，两个子节点的成本仍然相同，但冲突数却减少了
- **浪费**：本可以原地优化，却白白产生了两个节点

**Bypass 的思路**：
```
当前节点有 non-cardinal 冲突
  ↓
试图分支（添加各一个约束到两个agent）
  ↓
检查两个候选子节点：
  - 成本有没有增加？
  - 冲突集合是不是严格子集？
  ↓
如果某个候选满足条件，就原地更新当前节点，不提交到open list
  ↓
继续在当前节点处理下一个冲突
```

**严格子集判定** (`bypass.py`)：

```python
def choose_bypass_candidate(candidates, parent_cost, parent_conflicts):
    parent_conflict_set = {_conflict_to_hashable(c) for c in parent_conflicts}
    
    feasible = [
        c for c in candidates
        if c.cost <= parent_cost 
        and {_conflict_to_hashable(cf) for cf in c.conflicts} < parent_conflict_set
        #  ↑ 这个 < 是集合的strict subset 操作  
        #    意思是：候选冲突集 ⊂ 父冲突集，且不相等
    ]
    
    return min(feasible, key=lambda c: (len(c.conflicts), c.cost))
```

**转成哈希是必要的**，因为 `Conflict` 对象本身不可哈希（是 dataclass）：
```python
def _conflict_to_hashable(conflict):
    if isinstance(conflict, VertexConflict):
        return ("vertex", conflict.agent_i, conflict.agent_j, 
                conflict.time, conflict.cell)
    elif isinstance(conflict, EdgeConflict):
        return ("edge", conflict.agent_i, conflict.agent_j, 
                conflict.time, conflict.edge_i, conflict.edge_j)
```

---

## 三、主流程详解（`solve()` 函数）

```
┌─────────────────────────────────────────────────────────────┐
│  ICBS.solve(world, robots, objective)                      │
└─────────────────────────────────────────────────────────────┘
 │
 ├─ 第1步：初始化
 │  ├─ 重置所有统计计数器（MDD缓存、时间、bypass次数）
 │  └─ 构建根节点（每个agent独立规划，可能有冲突）
 │
 ├─ 第2步：冲突分类与选择（根节点版本）
 │  └─ 调用 _select_conflict() 找到第一个要处理的冲突
 │
 ├─ 第3步：主搜索循环
 │  │
 │  ├─ 从 open_list 弹出成本最低的节点
 │  │
 │  ├─ 检查结束条件
 │  │  ├─ 节点无冲突 → 找到解，返回 success
 │  │  ├─ 超时 → 返回 timeout
 │  │  └─ 超过Node budget → 返回 node_budget_exceeded
 │  │
 │  ├─ ★ BYPASS 循环（ICBS 特色）★
 │  │  │
 │  │  └─ while 节点还有冲突:
 │  │     └─ _try_bypass_node()
 │  │        ├─ 检查冲突的 cardinality
 │  │        ├─ 如果是 cardinal → 跳过 bypass，去常规分支
 │  │        ├─ 如果是 non/semi-cardinal:
 │  │        │  ├─ 对两个分支候选都做 A* 重规划
 │  │        │  ├─ 计算每个候选的成本和冲突集合
 │  │        │  └─ choose_bypass_candidate() 选择能减冲突+不增成本的
 │  │        │     ├─ 找到 → 原地更新节点，继续 bypass 循环
 │  │        │     └─ 没找到 → 跳出 bypass 循环，去普通分支
 │  │        → 更新的内容：node.constraints, node.paths, node.conflict, node.cost
 │  │
 │  ├─ 继续检查：更新后节点有没有冲突
 │  │  ├─ 有 → 去常规分支
 │  │  └─ 无 → 找到解，返回 success
 │  │
 │  ├─ ★ 常规分支（CBS 标准）★
 │  │  │
 │  │  ├─ 把当前冲突分成两个约束
 │  │  ├─ 对每个约束：
 │  │  │  ├─ 重规划受约束 agent 的路径（A*）
 │  │  │  ├─ 新路径可能带来新冲突
 │  │  │  ├─ _select_conflict() 找下一个冲突
 │  │  │  └─ 创建子节点，加入 open_list
 │  │  │
 │  │  └─ 所有子节点都按成本排序（heap）
 │
 └─ 返回结果（带上所有统计指标）
```

---

## 四、详细流程示例

假设有 3 个 agent，初始路径有 5 个冲突：

```
节点 1（根）
│  冲突: {Conf_0, Conf_1, Conf_2, Conf_3, Conf_4}
│  成本: 30
│
├─ _select_conflict() 选出：Conf_0 (t=2, cardinal)
│
└─ _try_bypass_node()
   ├─ 判定 Conf_0 是 cardinal → 无法 bypass
   │  bypass_attempts += 1
   │  bypass_successes += 0
   │
   └─ 跳出 bypass，走普通分支
      ├─ 添加约束到 Agent_a: {约束_Conf0_a}
      │  └─ 重规划 → 新路径，新冲突: {Conf_0, Conf_1}（去掉了Conf_2）
      │     └─ 子节点 2: 成本 32, 冲突 {Conf_0, Conf_1}, cardinal=false
      │
      └─ 添加约束到 Agent_b: {约束_Conf0_b}
         └─ 重规划 → 新路径，新冲突: {Conf_1, Conf_2}（去掉了Conf_0）
            └─ 子节点 3: 成本 32, 冲突 {Conf_1, Conf_2}

节点 2（展开）
│  冲突: {Conf_0, Conf_1}（只剩2个了）
│  成本: 32
│
├─ _select_conflict() 选出：Conf_1 (t=3, non-cardinal)
│
└─ _try_bypass_node()
   ├─ 判定 Conf_1 是 non-cardinal
   │
   ├─ 对两个候选分别 A* 重规划
   │  ├─ 候选 2a: 成本 32, 冲突 {Conf_1}（去掉 Conf_0）✓ 符合条件
   │  └─ 候选 2b: 成本 33, 冲突 {...}（不符合，成本增加）
   │
   ├─ choose_bypass_candidate() 选中 2a
   │
   └─ 原地更新节点
      ├─ node.constraints ← 候选2a的约束
      ├─ node.paths ← 候选2a的路径
      ├─ node.conflict ← Conf_0（下一个待处理冲突）
      ├─ node.cost ← 32（不变）
      │
      │  bypass_attempts += 1
      │  bypass_successes += 1
      │
      └─ 继续 bypass 循环，处理 Conf_0...
```

---

## 五、MDD 缓存与性能

**缓存键**：`(agent_id, frozenset(constraints), int(optimal_cost))`

**为什么这样设计**：
- 同一个 agent
- 同样的约束集合
- 同样的最优成本（由低层规划器返回）
→ MDD 必然相同，可以复用

**测试结果**：
```
MDD缓存命中: 6616 次
MDD缓存未中: 220 次
命中率: 96.8%

总耗时：
- 冲突检测: 0.28s
- MDD构建: 0.084s
- 冲突分类: 0.009s
- 冲突选择: 0.002s
- 低层重规划: 0.81s
```

**启示**：低层规划（A*）占主要耗时，ICBS 通过聪明的冲突选择来减少低层规划次数。

---

## 六、统计指标详解

```python
extra_metrics = {
    # MDD 相关
    "mdd_cache_hits": 6616,         # 缓存命中次数
    "mdd_cache_misses": 220,        # 缓存未中次数
    "mdd_build_count": 220,         # 构建新MDD次数
    
    # 冲突处理
    "select_conflict_calls": 1112,  # 调用冲突选择的次数
    "classified_conflicts_total": 7630,  # 分类过的冲突总数
    "avg_conflicts_per_select": 6.86,    # 平均每次冲突选择看几个冲突
    
    # Bypass 相关（新增）
    "bypass_attempts": 8,           # 尝试 bypass 的次数
    "bypass_successes": 2,          # 成功应用 bypass 的次数
    
    # 耗时
    "time_detect_conflicts": 0.283s,     # 整个搜索中检测冲突的总时间
    "time_get_mdd": 0.084s,              # 获取/构建MDD的总时间
    "time_classify": 0.009s,             # 分类冲突的总时间
    "time_select_rank": 0.002s,          # 选择冲突的总时间
    "time_low_level_replan": 0.815s,     # 低层A*重规划的总时间
}
```

---

## 七、与CBS的关键差异

| 方面 | CBS | ICBS |
|------|-----|------|
| **冲突选择** | 任意（通常最早） | Cardinal-first（有优先级） |
| **冲突分析** | 无 | MDD分类，判断必然性 |
| **Bypass** | 无 | 有（减少不必要的分支） |
| **缓存** | 无 | MDD缓存（hit rate ~97%） |
| **分支因子** | ~2（固定） | <2（bypass减少） |

---

## 八、核心优势

1. **更聪明的搜索方向**
   - Cardinal conflicts 必须分支，不浪费
   - Non-cardinal 先尝试 bypass，只在必须时分支

2. **显著的缓存命中**
   - MDD 在高度约束的区域复用率极高

3. **减少树展开**
   - Bypass 成功时，不产生额外节点
   - 测试中：expanded_ct 从 6246（CBS）降到 609（ICBS 前 bypass 版）

4. **可证明**的最优性
   - 仍然是最优的 MAPF 求解器
   - Bypass 不会错过最优解（只是改进内部搜索顺序）

---

## 总结

ICBS 通过**理解冲突的本质**，用更少的搜索节点找到相同质量的解。这比 CBS 的盲目搜索要聪明得多。
