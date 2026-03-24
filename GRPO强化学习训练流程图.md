# GRPO 强化学习训练流程图

## 训练流程 Mermaid 图

```mermaid
flowchart TB
    subgraph 初始化阶段
        A[开始训练] --> B[加载被训练模型]
        B --> C[初始化上下文 Context]
        C --> D[初始化暂存态度评定评估]
    end

    subgraph 智能体生成阶段
        D --> E[智能体生成参考内容]
        E --> F[将参考内容添加到提示词]
    end

    subgraph 模型采样阶段
        F --> G[被训练模型采样]
        G --> G1[样本1]
        G --> G2[样本2]
        G --> G3[样本3]
        G --> G4[样本4]
    end

    subgraph 态度评定评估阶段
        G1 --> H1[PANAS态度评定]
        G2 --> H2[PANAS态度评定]
        G3 --> H3[PANAS态度评定]
        G4 --> H4[PANAS态度评定]
        
        H1 --> I1[评定结果1]
        H2 --> I2[评定结果2]
        H3 --> I3[评定结果3]
        H4 --> I4[评定结果4]
    end

    subgraph 奖励计算阶段
        I1 --> J1[与暂存评估对比]
        I2 --> J2[与暂存评估对比]
        I3 --> J3[与暂存评估对比]
        I4 --> J4[与暂存评估对比]
        
        J1 --> K1[患者改善奖励1]
        J2 --> K2[患者改善奖励2]
        J3 --> K3[患者改善奖励3]
        J4 --> K4[患者改善奖励4]
        
        G1 --> L1[格式评定]
        G2 --> L2[格式评定]
        G3 --> L3[格式评定]
        G4 --> L4[格式评定]
        
        L1 --> M1[格式奖励1]
        L2 --> M2[格式奖励2]
        L3 --> M3[格式奖励3]
        L4 --> M4[格式奖励4]
        
        K1 --> N1[总奖励1]
        K2 --> N2[总奖励2]
        K3 --> N3[总奖励3]
        K4 --> N4[总奖励4]
        
        M1 --> N1
        M2 --> N2
        M3 --> N3
        M4 --> N4
    end

    subgraph GRPO强化学习阶段
        N1 --> O[GRPO算法计算]
        N2 --> O
        N3 --> O
        N4 --> O
        
        O --> P[更新模型参数]
    end

    subgraph 最优样本保存阶段
        N1 --> Q[比较总奖励]
        N2 --> Q
        N3 --> Q
        N4 --> Q
        
        Q --> R[选择最高分样本]
        R --> S[保存最优回答到上下文]
        R --> T[更新暂存态度评定评估]
    end

    subgraph 循环判断
        S --> U{是否继续训练?}
        T --> U
        P --> U
        
        U -->|是| E
        U -->|否| V[保存模型]
        V --> W[结束训练]
    end

    style A fill:#e1f5fe
    style W fill:#c8e6c9
    style O fill:#fff3e0
    style Q fill:#fce4ec
    style R fill:#f3e5f5
```

## 简化版流程图

```mermaid
flowchart LR
    subgraph 输入
        A[上下文 Context]
        B[暂存态度评估]
    end

    subgraph 生成
        C[智能体生成参考]
        D[模型采样×4]
    end

    subgraph 评估
        E[PANAS态度评定×4]
        F[格式评定×4]
    end

    subgraph 奖励
        G[患者改善奖励]
        H[格式奖励]
        I[总奖励]
    end

    subgraph 更新
        J[GRPO更新参数]
        K[保存最优样本]
    end

    subgraph 输出
        L[更新上下文]
        M[更新暂存评估]
    end

    A --> C
    B --> C
    C --> D
    D --> E
    D --> F
    E --> G
    F --> H
    G --> I
    H --> I
    I --> J
    I --> K
    J --> L
    K --> L
    K --> M
    L --> A
    M --> B
```

## 详细步骤说明

### 1. 初始化阶段
- 加载被训练模型
- 初始化上下文 Context
- 初始化暂存态度评定评估（上一轮的 PANAS 评分）

### 2. 智能体生成阶段
- 智能体生成参考内容
- 将参考内容添加到被训练模型的提示词中

### 3. 模型采样阶段
- 被训练模型采样 4 个样本
- 每个样本都是独立的输出

### 4. 态度评定评估阶段
- 对每个样本进行 PANAS 态度评定
- 输出 JSON 格式的评定结果

### 5. 奖励计算阶段
- **患者改善奖励**：当前评定与暂存评定对比
  - PA（积极情绪）增加 → 正奖励
  - NA（消极情绪）减少 → 正奖励
- **格式奖励**：检查输出格式是否正确
  - 格式正确 → 正奖励
  - 格式错误 → 负奖励或零奖励
- **总奖励** = 患者改善奖励 + 格式奖励

### 6. GRPO 强化学习阶段
- 使用 GRPO 算法计算梯度
- 更新模型参数

### 7. 最优样本保存阶段
- 比较 4 个样本的总奖励
- 选择最高分样本
- 保存最优回答到上下文
- 更新暂存态度评定评估（供下一轮使用）

### 8. 循环判断
- 判断是否继续训练
- 如果继续，返回智能体生成阶段
- 如果结束，保存模型

## 奖励函数设计

```python
def calculate_reward(current_panas, previous_panas, format_correct):
    # 患者改善奖励
    pa_improvement = current_panas['positive_affect']['total'] - previous_panas['positive_affect']['total']
    na_reduction = previous_panas['negative_affect']['total'] - current_panas['negative_affect']['total']
    
    patient_improvement_reward = pa_improvement * 0.5 + na_reduction * 0.5
    
    # 格式奖励
    format_reward = 1.0 if format_correct else -0.5
    
    # 总奖励
    total_reward = patient_improvement_reward + format_reward
    
    return total_reward
```

## 数据流示意

```
┌─────────────────────────────────────────────────────────────────┐
│                        第 N 轮训练                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  输入:                                                           │
│  - Context: [历史对话...]                                        │
│  - 暂存评估: {PA: 25, NA: 30}                                    │
│                                                                  │
│  智能体生成: "基于5Ps框架，建议询问..."                           │
│                                                                  │
│  模型采样:                                                        │
│  - 样本1: "你能告诉我更多关于..."                                 │
│  - 样本2: "我理解你的感受..."                                     │
│  - 样本3: "让我们来分析一下..."                                   │
│  - 样本4: "你觉得这个问题..."                                     │
│                                                                  │
│  态度评定:                                                        │
│  - 样本1: {PA: 27, NA: 28} → 改善奖励: +2.0                      │
│  - 样本2: {PA: 26, NA: 29} → 改善奖励: +1.0                      │
│  - 样本3: {PA: 28, NA: 27} → 改善奖励: +3.0                      │
│  - 样本4: {PA: 24, NA: 31} → 改善奖励: -1.0                      │
│                                                                  │
│  格式奖励:                                                        │
│  - 样本1: 格式正确 → +1.0                                        │
│  - 样本2: 格式正确 → +1.0                                        │
│  - 样本3: 格式正确 → +1.0                                        │
│  - 样本4: 格式错误 → -0.5                                        │
│                                                                  │
│  总奖励:                                                          │
│  - 样本1: 3.0                                                    │
│  - 样本2: 2.0                                                    │
│  - 样本3: 4.0 ← 最高分                                           │
│  - 样本4: -1.5                                                   │
│                                                                  │
│  输出:                                                           │
│  - 更新模型参数 (GRPO)                                           │
│  - 保存样本3到上下文                                             │
│  - 更新暂存评估: {PA: 28, NA: 27}                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

*文档创建时间: 2026-03-20*
