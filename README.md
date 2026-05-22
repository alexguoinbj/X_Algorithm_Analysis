<div align="center">

# X 开源推荐算法深度分析

**[English](README_EN.md)** | **中文**

---

*基于 X（原 Twitter）2026年5月开源的 "For You Feed" 推荐算法代码库分析*

</div>

---

## 快速导航

| 章节 | 内容 |
|------|------|
| [1. 整体架构与流程](README_CN.md#1-整体架构与流程) | 两阶段架构、数据流、核心设计理念 |
| [2. 召回策略详解](README_CN.md#2-召回策略详解) | Thunder In-Network、Phoenix 双塔模型 |
| [3. 排序模型剖析](README_CN.md#3-排序模型剖析) | Grok Transformer、Candidate Isolation、多目标预测 |
| [4. 负反馈与安全层](README_CN.md#4-负反馈与安全层) | 显式/隐式负反馈、Grox 内容理解 |
| [5. 实时性与近线处理](README_CN.md#5-实时性与近线处理) | Kafka 消费、实时特征、缓存降级 |
| [6. 实验与部署实践](README_CN.md#6-实验与部署实践) | A/B 测试、模型热更新、性能优化 |
| [7. 总结与启发](README_CN.md#7-总结与启发) | 技术亮点、架构优势、行业洞察 |

---

## 核心要点速览

### 系统架构
```
┌─────────────────────────────────────────────────────────────────┐
│                           FOR YOU FEED                          │
├─────────────────────────────────────────────────────────────────┤
│  HOME MIXER (Rust Orchestration)                                │
│  ├── Query Hydration (用户历史、社交图谱)                        │
│  ├── Candidate Sourcing                                         │
│  │   ├── Thunder (In-Network, < 5ms)                            │
│  │   └── Phoenix Retrieval (Out-of-Network, ~50ms)              │
│  ├── Pre-Scoring Filtering                                      │
│  ├── Scoring & Ranking (Grok Transformer)                       │
│  │   ├── Phoenix Scorer → 19种交互概率                          │
│  │   ├── Weighted Scorer → 加权求和                             │
│  │   └── Author Diversity → 多样性衰减                          │
│  └── Post-Selection Filtering (安全过滤)                        │
└─────────────────────────────────────────────────────────────────┘
```

### 关键技术创新

1. **零手工特征** - 完全依赖 Transformer 从交互序列学习
2. **Candidate Isolation** - 候选互不关注的注意力掩码
3. **19种交互预测** - 多目标联合优化
4. **Thunder 内存存储** - Sub-millisecond In-Network 检索
5. **Grok VLM 内容理解** - 智能安全检测

---

## 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| 编排层 | Rust (Home Mixer) | gRPC 服务、Pipeline 编排 |
| 召回层 | Rust (Thunder) + JAX (Phoenix) | In-Network + Out-of-Network |
| 排序层 | JAX (Phoenix) | Grok Transformer 推理 |
| 内容理解 | Python (Grox) | Grok VLM 安全分类 |
| 消息队列 | Kafka | 实时事件消费 |
| 可观测性 | Prometheus + B3 Tracing | 指标监控、分布式追踪 |

---

## 完整分析

**查看完整分析文档：[README_CN.md](README_CN.md)**

**English version: [README_EN.md](README_EN.md)**

---

<div align="center">

*分析完成时间：2026年5月22日*

*基于 X 开源代码库版本：2026年5月15日更新*

</div>
