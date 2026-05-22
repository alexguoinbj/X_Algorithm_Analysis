<div align="center">

# X (Twitter) Open-Source Recommendation Algorithm Deep Analysis

**English** | **[中文](README_CN.md)**

---

*In-depth analysis of X's "For You Feed" recommendation algorithm open-sourced in May 2026*

</div>

---

## Quick Navigation

| Chapter | Content |
|---------|---------|
| [1. Overall Architecture & Flow](README_EN.md#1-overall-architecture--flow) | Two-stage architecture, data flow, core design philosophy |
| [2. Retrieval Strategy Details](README_EN.md#2-retrieval-strategy-details) | Thunder In-Network, Phoenix Two-Tower Model |
| [3. Ranking Model Analysis](README_EN.md#3-ranking-model-analysis) | Grok Transformer, Candidate Isolation, Multi-objective Prediction |
| [4. Negative Feedback & Safety Layer](README_EN.md#4-negative-feedback--safety-layer) | Explicit/Implicit negative feedback, Grox content understanding |
| [5. Real-time & Nearline Processing](README_EN.md#5-real-time--nearline-processing) | Kafka consumption, real-time features, caching & degradation |
| [6. Experimentation & Deployment](README_EN.md#6-experimentation--deployment) | A/B testing, model hot updates, performance optimization |
| [7. Summary & Insights](README_EN.md#7-summary--insights) | Technical highlights, architectural advantages, industry insights |

---

## Key Points Overview

### System Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                           FOR YOU FEED                          │
├─────────────────────────────────────────────────────────────────┤
│  HOME MIXER (Rust Orchestration)                                │
│  ├── Query Hydration (user history, social graph)               │
│  ├── Candidate Sourcing                                         │
│  │   ├── Thunder (In-Network, < 5ms)                            │
│  │   └── Phoenix Retrieval (Out-of-Network, ~50ms)              │
│  ├── Pre-Scoring Filtering                                      │
│  ├── Scoring & Ranking (Grok Transformer)                       │
│  │   ├── Phoenix Scorer → 19 engagement probabilities           │
│  │   ├── Weighted Scorer → weighted sum                         │
│  │   └── Author Diversity → diversity decay                     │
│  └── Post-Selection Filtering (safety filtering)                │
└─────────────────────────────────────────────────────────────────┘
```

### Key Technical Innovations

1. **No Hand-Engineered Features** - Completely relies on Transformer to learn from interaction sequences
2. **Candidate Isolation** - Attention mask where candidates don't attend to each other
3. **19 Engagement Predictions** - Multi-objective joint optimization
4. **Thunder In-Memory Storage** - Sub-millisecond In-Network retrieval
5. **Grok VLM Content Understanding** - Intelligent safety detection

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Orchestration | Rust (Home Mixer) | gRPC service, Pipeline orchestration |
| Retrieval | Rust (Thunder) + JAX (Phoenix) | In-Network + Out-of-Network |
| Ranking | JAX (Phoenix) | Grok Transformer inference |
| Content Understanding | Python (Grox) | Grok VLM safety classification |
| Message Queue | Kafka | Real-time event consumption |
| Observability | Prometheus + B3 Tracing | Metrics monitoring, distributed tracing |

---

## Full Analysis

**View complete analysis: [README_EN.md](README_EN.md)**

**中文完整分析: [README_CN.md](README_CN.md)**

---

<div align="center">

*Analysis completed: May 22, 2026*

*Based on X open-source codebase version: May 15, 2026 update*

</div>
