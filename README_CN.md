# X 开源推荐算法深度分析

> 基于 X（原 Twitter）2026年5月开源的 "For You Feed" 推荐算法代码库分析

**[English](README_EN.md)** | **中文**

---

## 目录

1. [整体架构与流程](#1-整体架构与流程)
2. [召回策略详解](#2-召回策略详解)
3. [排序模型剖析](#3-排序模型剖析)
4. [负反馈与安全层](#4-负反馈与安全层)
5. [实时性与近线处理](#5-实时性与近线处理)
6. [实验与部署实践](#6-实验与部署实践)
7. [总结与启发](#7-总结与启发)

---

## 1. 整体架构与流程

### 1.1 系统概述

X 的 "For You" 推荐流算法是一个经典的**两阶段推荐架构**：召回（Retrieval）+ 排序（Ranking），辅以多层过滤机制。系统采用 Rust 和 Python/JAX 混合技术栈，核心服务通过 gRPC 通信。

**关键技术栈：**
- **Orchestration 层**：Rust（Home Mixer）
- **In-Network 存储**：Rust（Thunder）
- **ML 模型推理**：Python/JAX（Phoenix）
- **内容理解**：Python（Grox，基于 Grok VLM）

### 1.2 三阶段架构实现

虽然代码库重点展示了召回和排序两个核心阶段，但完整的请求流程包含以下阶段：

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FOR YOU FEED REQUEST                               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                   HOME MIXER                                    │
│                              (Rust Orchestration)                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 1: QUERY HYDRATION                                               │   │
│  │  ─────────────────────────                                              │   │
│  │  • User engagement history (likes, replies, reposts, dwells)            │   │
│  │  • Following list & social graph                                        │   │
│  │  • Muted keywords & blocked users                                       │   │
│  │  • Previously seen/served post IDs (bloom filters)                      │   │
│  │  • Device context (IP, timezone, network type)                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                         │
│                                       ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 2: CANDIDATE SOURCING                                            │   │
│  │  ───────────────────────────                                            │   │
│  │  ┌──────────────────┐    ┌────────────────────────────────────────┐     │   │
│  │  │  THUNDER         │    │  PHOENIX RETRIEVAL                     │     │   │
│  │  │  (In-Network)    │    │  (Out-of-Network)                      │     │   │
│  │  │                  │    │                                        │     │   │
│  │  │  Followed users  │    │  Two-tower ANN search                  │     │   │
│  │  │  recent posts    │    │  across global corpus                  │     │   │
│  │  └──────────────────┘    └────────────────────────────────────────┘     │   │
│  │         + Additional sources: Phoenix MoE, Topics, Ads, Who-to-Follow   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                         │
│                                       ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 3: CANDIDATE HYDRATION                                           │   │
│  │  ─────────────────────────────                                          │   │
│  │  • Core post metadata (text, media entities, creation time)             │   │
│  │  • Author info (username, verification, follower count)                 │   │
│  │  • Engagement counts (likes, replies, retweets)                         │   │
│  │  • Language codes, subscription status, brand safety signals            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                         │
│                                       ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 4: PRE-SCORING FILTERING                                         │   │
│  │  ──────────────────────────────                                         │   │
│  │  • DropDuplicatesFilter                                                 │   │
│  │  • CoreDataHydrationFilter                                              │   │
│  │  • AgeFilter (remove too old posts)                                     │   │
│  │  • SelfpostFilter (remove user's own posts)                             │   │
│  │  • RepostDeduplicationFilter                                            │   │
│  │  • IneligibleSubscriptionFilter                                         │   │
│  │  • PreviouslySeenPostsFilter                                            │   │
│  │  • MutedKeywordFilter                                                   │   │
│  │  • AuthorSocialgraphFilter (blocked/muted authors)                      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                         │
│                                       ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 5: SCORING & RANKING                                             │   │
│  │  ───────────────────────────                                            │   │
│  │  ┌──────────────────────────┐                                           │   │
│  │  │  Phoenix Scorer          │  → P(like), P(reply), P(repost), ...      │   │
│  │  │  (Grok Transformer)      │                                           │   │
│  │  └──────────────────────────┘                                           │   │
│  │               │                                                         │   │
│  │               ▼                                                         │   │
│  │  ┌──────────────────────────┐                                           │   │
│  │  │  Weighted Scorer         │  → Final Score = Σ(weight × P(action))    │   │
│  │  │  (Multi-obj combination) │                                           │   │
│  │  └──────────────────────────┘                                           │   │
│  │               │                                                         │   │
│  │               ▼                                                         │   │
│  │  ┌──────────────────────────┐                                           │   │
│  │  │  Author Diversity        │  → Attenuate repeated author scores       │   │
│  │  │  Scorer                  │                                           │   │
│  │  └──────────────────────────┘                                           │   │
│  │               │                                                         │   │
│  │               ▼                                                         │   │
│  │  ┌──────────────────────────┐                                           │   │
│  │  │  OON Weight Adjuster     │  → Downweight out-of-network content      │   │
│  │  └──────────────────────────┘                                           │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                         │
│                                       ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 6: SELECTION                                                     │   │
│  │  ─────────────────────                                                  │   │
│  │  • Sort by final score, select top K candidates                         │   │
│  │  • Apply interleaving for ads injection                                 │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                         │
│                                       ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 7: POST-SELECTION FILTERING                                      │   │
│  │  ─────────────────────────────────                                      │   │
│  │  • VFFilter (visibility filtering: deleted/spam/violence/gore)          │   │
│  │  • DedupConversationFilter (conversation thread dedup)                  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                RANKED FEED RESPONSE                             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 数据流与组件协作

| 组件 | 职责 | 输入 | 输出 | 技术栈 |
|------|------|------|------|--------|
| **Home Mixer** | 编排协调 | gRPC Request | Ranked Feed | Rust |
| **Thunder** | In-Network 候选 | Following List | Recent Posts | Rust (in-memory) |
| **Phoenix Retrieval** | Out-of-Network 候选 | User Embedding | Similar Posts | JAX |
| **Phoenix Ranking** | 精排打分 | Candidates + User History | Engagement Probs | JAX |
| **Grox** | 内容理解 | Post Content | Safety Labels | Python + Grok VLM |

### 1.4 核心设计理念

**零手工特征（No Hand-Engineered Features）：**
> "We have eliminated every single hand-engineered feature and most heuristics from the system. The Grok-based transformer does all the heavy lifting."

这是 X 推荐系统最激进的设计决策——完全依赖 Transformer 模型从用户交互历史中学习相关性，摒弃了传统的手工特征工程。

---

## 2. 召回策略详解

### 2.1 主要召回通道

| 通道名称 | 类型 | 候选量级 | 延迟目标 | 说明 |
|----------|------|----------|----------|------|
| **Thunder** | In-Network | ~500-1000 | < 5ms | 关注用户的最近推文，内存存储 |
| **Phoenix Retrieval** | Out-of-Network | ~200-500 | ~50ms | 双塔模型 ANN 检索 |
| **Phoenix MoE** | Out-of-Network | 可变 | ~50ms | Mixture of Experts 变体 |
| **Phoenix Topics** | Out-of-Network | 可变 | ~50ms | 基于话题的召回 |
| **Cached Posts** | 缓存 | 可变 | < 1ms | 之前请求的缓存结果 |
| **Ads** | 广告 | 可变 | ~50ms | 广告候选池 |
| **Who to Follow** | 用户推荐 | 可变 | ~30ms | 推荐关注用户 |

### 2.2 Thunder（In-Network 召回）

**架构特点：**
- **内存存储**：使用 Rust 的 `DashMap` 实现并发安全的内存哈希表
- **Kafka 消费**：实时消费推文创建/删除事件
- **分层存储**：原始推文、回复/转推、视频推文分别存储
- **自动清理**：定期删除超过保留期（默认2天）的旧推文

**数据结构：**
```rust
PostStore {
    posts: DashMap<i64, LightPost>,                    // 按 post_id 索引
    original_posts_by_user: DashMap<i64, VecDeque<TinyPost>>,  // 原创推文
    secondary_posts_by_user: DashMap<i64, VecDeque<TinyPost>>, // 回复/转推
    video_posts_by_user: DashMap<i64, VecDeque<TinyPost>>,     // 视频推文
}
```

**性能指标：**
- 支持 sub-millisecond 级别的查询
- 每个用户最多保留 N 条原创推文、M 条回复
- 请求超时可配置（默认无超时）

### 2.3 Phoenix Retrieval（Out-of-Network 召回）

**双塔架构（Two-Tower Model）：**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         PHOENIX RETRIEVAL MODEL                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────────────┐              ┌──────────────────────┐             │
│   │     USER TOWER       │              │   CANDIDATE TOWER    │             │
│   │                      │              │                      │             │
│   │  Input:              │              │  Input:              │             │
│   │  • User hashes       │              │  • Post hashes       │             │
│   │  • Engagement seq    │              │  • Author hashes     │             │
│   │    (likes, replies)  │              │  • Product surface   │             │
│   │                      │              │                      │             │
│   │  ┌────────────────┐  │              │  ┌────────────────┐  │             │
│   │  │  Transformer   │  │              │  │  Projection    │  │             │
│   │  │  (4 layers)    │  │              │  │  Layer         │  │             │
│   │  └────────────────┘  │              │  └────────────────┘  │             │
│   │         │            │              │         │            │             │
│   │         ▼            │              │         ▼            │             │
│   │  User Embedding      │              │  Post Embedding      │             │
│   │    [B, D]            │              │    [N, D]            │             │
│   └──────────┬───────────┘              └──────────┬───────────┘             │
│              │                                     │                         │
│              │            ┌────────────┐           │                         │
│              └────────────┤  Dot Prod  ├───────────┘                         │
│                           └─────┬──────┘                                     │
│                                 │                                            │
│                                 ▼                                            │
│                     Top-K Similar Posts                                      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**关键技术细节：**

1. **Hash-Based Embeddings**：
   - 用户/推文/作者各使用 2 个哈希函数
   - 词汇表大小：1,000,000（每个实体）
   - 多个哈希嵌入通过线性投影合并

2. **模型配置（Mini 版）**：
   - 嵌入维度：128
   - Transformer 层数：4
   - 注意力头数：4
   - Key 大小：32

3. **检索流程**：
   - 用户历史序列 → Transformer → 用户嵌入
   - 候选语料库 → 预计算候选嵌入
   - 点积相似度 → Top-K 检索（默认 200）

### 2.4 多样性平衡机制

**Out-of-Network 权重调整：**
```rust
// OON 内容的基础权重因子
let oon_weight_factor = params.get(OonWeightFactor);

// 新用户特殊处理
if is_eligible_new_user {
    NEW_USER_OON_WEIGHT_FACTOR  // 新用户给予更多 OON 内容
}

// 话题请求特殊处理
if !query.topic_ids.is_empty() {
    return params.get(TopicOonWeightFactor);
}
```

**作者多样性衰减：**
```rust
fn diversity_multiplier(decay_factor: f64, floor: f64, position: usize) -> f64 {
    (1.0 - floor) * decay_factor.powf(position as f64) + floor
}
```
同一作者的第 N 条推文得分乘以衰减因子，确保 Feed 中不会过度集中于少数作者。

---

## 3. 排序模型剖析

### 3.1 Phoenix Ranking 模型架构

Phoenix 排序模型基于 **Grok-1 Transformer** 架构，专门为推荐场景优化，核心创新是 **Candidate Isolation（候选隔离）**。

**模型架构图：**

```
                              PHOENIX RANKING MODEL
    ┌────────────────────────────────────────────────────────────────────────────┐
    │                                                                            │
    │                              OUTPUT LOGITS                                 │
    │                        [B, num_candidates, num_actions]                    │
    │                                    │                                       │
    │                                    │ Unembedding                           │
    │                                    │ Projection                            │
    │                                    │                                       │
    │                    ┌───────────────┴───────────────┐                       │
    │                    │                               │                       │
    │                    │    Extract Candidate Outputs  │                       │
    │                    │    (positions after history)  │                       │
    │                    │                               │                       │
    │                    └───────────────┬───────────────┘                       │
    │                                    │                                       │
    │                    ┌───────────────┴───────────────┐                       │
    │                    │                               │                       │
    │                    │         Transformer           │                       │
    │                    │     (with special masking)    │                       │
    │                    │                               │                       │
    │                    │   Candidates CANNOT attend    │                       │
    │                    │   to each other               │                       │
    │                    │                               │                       │
    │                    └───────────────┬───────────────┘                       │
    │                                    │                                       │
    │    ┌───────────────────────────────┼───────────────────────────────┐       │
    │    │                               │                               │       │
    │    ▼                               ▼                               ▼       │
    │ ┌──────────┐              ┌─────────────────┐              ┌────────────┐  │
    │ │   User   │              │     History     │              │ Candidates │  │
    │ │Embedding │              │   Embeddings    │              │ Embeddings │  │
    │ │  [B, 1]  │              │    [B, S, D]    │              │  [B, C, D] │  │
    │ └──────────┘              └─────────────────┘              └────────────┘  │
    │                                                                            │
    └────────────────────────────────────────────────────────────────────────────┘
```

**候选隔离注意力掩码（Candidate Isolation Mask）：**

```
                    ATTENTION MASK VISUALIZATION

         Keys (what we attend TO)
         ─────────────────────────────────────────────▶

         │ User │    History (S positions)    │   Candidates (C positions)    │
    ┌────┼──────┼─────────────────────────────┼───────────────────────────────┤
    │    │      │                             │                               │
    │ U  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✗   ✗   ✗   ✗   ✗   ✗   ✗    │
    │    │      │                             │                               │
    ├────┼──────┼─────────────────────────────┼───────────────────────────────┤
 Q  │    │      │                             │                               │
 u  │ H  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✗   ✗   ✗   ✗   ✗   ✗   ✗    │
 e  │ i  │      │                             │                               │
 r  │ s  │      │                             │                               │
 i  │ t  │      │                             │                               │
 e  │    │      │                             │                               │
 s  ├────┼──────┼─────────────────────────────┼───────────────────────────────┤
    │    │      │                             │  DIAGONAL ONLY (self-attend)  │
 │  │ C  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✓   ✗   ✗   ✗   ✗   ✗   ✗    │
 │  │ a  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✗   ✓   ✗   ✗   ✗   ✗   ✗    │
 │  │ n  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✗   ✗   ✓   ✗   ✗   ✗   ✗    │
 ▼  │ d  │      │                             │                               │
    │ s  │      │                             │                               │
    └────┴──────┴─────────────────────────────┴───────────────────────────────┘

    ✓ = Can attend (1)          ✗ = Cannot attend (0)
```

**核心设计优势：**
- 候选推文的得分不依赖于批次中的其他推文
- 使得分数具有一致性和可缓存性
- 支持并行推理

### 3.2 特征体系

**1. 用户侧特征：**
- `user_hashes`: 用户 ID 的多哈希嵌入（2个哈希函数）
- `user_ip_embeddings`: IP 地址嵌入（可选）
- `engagement_history`: 用户交互序列（最近 127 条）

**2. 历史序列特征（每条交互）：**
- `history_post_hashes`: 推文 ID 哈希
- `history_author_hashes`: 作者 ID 哈希
- `history_actions`: 交互类型（like, reply, repost 等）
- `history_product_surface`: 产品表面（iOS, Android, Web）
- `history_continuous_actions`: 连续值（如停留时间）

**3. 候选推文特征：**
- `candidate_post_hashes`: 推文 ID 哈希
- `candidate_author_hashes`: 作者 ID 哈希
- `candidate_product_surface`: 产品表面
- `post_age_bucket`: 推文年龄桶（1小时粒度，最大80小时）

**4. 交叉特征：**
- 通过 Transformer 自注意力机制隐式学习
- 用户历史与候选推文之间的交互

### 3.3 多目标预测

模型同时预测 **19 种交互类型的概率**：

| 类别 | 预测目标 | 说明 |
|------|----------|------|
| **正向** | `favorite` | 点赞概率 |
| | `reply` | 回复概率 |
| | `repost` | 转推概率 |
| | `quote` | 引用概率 |
| | `click` | 点击概率 |
| | `profile_click` | 点击主页概率 |
| | `video_view` | 视频观看概率 |
| | `photo_expand` | 图片展开概率 |
| | `share` | 分享概率 |
| | `dwell` | 停留概率 |
| | `follow_author` | 关注作者概率 |
| **负向** | `not_interested` | 不感兴趣概率 |
| | `block_author` | 屏蔽作者概率 |
| | `mute_author` | 静音作者概率 |
| | `report` | 举报概率 |

### 3.4 加权评分公式

**最终得分计算：**

```rust
Final_Score = Σ (weight_i × P(action_i))

// 正向行为权重（示例）
favorite: weight × P(like)
reply: weight × P(reply)
retweet: weight × P(repost)
quote: weight × P(quote)
click: weight × P(click)
dwell: weight × P(dwell)
follow_author: weight × P(follow)

// 负向行为权重（负值，拉低分数）
not_interested: weight × P(not_interested)
block_author: weight × P(block_author)
mute_author: weight × P(mute_author)
report: weight × P(report)
```

**分数偏移处理：**
```rust
fn offset_score(combined_score: f64, w: &ScoringWeights) -> f64 {
    if combined_score < 0.0 {
        (combined_score + w.negative_sum) / w.total_sum * NEGATIVE_SCORES_OFFSET
    } else {
        combined_score + NEGATIVE_SCORES_OFFSET
    }
}
```

### 3.5 推理机制

**模型配置（生产版推测）：**
| 参数 | Mini 版 | 生产版（推测） |
|------|---------|----------------|
| 嵌入维度 | 128 | 512-1024 |
| Transformer 层数 | 4 | 12-24 |
| 注意力头数 | 4 | 16-32 |
| 历史序列长度 | 127 | 127-255 |
| 候选序列长度 | 64 | 64-128 |
| 推理精度 | bfloat16 | bfloat16 |

**实时特征处理：**
- 用户交互历史通过 `scoring_sequence` 实时传入
- 推文年龄基于 `impr_ts` 和 `post_creation_ts` 实时计算
- 停留时间等连续特征通过 MLP 投影到嵌入空间

**连续值归一化：**
```python
def normalize_continuous_value(values, config):
    values_clamped = jnp.clip(values, 0.0, config.norm_scale)
    if config.use_log:
        return jnp.log1p(values_clamped) / jnp.log1p(config.norm_scale)
    else:
        return values_clamped / config.norm_scale
```

---

## 4. 负反馈与安全层

### 4.1 显式负反馈处理

**预评分过滤器中的负反馈：**

| 过滤器 | 功能 | 位置 |
|--------|------|------|
| `MutedKeywordFilter` | 移除包含用户静音关键词的推文 | Pre-Scoring |
| `AuthorSocialgraphFilter` | 移除被屏蔽/静音用户的推文 | Pre-Scoring |
| `PreviouslySeenPostsFilter` | 移除用户已看过的推文 | Pre-Scoring |

**评分阶段的负反馈建模：**

模型显式预测负向行为概率：
- `P(not_interested)` - 不感兴趣
- `P(block_author)` - 屏蔽作者
- `P(mute_author)` - 静音作者
- `P(report)` - 举报

这些概率乘以负权重后加入总分，自动压低用户可能反感的内容。

### 4.2 隐式负反馈建模

**停留时间（Dwell Time）分析：**

代码中支持连续值特征，特别关注停留时间：
```python
# 历史中的停留时间嵌入
dwell_values = batch.history_continuous_actions[:, :, 1]  # index 1 = dwell_time
history_continuous_embeddings = self._project_continuous_value_to_embedding(
    dwell_values,
    config.emb_size,
    "history_dwell_time",
    config.continuous_action_config.norm_config,
    config.continuous_action_hidden_dim,
)
```

**"未停留"信号（Not Dwelled）：**
```rust
// 评分权重中包含 not_dwelled
not_dwelled: params.get(NotDwelledWeight)
```
快速划过（未产生有效停留）的推文会被标记，模型学习这种隐式负反馈。

### 4.3 内容安全层

**Grox 内容理解管道：**

```
┌─────────────────────────────────────────────────────────────────┐
│                        GROX PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────────┐                                        │
│  │  Post Input        │                                        │
│  └─────────┬──────────┘                                        │
│            │                                                    │
│            ▼                                                    │
│  ┌────────────────────┐                                        │
│  │  Spam Detection    │  ← 低粉丝用户垃圾评论检测               │
│  │  (Grok VLM)        │                                        │
│  └─────────┬──────────┘                                        │
│            │                                                    │
│            ▼                                                    │
│  ┌────────────────────┐                                        │
│  │  Safety Screening  │  ← PTOS 政策合规检测                   │
│  │  (Multi-classifier)│                                        │
│  └─────────┬──────────┘                                        │
│            │                                                    │
│            ▼                                                    │
│  ┌────────────────────┐                                        │
│  │  Content Category  │  ← 推文分类（体育/新闻/娱乐等）         │
│  │  Classification    │                                        │
│  └─────────┬──────────┘                                        │
│            │                                                    │
│            ▼                                                    │
│  ┌────────────────────┐                                        │
│  │  Multimodal        │  ← 图文多模态嵌入生成                   │
│  │  Embedding         │                                        │
│  └─────────┬──────────┘                                        │
│            │                                                    │
│            ▼                                                    │
│  ┌────────────────────┐                                        │
│  │  Annotations Sink  │  ← 结果写入存储                        │
│  └────────────────────┘                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**安全过滤实现：**

```rust
// VF (Visibility Filtering) 过滤器
fn should_drop(reason: &Option<FilteredReason>) -> bool {
    match reason {
        Some(FilteredReason::SafetyResult(safety_result)) => {
            matches!(safety_result.action, Action::Drop(_))
        }
        Some(_) => true,  // 其他过滤原因也移除
        None => false,
    }
}
```

**品牌安全（Brand Safety）：**
- `AdsBrandSafetyHydrator`: 广告品牌安全标记
- `AdsBrandSafetyVfHydrator`: 广告可见性过滤安全

### 4.4 安全层介入位置

| 阶段 | 安全机制 | 说明 |
|------|----------|------|
| **预评分** | `AuthorSocialgraphFilter` | 屏蔽/静音用户内容 |
| **预评分** | `MutedKeywordFilter` | 静音关键词过滤 |
| **后选择** | `VFFilter` | 可见性过滤（spam/violence/gore） |
| **异步** | Grox Pipeline | 深度内容理解（VLM） |

---

## 5. 实时性与近线处理

### 5.1 Thunder 实时管道

**Kafka 消费架构：**
```rust
// Thunder 实时消费推文事件
- 消费 topic: tweet_create, tweet_delete
- 内存存储，无需磁盘 I/O
- 支持自动 trim 过期推文
```

**保留策略：**
- 默认保留期：2天（172,800秒）
- 定期自动清理（可配置间隔）
- 按用户维度裁剪，限制每用户最大推文数

### 5.2 实时特征计算

**推文年龄实时计算：**
```python
def compute_post_age_bucket(impr_ts_sec, post_creation_ts_sec, granularity_mins=60):
    post_age_minutes = (impr_ts_sec - post_creation_ts_sec) // 60
    bucket = (post_age_minutes // granularity_mins) + 1
    bucket = jnp.clip(bucket, 0, overflow_bucket)
    return bucket.astype(jnp.int32)
```

**停留时间实时投影：**
```python
# 实时归一化 → MLP 投影 → 嵌入
values_normalized = normalize_continuous_value(values, config.norm_config)
hidden = jax.nn.gelu(jnp.dot(values_expanded, proj1))
embedding = jnp.dot(hidden, proj2)
```

### 5.3 缓存与降级策略

**缓存机制：**
```rust
// PhoenixScorer 中的缓存检查
fn enable(&self, query: &ScoredPostsQuery) -> bool {
    !query.has_cached_posts  // 有缓存时跳过评分
}
```

**Egress Sidecar 降级：**
```rust
let use_egress: bool = query.params.get(UseEgressSidecar);
let client = if use_egress { &self.egress_client } else { &self.phoenix_client };

let mut predictions = client.predict(cluster, request.clone()).await;

if predictions.is_err() && use_egress {
    tracing::debug!("Egress predict failed, falling back");
    predictions = self.phoenix_client.predict(cluster, request).await;  // 降级
}
```

### 5.4 Grox 近线处理

**异步任务执行：**
```python
# Grox 引擎支持异步任务调度
class Engine:
    async def start(self):
        # 启动 Kafka consumer
        # 启动任务调度器
        # 启动 gRPC server
```

**数据加载器：**
- `KafkaLoader`: 从 Kafka 消费事件
- `StratoLoader`: 从 Strato（X 内部存储）加载推文数据
- `MediaLoader`: 加载媒体内容（图片/视频）

---

## 6. 实验与部署实践

### 6.1 A/B 测试支持

**Feature Switches 机制：**
```rust
let recipient = RecipientBuilder::new()
    .user_id(proto_query.viewer_id)
    .country(&proto_query.country_code)
    .language(&proto_query.language_code)
    .client_app_id(proto_query.client_app_id as i64)
    .custom_string("datacenter", &self.datacenter)
    .custom_i64("account_age_days", days_since_creation(proto_query.viewer_id))
    .custom_bool("has_phone_number", has_phone_number);

let mut results = self.feature_switches.match_recipient(&recipient.build());
```

**Decider 实验控制：**
```rust
// 基于 decider 的集群切换
if let Some(decider) = &query.decider {
    match configured_cluster {
        PhoenixCluster::Experiment1Fou if decider.enabled("override_qf_use_lap7") => {
            return PhoenixCluster::Experiment1Lap7;
        }
        // ...
    }
}
```

### 6.2 模型热更新

**多集群部署：**
- `PhoenixCluster::Experiment1Fou`
- `PhoenixCluster::Experiment1Lap7`
- `PhoenixRetrievalCluster::Experiment1Fou`
- `PhoenixRetrievalCluster::Experiment1Lap7`

**新用户专用集群：**
```rust
let threshold: u64 = query.params.get(PhoenixRankerNewUserHistoryThreshold);
if threshold > 0 {
    let action_count = query.scoring_sequence.as_ref()
        .and_then(|s| s.metadata.as_ref())
        .map(|m| m.length)
        .unwrap_or(0);

    if action_count < threshold {
        return PhoenixCluster::parse(
            &query.params.get(PhoenixRankerNewUserInferenceClusterId)
        );
    }
}
```

### 6.3 推理性能优化

**压缩编码：**
```rust
// gRPC 响应压缩
.accept_compressed(CompressionEncoding::Gzip)
.accept_compressed(CompressionEncoding::Zstd)
.send_compressed(CompressionEncoding::Gzip)
.send_compressed(CompressionEncoding::Zstd)
```

**模型精度：**
```python
fprop_dtype: Any = jnp.bfloat16  # 使用 bfloat16 加速推理
```

**并行执行框架：**
```rust
// Candidate Pipeline 框架支持并行执行
// - Sources 并行执行
// - Hydrators 并行执行
// - Filters 和 Scorers 顺序执行
```

### 6.4 监控与追踪

**B3 分布式追踪：**
```rust
let b3_info = extract_b3_info(request.metadata());
let root_span = b3_info.root_span(info_span!(
    "request",
    endpoint = span_name,
    trace = %b3_info.trace_id_str,
    user = %query.user_id,
));
```

**Prometheus 指标：**
```rust
POST_STORE_USER_COUNT.set(user_count as f64);
POST_STORE_TOTAL_POSTS.set(total_posts as f64);
POST_STORE_POSTS_RETURNED.observe(posts.len() as f64);
```

---

## 7. 总结与启发

### 7.1 核心技术亮点

1. **零手工特征的 Transformer 架构**
   - 完全摒弃传统推荐系统中的手工特征工程
   - 通过用户交互序列直接学习相关性
   - 大幅简化数据管道和维护成本

2. **Candidate Isolation 设计**
   - 创新的注意力掩码机制
   - 保证分数的一致性和可缓存性
   - 支持高效的并行推理

3. **多目标联合预测**
   - 同时预测 19 种交互类型
   - 通过加权组合实现帕累托优化
   - 正负向行为的平衡建模

4. **In-Memory 实时存储（Thunder）**
   - Sub-millisecond 级别的 In-Network 内容检索
   - Kafka 驱动的实时事件消费
   - 自动化的数据生命周期管理

5. **Grok VLM 驱动的内容理解**
   - 利用大语言模型进行内容分类
   - 垃圾信息、安全策略的智能检测
   - 多模态内容的统一处理

### 7.2 架构优势

| 维度 | 优势 |
|------|------|
| **可维护性** | 无手工特征，模型自动学习 |
| **可扩展性** | Pipeline 框架支持灵活添加新组件 |
| **性能** | 内存存储 + bfloat16 + 并行推理 |
| **实验性** | Feature Switches + Decider 支持快速实验 |
| **安全性** | 多层过滤 + VLM 深度理解 |

### 7.3 潜在挑战

1. **冷启动问题**
   - 新用户缺少交互历史，依赖新用户专用模型集群
   - 新推文需要快速积累嵌入和交互信号

2. **计算成本**
   - Transformer 推理成本高
   - 需要大量 GPU 资源支持实时推理

3. **可解释性**
   - 深度学习模型的黑盒特性
   - 难以解释为什么推荐某条内容

4. **数据偏差**
   - 交互历史可能强化信息茧房
   - 需要多样性机制平衡

### 7.4 对行业的启发

1. **推荐系统的 Transformer 化**
   - X 的实践证明了 Transformer 在推荐场景的可行性
   - 预计更多平台会跟进类似架构

2. **端到端学习的极限**
   - 完全摒弃手工特征是激进但有效的选择
   - 需要足够的数据和计算资源支撑

3. **多模态理解的重要性**
   - Grox 管道展示了 VLM 在内容安全中的价值
   - 未来的推荐系统需要更强的内容理解能力

4. **实时性与质量的平衡**
   - Thunder 的内存存储 + Phoenix 的 GPU 推理
   - 分层架构实现延迟与质量的最优平衡

---

## 附录：关键文件索引

| 模块 | 文件路径 | 说明 |
|------|----------|------|
| **Phoenix Ranking** | `phoenix/recsys_model.py` | 排序模型核心实现 |
| **Phoenix Retrieval** | `phoenix/recsys_retrieval_model.py` | 召回模型实现 |
| **Thunder Store** | `thunder/posts/post_store.rs` | In-Network 内存存储 |
| **Home Mixer Server** | `home-mixer/server.rs` | 编排层 gRPC 服务 |
| **Phoenix Scorer** | `home-mixer/scorers/phoenix_scorer.rs` | Phoenix 评分调用 |
| **Ranking Scorer** | `home-mixer/scorers/ranking_scorer.rs` | 加权评分逻辑 |
| **Phoenix Source** | `home-mixer/sources/phoenix_source.rs` | OON 候选来源 |
| **Thunder Source** | `home-mixer/sources/thunder_source.rs` | In-Network 候选来源 |
| **VF Filter** | `home-mixer/filters/vf_filter.rs` | 可见性过滤 |
| **Grox Main** | `grox/main.py` | 内容理解服务入口 |
| **Spam Classifier** | `grox/classifiers/content/spam.py` | 垃圾信息检测 |

---

*分析完成时间：2026年5月22日*
*基于 X 开源代码库版本：2026年5月15日更新*
