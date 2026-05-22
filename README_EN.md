# X (Twitter) Open-Source Recommendation Algorithm Deep Analysis

> In-depth analysis of X's "For You Feed" recommendation algorithm open-sourced in May 2026

**English** | **[中文](README_CN.md)**

---

## Table of Contents

1. [Overall Architecture & Flow](#1-overall-architecture--flow)
2. [Retrieval Strategy Details](#2-retrieval-strategy-details)
3. [Ranking Model Analysis](#3-ranking-model-analysis)
4. [Negative Feedback & Safety Layer](#4-negative-feedback--safety-layer)
5. [Real-time & Nearline Processing](#5-real-time--nearline-processing)
6. [Experimentation & Deployment](#6-experimentation--deployment)
7. [Summary & Insights](#7-summary--insights)

---

## 1. Overall Architecture & Flow

### 1.1 System Overview

X's "For You" feed algorithm implements a classic **two-stage recommendation architecture**: Retrieval + Ranking, supplemented by multi-layer filtering. The system uses a hybrid tech stack of Rust and Python/JAX, with core services communicating via gRPC.

**Key Tech Stack:**
- **Orchestration Layer**: Rust (Home Mixer)
- **In-Network Storage**: Rust (Thunder)
- **ML Model Inference**: Python/JAX (Phoenix)
- **Content Understanding**: Python (Grox, based on Grok VLM)

### 1.2 Three-Stage Architecture Implementation

While the codebase primarily showcases two core stages (retrieval and ranking), the complete request flow includes the following stages:

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

### 1.3 Data Flow & Component Collaboration

| Component | Responsibility | Input | Output | Tech Stack |
|-----------|----------------|-------|--------|------------|
| **Home Mixer** | Orchestration | gRPC Request | Ranked Feed | Rust |
| **Thunder** | In-Network Candidates | Following List | Recent Posts | Rust (in-memory) |
| **Phoenix Retrieval** | Out-of-Network Candidates | User Embedding | Similar Posts | JAX |
| **Phoenix Ranking** | Fine-grained Scoring | Candidates + User History | Engagement Probs | JAX |
| **Grox** | Content Understanding | Post Content | Safety Labels | Python + Grok VLM |

### 1.4 Core Design Philosophy

**No Hand-Engineered Features:**
> "We have eliminated every single hand-engineered feature and most heuristics from the system. The Grok-based transformer does all the heavy lifting."

This is X's most radical design decision—completely relying on the Transformer model to learn relevance from user interaction history, abandoning traditional hand-crafted feature engineering.

---

## 2. Retrieval Strategy Details

### 2.1 Main Retrieval Channels

| Channel Name | Type | Candidate Volume | Latency Target | Description |
|--------------|------|------------------|----------------|-------------|
| **Thunder** | In-Network | ~500-1000 | < 5ms | Recent posts from followed users, in-memory storage |
| **Phoenix Retrieval** | Out-of-Network | ~200-500 | ~50ms | Two-tower model ANN search |
| **Phoenix MoE** | Out-of-Network | Variable | ~50ms | Mixture of Experts variant |
| **Phoenix Topics** | Out-of-Network | Variable | ~50ms | Topic-based retrieval |
| **Cached Posts** | Cache | Variable | < 1ms | Cached results from previous requests |
| **Ads** | Advertising | Variable | ~50ms | Ad candidate pool |
| **Who to Follow** | User Recommendation | Variable | ~30ms | Recommended users to follow |

### 2.2 Thunder (In-Network Retrieval)

**Architecture Features:**
- **In-Memory Storage**: Uses Rust's `DashMap` for concurrent-safe in-memory hash tables
- **Kafka Consumption**: Real-time consumption of tweet create/delete events
- **Layered Storage**: Original tweets, replies/retweets, and video tweets stored separately
- **Auto Cleanup**: Periodic deletion of old posts beyond retention period (default 2 days)

**Data Structure:**
```rust
PostStore {
    posts: DashMap<i64, LightPost>,                    // Indexed by post_id
    original_posts_by_user: DashMap<i64, VecDeque<TinyPost>>,  // Original tweets
    secondary_posts_by_user: DashMap<i64, VecDeque<TinyPost>>, // Replies/retweets
    video_posts_by_user: DashMap<i64, VecDeque<TinyPost>>,     // Video tweets
}
```

**Performance Metrics:**
- Supports sub-millisecond level queries
- Each user retains up to N original tweets, M replies
- Configurable request timeout (default: no timeout)

### 2.3 Phoenix Retrieval (Out-of-Network Retrieval)

**Two-Tower Architecture:**

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

**Key Technical Details:**

1. **Hash-Based Embeddings**:
   - Users/posts/authors each use 2 hash functions
   - Vocabulary size: 1,000,000 (per entity)
   - Multiple hash embeddings combined through linear projection

2. **Model Configuration (Mini Version)**:
   - Embedding dimension: 128
   - Transformer layers: 4
   - Attention heads: 4
   - Key size: 32

3. **Retrieval Flow**:
   - User history sequence → Transformer → User embedding
   - Candidate corpus → Pre-computed candidate embeddings
   - Dot product similarity → Top-K retrieval (default 200)

### 2.4 Diversity Balancing Mechanism

**Out-of-Network Weight Adjustment:**
```rust
// Base weight factor for OON content
let oon_weight_factor = params.get(OonWeightFactor);

// Special handling for new users
if is_eligible_new_user {
    NEW_USER_OON_WEIGHT_FACTOR  // Give new users more OON content
}

// Special handling for topic requests
if !query.topic_ids.is_empty() {
    return params.get(TopicOonWeightFactor);
}
```

**Author Diversity Decay:**
```rust
fn diversity_multiplier(decay_factor: f64, floor: f64, position: usize) -> f64 {
    (1.0 - floor) * decay_factor.powf(position as f64) + floor
}
```
The Nth tweet from the same author has its score multiplied by a decay factor, ensuring the feed doesn't over-concentrate on a few authors.

---

## 3. Ranking Model Analysis

### 3.1 Phoenix Ranking Model Architecture

The Phoenix ranking model is based on the **Grok-1 Transformer** architecture, specifically optimized for recommendation scenarios. Its core innovation is **Candidate Isolation**.

**Model Architecture:**

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

**Candidate Isolation Attention Mask:**

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

**Core Design Advantages:**
- Candidate tweet scores don't depend on other tweets in the batch
- Ensures score consistency and cacheability
- Supports efficient parallel inference

### 3.2 Feature System

**1. User-Side Features:**
- `user_hashes`: Multi-hash embeddings of user ID (2 hash functions)
- `user_ip_embeddings`: IP address embedding (optional)
- `engagement_history`: User interaction sequence (last 127 items)

**2. History Sequence Features (per interaction):**
- `history_post_hashes`: Tweet ID hashes
- `history_author_hashes`: Author ID hashes
- `history_actions`: Interaction types (like, reply, repost, etc.)
- `history_product_surface`: Product surface (iOS, Android, Web)
- `history_continuous_actions`: Continuous values (e.g., dwell time)

**3. Candidate Tweet Features:**
- `candidate_post_hashes`: Tweet ID hashes
- `candidate_author_hashes`: Author ID hashes
- `candidate_product_surface`: Product surface
- `post_age_bucket`: Tweet age bucket (1-hour granularity, max 80 hours)

**4. Cross Features:**
- Implicitly learned through Transformer self-attention mechanism
- Interactions between user history and candidate tweets

### 3.3 Multi-Objective Prediction

The model simultaneously predicts probabilities for **19 interaction types**:

| Category | Prediction Target | Description |
|----------|-------------------|-------------|
| **Positive** | `favorite` | Like probability |
| | `reply` | Reply probability |
| | `repost` | Retweet probability |
| | `quote` | Quote probability |
| | `click` | Click probability |
| | `profile_click` | Profile click probability |
| | `video_view` | Video view probability |
| | `photo_expand` | Photo expand probability |
| | `share` | Share probability |
| | `dwell` | Dwell probability |
| | `follow_author` | Follow author probability |
| **Negative** | `not_interested` | Not interested probability |
| | `block_author` | Block author probability |
| | `mute_author` | Mute author probability |
| | `report` | Report probability |

### 3.4 Weighted Scoring Formula

**Final Score Calculation:**

```rust
Final_Score = Σ (weight_i × P(action_i))

// Positive behavior weights (example)
favorite: weight × P(like)
reply: weight × P(reply)
retweet: weight × P(repost)
quote: weight × P(quote)
click: weight × P(click)
dwell: weight × P(dwell)
follow_author: weight × P(follow)

// Negative behavior weights (negative values, lower score)
not_interested: weight × P(not_interested)
block_author: weight × P(block_author)
mute_author: weight × P(mute_author)
report: weight × P(report)
```

**Score Offset Processing:**
```rust
fn offset_score(combined_score: f64, w: &ScoringWeights) -> f64 {
    if combined_score < 0.0 {
        (combined_score + w.negative_sum) / w.total_sum * NEGATIVE_SCORES_OFFSET
    } else {
        combined_score + NEGATIVE_SCORES_OFFSET
    }
}
```

### 3.5 Inference Mechanism

**Model Configuration (Production Estimate):**
| Parameter | Mini Version | Production (Estimate) |
|-----------|--------------|----------------------|
| Embedding Dimension | 128 | 512-1024 |
| Transformer Layers | 4 | 12-24 |
| Attention Heads | 4 | 16-32 |
| History Sequence Length | 127 | 127-255 |
| Candidate Sequence Length | 64 | 64-128 |
| Inference Precision | bfloat16 | bfloat16 |

**Real-time Feature Processing:**
- User interaction history passed in real-time via `scoring_sequence`
- Tweet age calculated based on `impr_ts` and `post_creation_ts` in real-time
- Continuous features like dwell time projected to embedding space via MLP

**Continuous Value Normalization:**
```python
def normalize_continuous_value(values, config):
    values_clamped = jnp.clip(values, 0.0, config.norm_scale)
    if config.use_log:
        return jnp.log1p(values_clamped) / jnp.log1p(config.norm_scale)
    else:
        return values_clamped / config.norm_scale
```

---

## 4. Negative Feedback & Safety Layer

### 4.1 Explicit Negative Feedback Handling

**Negative Feedback in Pre-Scoring Filters:**

| Filter | Function | Location |
|--------|----------|----------|
| `MutedKeywordFilter` | Remove posts containing user's muted keywords | Pre-Scoring |
| `AuthorSocialgraphFilter` | Remove posts from blocked/muted users | Pre-Scoring |
| `PreviouslySeenPostsFilter` | Remove posts user has already seen | Pre-Scoring |

**Negative Feedback Modeling in Scoring:**

The model explicitly predicts negative behavior probabilities:
- `P(not_interested)` - Not interested
- `P(block_author)` - Block author
- `P(mute_author)` - Mute author
- `P(report)` - Report

These probabilities are multiplied by negative weights and added to the total score, automatically suppressing content the user would likely反感.

### 4.2 Implicit Negative Feedback Modeling

**Dwell Time Analysis:**

The code supports continuous value features, with special attention to dwell time:
```python
# Dwell time embedding in history
dwell_values = batch.history_continuous_actions[:, :, 1]  # index 1 = dwell_time
history_continuous_embeddings = self._project_continuous_value_to_embedding(
    dwell_values,
    config.emb_size,
    "history_dwell_time",
    config.continuous_action_config.norm_config,
    config.continuous_action_hidden_dim,
)
```

**"Not Dwelled" Signal:**
```rust
// Scoring weights include not_dwelled
not_dwelled: params.get(NotDwelledWeight)
```
Posts that are quickly scrolled past (no effective dwell) are marked, and the model learns this implicit negative feedback.

### 4.3 Content Safety Layer

**Grox Content Understanding Pipeline:**

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
│  │  Spam Detection    │  ← Low-follower user spam detection    │
│  │  (Grok VLM)        │                                        │
│  └─────────┬──────────┘                                        │
│            │                                                    │
│            ▼                                                    │
│  ┌────────────────────┐                                        │
│  │  Safety Screening  │  ← PTOS policy compliance check        │
│  │  (Multi-classifier)│                                        │
│  └─────────┬──────────┘                                        │
│            │                                                    │
│            ▼                                                    │
│  ┌────────────────────┐                                        │
│  │  Content Category  │  ← Tweet categorization (sports/news/  │
│  │  Classification    │     entertainment, etc.)                │
│  └─────────┬──────────┘                                        │
│            │                                                    │
│            ▼                                                    │
│  ┌────────────────────┐                                        │
│  │  Multimodal        │  ← Image-text multimodal embedding     │
│  │  Embedding         │     generation                         │
│  └─────────┬──────────┘                                        │
│            │                                                    │
│            ▼                                                    │
│  ┌────────────────────┐                                        │
│  │  Annotations Sink  │  ← Write results to storage            │
│  └────────────────────┘                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Safety Filtering Implementation:**

```rust
// VF (Visibility Filtering) filter
fn should_drop(reason: &Option<FilteredReason>) -> bool {
    match reason {
        Some(FilteredReason::SafetyResult(safety_result)) => {
            matches!(safety_result.action, Action::Drop(_))
        }
        Some(_) => true,  // Other filtering reasons also remove
        None => false,
    }
}
```

**Brand Safety:**
- `AdsBrandSafetyHydrator`: Ad brand safety marking
- `AdsBrandSafetyVfHydrator`: Ad visibility filtering safety

### 4.4 Safety Layer Intervention Points

| Stage | Safety Mechanism | Description |
|-------|------------------|-------------|
| **Pre-Scoring** | `AuthorSocialgraphFilter` | Block/muted user content |
| **Pre-Scoring** | `MutedKeywordFilter` | Muted keyword filtering |
| **Post-Selection** | `VFFilter` | Visibility filtering (spam/violence/gore) |
| **Async** | Grox Pipeline | Deep content understanding (VLM) |

---

## 5. Real-time & Nearline Processing

### 5.1 Thunder Real-time Pipeline

**Kafka Consumption Architecture:**
```rust
// Thunder consumes tweet events in real-time
- Consumes topics: tweet_create, tweet_delete
- In-memory storage, no disk I/O needed
- Supports automatic trim of expired tweets
```

**Retention Policy:**
- Default retention period: 2 days (172,800 seconds)
- Periodic auto-cleanup (configurable interval)
- Per-user dimension trimming, limiting max tweets per user

### 5.2 Real-time Feature Calculation

**Tweet Age Real-time Calculation:**
```python
def compute_post_age_bucket(impr_ts_sec, post_creation_ts_sec, granularity_mins=60):
    post_age_minutes = (impr_ts_sec - post_creation_ts_sec) // 60
    bucket = (post_age_minutes // granularity_mins) + 1
    bucket = jnp.clip(bucket, 0, overflow_bucket)
    return bucket.astype(jnp.int32)
```

**Dwell Time Real-time Projection:**
```python
# Real-time normalization → MLP projection → embedding
values_normalized = normalize_continuous_value(values, config.norm_config)
hidden = jax.nn.gelu(jnp.dot(values_expanded, proj1))
embedding = jnp.dot(hidden, proj2)
```

### 5.3 Caching & Degradation Strategy

**Caching Mechanism:**
```rust
// Cache check in PhoenixScorer
fn enable(&self, query: &ScoredPostsQuery) -> bool {
    !query.has_cached_posts  // Skip scoring when cache exists
}
```

**Egress Sidecar Degradation:**
```rust
let use_egress: bool = query.params.get(UseEgressSidecar);
let client = if use_egress { &self.egress_client } else { &self.phoenix_client };

let mut predictions = client.predict(cluster, request.clone()).await;

if predictions.is_err() && use_egress {
    tracing::debug!("Egress predict failed, falling back");
    predictions = self.phoenix_client.predict(cluster, request).await;  // Degrade
}
```

### 5.4 Grox Nearline Processing

**Async Task Execution:**
```python
# Grox engine supports async task scheduling
class Engine:
    async def start(self):
        # Start Kafka consumer
        # Start task scheduler
        # Start gRPC server
```

**Data Loaders:**
- `KafkaLoader`: Consume events from Kafka
- `StratoLoader`: Load tweet data from Strato (X's internal storage)
- `MediaLoader`: Load media content (images/videos)

---

## 6. Experimentation & Deployment

### 6.1 A/B Testing Support

**Feature Switches Mechanism:**
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

**Decider Experiment Control:**
```rust
// Decider-based cluster switching
if let Some(decider) = &query.decider {
    match configured_cluster {
        PhoenixCluster::Experiment1Fou if decider.enabled("override_qf_use_lap7") => {
            return PhoenixCluster::Experiment1Lap7;
        }
        // ...
    }
}
```

### 6.2 Model Hot Updates

**Multi-Cluster Deployment:**
- `PhoenixCluster::Experiment1Fou`
- `PhoenixCluster::Experiment1Lap7`
- `PhoenixRetrievalCluster::Experiment1Fou`
- `PhoenixRetrievalCluster::Experiment1Lap7`

**New User Dedicated Cluster:**
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

### 6.3 Inference Performance Optimization

**Compression Encoding:**
```rust
// gRPC response compression
.accept_compressed(CompressionEncoding::Gzip)
.accept_compressed(CompressionEncoding::Zstd)
.send_compressed(CompressionEncoding::Gzip)
.send_compressed(CompressionEncoding::Zstd)
```

**Model Precision:**
```python
fprop_dtype: Any = jnp.bfloat16  # Use bfloat16 to accelerate inference
```

**Parallel Execution Framework:**
```rust
// Candidate Pipeline framework supports parallel execution
// - Sources execute in parallel
// - Hydrators execute in parallel
// - Filters and Scorers execute sequentially
```

### 6.4 Monitoring & Tracing

**B3 Distributed Tracing:**
```rust
let b3_info = extract_b3_info(request.metadata());
let root_span = b3_info.root_span(info_span!(
    "request",
    endpoint = span_name,
    trace = %b3_info.trace_id_str,
    user = %query.user_id,
));
```

**Prometheus Metrics:**
```rust
POST_STORE_USER_COUNT.set(user_count as f64);
POST_STORE_TOTAL_POSTS.set(total_posts as f64);
POST_STORE_POSTS_RETURNED.observe(posts.len() as f64);
```

---

## 7. Summary & Insights

### 7.1 Core Technical Highlights

1. **Zero Hand-Engineered Features Transformer Architecture**
   - Completely abandons traditional hand-crafted feature engineering in recommendation systems
   - Learns relevance directly from user interaction sequences
   - Significantly simplifies data pipelines and maintenance costs

2. **Candidate Isolation Design**
   - Innovative attention mask mechanism
   - Ensures score consistency and cacheability
   - Supports efficient parallel inference

3. **Multi-Objective Joint Prediction**
   - Simultaneously predicts 19 interaction types
   - Achieves Pareto optimization through weighted combination
   - Balanced modeling of positive and negative behaviors

4. **In-Memory Real-time Storage (Thunder)**
   - Sub-millisecond level In-Network content retrieval
   - Kafka-driven real-time event consumption
   - Automated data lifecycle management

5. **Grok VLM-Driven Content Understanding**
   - Uses large language models for content classification
   - Intelligent detection of spam and safety policy violations
   - Unified processing of multimodal content

### 7.2 Architectural Advantages

| Dimension | Advantage |
|-----------|-----------|
| **Maintainability** | No hand-engineered features, model learns automatically |
| **Scalability** | Pipeline framework supports flexible addition of new components |
| **Performance** | In-memory storage + bfloat16 + parallel inference |
| **Experimentation** | Feature Switches + Decider support rapid experimentation |
| **Security** | Multi-layer filtering + VLM deep understanding |

### 7.3 Potential Challenges

1. **Cold Start Problem**
   - New users lack interaction history, rely on new user dedicated model clusters
   - New tweets need to quickly accumulate embeddings and interaction signals

2. **Computational Cost**
   - Transformer inference cost is high
   - Requires substantial GPU resources for real-time inference

3. **Interpretability**
   - Black-box nature of deep learning models
   - Difficult to explain why certain content is recommended

4. **Data Bias**
   - Interaction history may reinforce filter bubbles
   - Requires diversity mechanisms for balance

### 7.4 Industry Insights

1. **Transformer-ification of Recommendation Systems**
   - X's practice proves the feasibility of Transformers in recommendation scenarios
   - Expected that more platforms will follow similar architectures

2. **Limits of End-to-End Learning**
   - Completely abandoning hand-engineered features is a radical but effective choice
   - Requires sufficient data and computational resources to support

3. **Importance of Multimodal Understanding**
   - Grox pipeline demonstrates the value of VLM in content safety
   - Future recommendation systems need stronger content understanding capabilities

4. **Balancing Real-time and Quality**
   - Thunder's in-memory storage + Phoenix's GPU inference
   - Layered architecture achieves optimal balance between latency and quality

---

## Appendix: Key File Index

| Module | File Path | Description |
|--------|-----------|-------------|
| **Phoenix Ranking** | `phoenix/recsys_model.py` | Core ranking model implementation |
| **Phoenix Retrieval** | `phoenix/recsys_retrieval_model.py` | Retrieval model implementation |
| **Thunder Store** | `thunder/posts/post_store.rs` | In-Network in-memory storage |
| **Home Mixer Server** | `home-mixer/server.rs` | Orchestration layer gRPC service |
| **Phoenix Scorer** | `home-mixer/scorers/phoenix_scorer.rs` | Phoenix scoring invocation |
| **Ranking Scorer** | `home-mixer/scorers/ranking_scorer.rs` | Weighted scoring logic |
| **Phoenix Source** | `home-mixer/sources/phoenix_source.rs` | OON candidate source |
| **Thunder Source** | `home-mixer/sources/thunder_source.rs` | In-Network candidate source |
| **VF Filter** | `home-mixer/filters/vf_filter.rs` | Visibility filtering |
| **Grox Main** | `grox/main.py` | Content understanding service entry |
| **Spam Classifier** | `grox/classifiers/content/spam.py` | Spam detection |

---

*Analysis completed: May 22, 2026*
*Based on X open-source codebase version: May 15, 2026 update*
