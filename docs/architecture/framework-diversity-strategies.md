# Framework Diversity Exploration Strategies

**Version**: v1.0.0  
**Date**: 2025-11-07  
**Status**: 🧪 Architecture Exploration

---

## 📋 Document Purpose

This document records the **diversity exploration process** of the DAML-RAG meta-learning framework, including:
1. Technical decision-making discussions
2. Architecture solution comparisons
3. Future planning design思路
4. Framework application diversity practices

**Core Philosophy**: We are not designing a simple application, but exploring specific ways to apply the framework. The framework needs to maintain diversity, and all discussions can become part of the framework's diversity planning.

---

## 🎯 Exploration Theme 1: Data Preloading Strategy

### Problem Background

**Generic Scenario**:
- In multi-agent orchestration systems, certain contextual data (e.g., user configuration, system state) needs to be accessed repeatedly by multiple tools
- Orchestrator DAG dependencies may call the same data retrieval tools repeatedly
- While orchestrators provide caching mechanisms (e.g., TTL cache), they lack intelligent preloading capabilities

### Technical Solution Comparison

| Solution | Strategy | Pros | Cons | Use Case |
|----------|----------|------|------|----------|
| **1a** | Session-level preloading | Simple, easy to implement | Cannot handle intra-session data updates | Short sessions (<5min) |
| **2a** | Preload before each tool call | Data always fresh | Frequent calls, poor performance | High real-time requirements |
| **2b** | Long-term cache (1 hour) | Excellent performance | Data may be stale | Data rarely changes |
| **2c** | Orchestrator intelligent preloading (on-demand) | Balance performance and accuracy | Complex implementation | Production environments ✅ |
| **3a** | Failure degradation | Strong robustness | - | All scenarios ✅ |
| **3b** | Immediate failure | Simple | Poor user experience | Not recommended |
| **3c** | Use stale cache | Good fault tolerance | May return old data | Specific scenarios |

### Recommended Approach

**Strategy Combination**:
- ✅ **Strategy 1a**: Session-level preloading - fetch commonly used data once at request start
- ✅ **Strategy 2c**: Orchestrator intelligent preloading - detect data retrieval tools in dependency chain, prioritize using preloaded data
- ✅ **Strategy 3a**: Failure degradation - if preloading fails, tools fetch themselves (maintain robustness)

**Expected Effects**:
- Duplicate data call count reduced by 50%+
- Response latency reduced by 10%+
- Maintain robustness (degradation strategy)

**Generic Implementation Pattern**:
```python
# Domain orchestrator implementation example
preloaded_data = None
if context and 'preloaded_context_data' in context:
    preloaded_data = context['preloaded_context_data']
    
    # Validate preloaded data validity
    if preloaded_data and isinstance(preloaded_data, dict):
        logger.info("✅ [Intelligent Preloading] Valid preloaded data detected")
        
        # Remove data retrieval task (if exists)
        tasks = [task for task in tasks if task.task_id != 'get_context_data']
    else:
        # Invalid preloaded data, degrade to standard DAG flow
        logger.warning("⚠️ [Intelligent Preloading] Invalid preloaded data, degrading to standard DAG flow")
        preloaded_data = None
```

### Future Optimization Directions

1. **Dynamic TTL adjustment**: Adjust cache duration based on user activity
2. **Differential updates**: Only update changed fields, reduce data transfer
3. **Predictive preloading**: Predict needed data based on user behavior patterns

---

## 🎯 Exploration Theme 2: LLM Selection Logic Optimization

### Problem Background

**Phenomenon**:
- Current implementation: `_should_use_teacher_model()` uses hardcoded keyword list to determine complexity
- Limitation: Limited keyword list (only 14), cannot cover all complex scenarios

### Technical Solution Comparison

| Solution | Implementation | Pros | Cons | Accuracy | Cost |
|----------|---------------|------|------|----------|------|
| **Hardcoded keywords** | String matching | Simple, fast, no dependencies | Low coverage, rigid rules | 60-70% | Free |
| **Rule engine** | Regex + weights | Flexible, configurable | High maintenance cost | 70-75% | Free |
| **BGE vector model** | Semantic similarity | High accuracy, adaptive | Model loading cost | 80-90% ✅ | Local free |
| **GPT-4 classification** | LLM API call | Highest accuracy | High API cost, high latency | 95%+ | $0.01/call |
| **Fine-tuned classifier** | Train small model | Accurate + fast | Needs labeled data | 85-90% | One-time training cost |

### Final Decision

**Adopted Solutions**:
- ✅ **BGE vector model** (`BAAI/bge-base-zh-v1.5` or equivalent): Calculate query vectors, classify based on cosine similarity
- ✅ **Retain hardcoded keywords as fallback**: Use keyword matching when model loading fails

**Classification Logic**:
1. Pre-define "complex query" vector library (training plans, rehabilitation plans, nutritional design, etc.)
2. Calculate cosine similarity between current query and complex queries
3. Decision rules:
   - Similarity **≥ 0.7** → Teacher model (DeepSeek or similar) - Complex query
   - Similarity **< 0.5** → Student model (Ollama or similar) - Simple query
   - **0.5-0.7** → Few-Shot judgment (medium complexity)

**Complex Query Vector Library Example**:
```python
COMPLEX_QUERY_EXAMPLES = [
    # Domain-specific complex queries
    "Design a comprehensive solution considering condition X with restrictions Y",
    "Create a detailed plan for goal Z with specific requirements A, B, and C",
    
    # Multi-step reasoning queries
    "How to handle situation X while addressing concern Y?",
    "What's the best approach for Z considering factors A and B?",
    
    # Deep analysis queries
    "Analyze the relationship between X and Y, provide recommendations",
    "Evaluate options A, B, C for scenario Z with trade-offs",
    
    # ... (customize 15-20 examples for your domain)
]
```

**Implementation Effects**:
- LLM selection accuracy improved by 20%+ (60% → 80%+)
- Support natural language variants ("design a plan" vs "create a strategy")
- Retain fallback strategy (model loading fails → keyword matching)

**Code Implementation Example**:
```python
# chat_service.py or equivalent
if self.query_complexity_classifier:
    try:
        is_complex, similarity, reason = self.query_complexity_classifier.classify_complexity(
            message
        )
        
        if is_complex:
            logger.info(f"✅ [BGE Classification] Complex query (similarity={similarity:.2f}) → Teacher model")
            return True
        else:
            logger.info(f"✅ [BGE Classification] Simple query (similarity={similarity:.2f}) → Student model")
            return False
            
    except Exception as e:
        logger.warning(f"⚠️ [BGE Classification] Classification failed: {e}, degrading to keyword matching")
        # Fallback to hardcoded keywords
```

### Future Optimization Directions

1. **Online learning**: Dynamically update complex query vector library based on user feedback
2. **Multi-modal classification**: Combine query text + user history + temporal context
3. **A/B testing**: Compare effects of different similarity thresholds
4. **Fine-tuned small model**: Train dedicated classifier after data accumulation (85-90% accuracy + faster speed)

---

## 🎯 Exploration Theme 3: Framework Core Algorithm Preservation

### Thompson Sampling + Contextual Bandit

**Confirmed Decision**:
- ✅ **Thompson Sampling + Contextual Bandit** retained in `daml-rag-framework`
- **Reason**: This is the framework's core adaptive learning capability, verified effective

**Theoretical Foundation**:
1. **Multi-Armed Bandit (MAB)**: Balance exploration and exploitation
2. **Thompson Sampling**: Bayesian MAB algorithm, adaptive exploration
3. **Contextual Bandit**: MAB considering context (query vectors)
4. **Beta Distribution**: Model success/failure binomial distribution

**Workflow**:
```
User Query
    ↓
Retrieve Similar Historical Cases (Contextual)
    ↓
Statistics Tool Chain Performance (Beta distribution parameters: α successes, β failures)
    ↓
Thompson Sampling Selects Optimal Tool Chain
    ↓
ε-greedy Explores Insufficiently Tried Tool Chains
    ↓
Update Beta Distribution Based on User Feedback
```

**Mathematical Principle**:
```
For tool chain i, maintain Beta distribution Beta(α_i, β_i)
- α_i: Success count + 1 (prior)
- β_i: Failure count + 1 (prior)

Thompson Sampling Steps:
1. For each tool chain i, sample θ_i from Beta(α_i, β_i)
2. Select tool chain with maximum θ_i
3. Update based on feedback:
   - If successful (reward >= 4.0): α_i += 1
   - If failed (reward < 4.0): β_i += 1
```

**Retention Reasons**:
- Adaptive learning capability is core framework value
- No need for manual data annotation
- Continuously optimize tool selection strategy
- Applicable to various vertical domain application scenarios

---

## 🎯 Exploration Theme 4: Data Cleaning and Fine-tuning Architecture Planning

### Planning Status

⚠️ **Architecture design stage only, not immediately implemented**

### Data Cleaning Pipeline

**Trigger Conditions**:
- Weekly scheduled task
- **OR** conversation count reaches threshold

**Cleaning Rules**:
```python
# 1. Filter low-quality conversations
conversations = db.query("""
    SELECT * FROM best_practices
    WHERE reward >= 3.0  -- Filter conversations with rating < 3.0
    AND created_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
""")

# 2. Deduplicate queries
deduplicated = deduplicate_by_similarity(conversations, threshold=0.95)

# 3. Label tool chains
for conv in deduplicated:
    conv['tool_chain'] = extract_tool_chain(conv['tools_used'])
    conv['domain'] = classify_domain(conv['query'])  # Domain-specific

# 4. Store in training database
save_to_training_db(deduplicated)
```

**Storage Location**:
- Database `best_practices` table
- New fields: `tool_chain`, `domain`, `is_cleaned`

**Detailed Cleaning Rules**:
| Rule | Condition | Action |
|------|-----------|--------|
| Low-quality conversation | Rating < 3.0 | Discard |
| Duplicate query | Similarity > 0.95 | Keep highest rated |
| Incomplete conversation | Missing tool chain/response | Discard |
| Test data | `user_id` = "test_*" | Discard |
| Error response | Contains error keywords | Mark as negative sample |

### Fine-tuning Trigger Conditions

**Data Volume Conditions**:
- ✅ High-quality conversation accumulation **≥ 5000** (rating ≥ 4.5)
- ✅ Cover main scenarios **≥ 80%**
- ✅ Few-Shot retrieval hit rate **< 60%** (indicates insufficient historical cases)

**Scenario Coverage Calculation**:
```python
coverage = {
    "domain_a": len([c for c in conversations if c['domain'] == 'domain_a']) / total,
    "domain_b": len([c for c in conversations if c['domain'] == 'domain_b']) / total,
    "domain_c": len([c for c in conversations if c['domain'] == 'domain_c']) / total,
    "domain_d": len([c for c in conversations if c['domain'] == 'domain_d']) / total
}

# Require at least 20% coverage for each scenario
is_covered = all(v >= 0.20 for v in coverage.values())
```

### Fine-tuning Architecture

**Base Model**:
- Choose appropriate base model for your domain
- Parameters: Model size
- Context length: Context window

**Fine-tuning Method**:
- **LoRA** (Low-Rank Adaptation)
- Rank: 8-16
- Alpha: 16-32
- Dropout: 0.05
- Training epochs: 3-5

**Training Data Format**:
```json
{
    "query": "User query with domain-specific context",
    "tool_chain": [
        "tool1",
        "tool2",
        "tool3"
    ],
    "response": "Professional domain-specific response...",
    "rating": 4.8
}
```

**Deployment Strategy**:
```
Data Accumulation Meets Criteria
    ↓
Train LoRA Adapter (GPU Server)
    ↓
Offline Validation (score >= baseline model)
    ↓
A/B Testing (50% traffic, 7 days)
    ↓
Effect Evaluation (user satisfaction, tool chain accuracy)
    ↓
Gradual Rollout (70% → 90% → 100%)
    ↓
Full Replacement (keep original model as fallback)
```

### Evaluation Metrics

| Metric | Baseline | Target | Importance |
|--------|----------|--------|------------|
| **User Satisfaction** | 4.2 | 4.5+ | ⭐⭐⭐⭐⭐ |
| **Tool Chain Accuracy** | 75% | 85%+ | ⭐⭐⭐⭐⭐ |
| **Response Time (P95)** | 3.5s | < 4.0s | ⭐⭐⭐⭐ |
| **Few-Shot Hit Rate** | 55% | 70%+ | ⭐⭐⭐ |
| **API Cost Reduction** | Baseline | 30%+ | ⭐⭐⭐⭐ |

### Implementation Timeline

- ✅ **Data Cleaning**: Start 2 weeks after framework stabilization
- ✅ **Fine-tuning**: After data accumulation meets criteria (estimated 3-6 months)

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Insufficient data quality | Medium | High | Strict cleaning rules + manual review |
| Fine-tuning overfitting | Medium | Medium | Early stopping + validation monitoring |
| Performance degradation | Low | High | A/B testing + rollback plan |
| Insufficient scenario coverage | Medium | Medium | Actively generate supplementary data |
| GPU resource shortage | Low | Medium | Rent cloud GPU (AWS/Azure) |

---

## 📊 Framework Diversity Practice Summary

### Core Principles

1. **Maintain Openness**: Framework design allows multiple implementation methods
2. **Record Decision Process**: All technical solution comparisons are valuable
3. **Iterative Optimization**: Implement basic solutions first, then optimize based on data
4. **Data-Driven**: Verify architectural assumptions with real data

### Implemented Optimizations

| Optimization | Version | Status | Effect |
|--------------|---------|--------|--------|
| Intelligent Preloading | v3.2.0 | ✅ Deployed | MCP calls -50% |
| BGE Classifier | v3.2.0 | ✅ Deployed | LLM selection accuracy +20% |
| Thompson Sampling | v2.0.0 | ✅ Production | Adaptive learning |
| Data Cleaning Architecture | Planning | 📋 Design stage | Expected in 3 months |
| Fine-tuning Architecture | Planning | 📋 Design stage | Expected in 6 months |

### Future Exploration Directions

1. **Multi-modal RAG**: Support image/video retrieval
2. **Distributed Orchestration**: Support cross-server tool calls
3. **Reinforcement Learning Optimization**: Replace Thompson Sampling
4. **Federated Learning**: Multi-user data aggregation training

---

## 🔗 Related Documents

- **Data Cleaning and Fine-tuning Architecture**: [data-cleaning-and-finetuning.md](./data-cleaning-and-finetuning.md)
- **Framework Theory**: [../theory/FRAMEWORK_OVERVIEW.md](../theory/FRAMEWORK_OVERVIEW.md)

---

**Maintainer**: DAML-RAG Framework Team  
**Last Review**: 2025-11-07

<div align="center">
<strong>🎯 Explore Framework Diversity · 📊 Data-Driven Optimization · 🚀 Continuous Evolution</strong>
</div>

---

## 📖 Application-Specific Implementation

This diversity exploration strategy has been implemented in:

- **BUILD_BODY Fitness Application**: See [FRAMEWORK_DIVERSITY_EXPLORATION.md](../../build_body/mcp-servers/meta-learning-mcp/docs/07-综合架构分析/FRAMEWORK_DIVERSITY_EXPLORATION.md) for fitness-specific exploration details.

To adapt these strategies to your domain:
1. Define domain-specific complex query examples
2. Customize data cleaning rules for your domain
3. Adjust evaluation metrics based on your use case
4. Select appropriate similarity thresholds through A/B testing

