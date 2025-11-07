# Data Cleaning and Fine-tuning Architecture

**Version**: v1.0.0  
**Date**: 2025-11-07  
**Status**: 📋 Architecture Planning Stage (Not Implemented)  

---

## ⚠️ Important Notice

This document is an **architecture planning document** and is **not immediately implemented**.

This is a **generic framework module** that can be applied to any DAML-RAG application.

**Implementation Timeline**:
- **Data Cleaning**: Start 2 weeks after framework stabilization
- **Fine-tuning**: After data accumulation meets criteria (estimated 3-6 months)

---

## 📋 Document Purpose

This document defines the **generic data cleaning and fine-tuning architecture** for DAML-RAG framework applications. It is **application-agnostic** and can be adapted to:

- Fitness coaching systems
- Customer service chatbots
- Educational tutoring systems
- Medical consultation systems
- Any domain-specific AI assistant

---

## 1. Data Cleaning Pipeline

### 1.1 Trigger Conditions

```python
# Trigger conditions (any of the following)
triggers = {
    "weekly_schedule": "Execute every Sunday at 2:00 AM",
    "conversation_threshold": "When conversation count reaches 1000"
}
```

### 1.2 Cleaning Rules

#### Rule 1: Filter Low-Quality Conversations

```sql
-- Keep high-quality conversations (rating >= 3.0)
SELECT *
FROM best_practices
WHERE reward >= 3.0
  AND created_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
  AND user_id NOT LIKE 'test_%';  -- Exclude test data
```

**Criteria**:
- ✅ User rating ≥ 3.0
- ✅ Conversations within last 7 days
- ✅ Non-test users

#### Rule 2: Deduplicate by Semantic Similarity

```python
def deduplicate_by_similarity(conversations, threshold=0.95):
    """
    Deduplicate based on semantic similarity
    
    Args:
        conversations: List of conversations
        threshold: Similarity threshold (>= considered duplicate)
    
    Returns:
        Deduplicated conversation list
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
    
    # Encode all queries
    queries = [conv['query'] for conv in conversations]
    embeddings = model.encode(queries, normalize_embeddings=True)
    
    # Calculate similarity matrix
    similarity_matrix = np.dot(embeddings, embeddings.T)
    
    # Deduplicate (keep highest rated)
    deduplicated = []
    used_indices = set()
    
    for i in range(len(conversations)):
        if i in used_indices:
            continue
        
        # Find all indices similar to current query
        similar_indices = np.where(similarity_matrix[i] >= threshold)[0]
        
        # Select the one with highest rating
        best_idx = max(similar_indices, key=lambda idx: conversations[idx]['reward'])
        deduplicated.append(conversations[best_idx])
        
        # Mark as used
        used_indices.update(similar_indices)
    
    return deduplicated
```

**Deduplication Strategy**:
- Similarity **≥ 0.95** → Considered duplicate
- Keep conversation with highest rating
- Record deduplication statistics

#### Rule 3: Label Tool Chains

```python
def extract_tool_chain(tools_used_str):
    """
    Extract tool chain from tools_used field
    
    Args:
        tools_used_str: "tool1,tool2,tool3" or JSON string
    
    Returns:
        List[str]: Tool chain list
    """
    import json
    
    try:
        # Try parsing JSON
        if tools_used_str.startswith('['):
            return json.loads(tools_used_str)
        else:
            # Comma separated
            return [tool.strip() for tool in tools_used_str.split(',')]
    except:
        return []

def classify_domain(query):
    """
    Classify query domain
    
    Args:
        query: User query
    
    Returns:
        str: Domain label
    """
    # Domain-specific keywords (customize for your application)
    keywords = {
        "domain_a": ["keyword1", "keyword2"],
        "domain_b": ["keyword3", "keyword4"],
        "domain_c": ["keyword5", "keyword6"],
        "general": []  # General/other
    }
    
    for domain, kws in keywords.items():
        for kw in kws:
            if kw in query.lower():
                return domain
    
    return "general"
```

#### Rule 4: Data Validation

```python
def validate_conversation(conv):
    """
    Validate conversation data integrity
    
    Returns:
        Tuple[bool, str]: (is_valid, error_reason)
    """
    # Required fields check
    required_fields = ['query', 'response', 'tools_used', 'reward']
    for field in required_fields:
        if not conv.get(field):
            return False, f"Missing required field: {field}"
    
    # Tool chain check
    if len(conv['tools_used']) == 0:
        return False, "Empty tool chain"
    
    # Response length check
    if len(conv['response']) < 50:
        return False, "Response too short"
    
    # Rating range check
    if not (1 <= conv['reward'] <= 5):
        return False, "Rating out of range [1, 5]"
    
    return True, ""
```

### 1.3 Cleaning Pipeline Flowchart

```
Raw Conversation Data (Database: best_practices)
    ↓
[Rule 1] Filter low-quality conversations (reward < 3.0)
    ↓
[Rule 2] Deduplicate queries (similarity >= 0.95)
    ↓
[Rule 3] Label tool chains and domains
    ↓
[Rule 4] Data validation (integrity check)
    ↓
Store in training database (training_dataset table)
    ↓
Generate cleaning report (logs + statistics)
```

### 1.4 Storage Structure

```sql
CREATE TABLE IF NOT EXISTS training_dataset (
    id INT PRIMARY KEY AUTO_INCREMENT,
    conversation_id VARCHAR(50) UNIQUE NOT NULL,  -- Original conversation ID
    user_id INT NOT NULL,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    tool_chain JSON NOT NULL,  -- ["tool1", "tool2", ...]
    domain VARCHAR(50) NOT NULL,  -- domain_a, domain_b, domain_c
    reward FLOAT NOT NULL,
    similarity_cluster INT DEFAULT NULL,  -- Similarity cluster ID
    is_validated BOOLEAN DEFAULT FALSE,  -- Manually validated
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_domain (domain),
    INDEX idx_reward (reward),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

---

## 2. Fine-tuning Trigger Conditions

### 2.1 Data Volume Conditions

```python
def check_finetuning_readiness():
    """
    Check fine-tuning readiness
    
    Returns:
        Dict: Check results for each metric
    """
    # 1. High-quality conversation count (rating >= 4.5)
    high_quality_count = db.query("""
        SELECT COUNT(*) as cnt
        FROM training_dataset
        WHERE reward >= 4.5
    """).fetch_one()['cnt']
    
    # 2. Scenario coverage
    coverage = db.query("""
        SELECT 
            domain,
            COUNT(*) as cnt,
            COUNT(*) * 100.0 / (SELECT COUNT(*) FROM training_dataset) as percentage
        FROM training_dataset
        GROUP BY domain
    """).fetch_all()
    
    coverage_dict = {row['domain']: row['percentage'] for row in coverage}
    
    # 3. Few-Shot retrieval hit rate
    few_shot_hit_rate = calculate_few_shot_hit_rate()  # To be implemented
    
    return {
        "high_quality_count": {
            "value": high_quality_count,
            "threshold": 5000,
            "ready": high_quality_count >= 5000
        },
        "coverage": {
            "value": coverage_dict,
            "threshold": 0.20,  # At least 20% for each scenario
            "ready": all(v >= 20 for v in coverage_dict.values())
        },
        "few_shot_hit_rate": {
            "value": few_shot_hit_rate,
            "threshold": 0.60,
            "ready": few_shot_hit_rate < 0.60  # Low hit rate indicates need for fine-tuning
        }
    }
```

### 2.2 Trigger Thresholds

| Metric | Threshold | Current Status (Example) | Ready |
|--------|-----------|-------------------------|-------|
| **High-quality conversations** | ≥ 5000 | 1200 | ❌ |
| **Scenario coverage** | Each scenario ≥ 20% | DomainA 35% / DomainB 30% / DomainC 15% / DomainD 20% | ❌ (DomainC insufficient) |
| **Few-Shot hit rate** | < 60% | 55% | ✅ |

**Comprehensive Judgment**: Only trigger fine-tuning when **all metrics** are met.

---

## 3. Fine-tuning Architecture Design

### 3.1 Base Model

```yaml
model:
  name: [Your chosen base model]
  parameters: [Model size]
  context_length: [Context window size]
  license: [License type]
  download: [Model source]
```

### 3.2 Fine-tuning Method: LoRA

**LoRA (Low-Rank Adaptation)** Parameters:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                # Rank: controls adapter parameter count
    lora_alpha=32,       # Alpha: scaling factor (usually 2x rank)
    lora_dropout=0.05,   # Dropout: prevent overfitting
    target_modules=[     # Target modules: which layers to apply LoRA
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    bias="none",         # Don't train bias
    task_type="CAUSAL_LM"  # Task type: Causal Language Model
)
```

**Why LoRA?**
- ✅ Small parameter count (<1% of original model)
- ✅ Fast training (80%+ GPU time reduction)
- ✅ Flexible deployment (switchable adapters)
- ✅ Avoid catastrophic forgetting

### 3.3 Training Data Format

```json
{
    "instruction": "You are a professional AI assistant. Based on user queries and tool call results, generate professional responses.",
    "input": "User query: [query]\n\nTool call results:\n- tool1: {...}\n- tool2: {...}",
    "output": "Based on the information provided...",
    "metadata": {
        "domain": "domain_a",
        "tool_chain": ["tool1", "tool2", "tool3"],
        "rating": 4.8
    }
}
```

### 3.4 Training Configuration

```python
training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=3,          # Number of training epochs
    per_device_train_batch_size=4,  # Batch size (adjust based on GPU memory)
    gradient_accumulation_steps=8,  # Gradient accumulation (simulate larger batch)
    learning_rate=3e-4,          # Learning rate
    lr_scheduler_type="cosine",  # Learning rate scheduler
    warmup_ratio=0.1,            # Warmup ratio
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    fp16=True,                   # Mixed precision training (save memory)
    gradient_checkpointing=True, # Gradient checkpointing (further save memory)
    max_grad_norm=1.0,          # Gradient clipping
    save_total_limit=3,         # Only keep last 3 checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)
```

### 3.5 GPU Resource Requirements

| Configuration | Memory Required | Training Time | Cost (AWS/Azure) |
|--------------|----------------|---------------|------------------|
| **Minimum** | 24GB (RTX 4090) | 12-18 hours | $50-80 |
| **Recommended** | 40GB (A100) | 6-8 hours | $100-150 |
| **High Performance** | 80GB (A100 80GB) | 3-4 hours | $200-300 |

---

## 4. Deployment Strategy

### 4.1 Complete Deployment Pipeline

```
Data Accumulation Meets Criteria
    ↓
[Stage 1] Train LoRA Adapter (GPU Server)
    ├─ Data preparation (train/val/test = 8:1:1)
    ├─ Training (3-5 epochs)
    ├─ Model merging (BaseModel + LoRA Adapter)
    └─ Offline evaluation (test set)
    ↓
[Stage 2] Offline Validation
    ├─ Manual evaluation (sample 100 items)
    ├─ Automated evaluation (tool chain accuracy, response quality)
    └─ Pass criteria: score >= baseline model
    ↓
[Stage 3] A/B Testing (7 days)
    ├─ 50% traffic → fine-tuned model
    ├─ 50% traffic → baseline model
    ├─ Monitor metrics (user satisfaction, response time, tool chain accuracy)
    └─ Decision: Continue or rollback
    ↓
[Stage 4] Gradual Rollout
    ├─ 70% traffic (3 days)
    ├─ 90% traffic (3 days)
    └─ 100% traffic (full deployment)
    ↓
[Stage 5] Full Replacement
    ├─ Update deployed model
    ├─ Keep original model as fallback
    └─ Continue monitoring for 30 days
```

### 4.2 Degradation Strategy

```python
class ModelManager:
    def __init__(self):
        self.current_model = "finetuned-model"
        self.baseline_model = "base-model"
        self.degraded = False
    
    def check_health(self):
        """Check model health"""
        metrics = get_recent_metrics(hours=1)
        
        # Degradation conditions
        if (metrics['user_satisfaction'] < 3.8 or
            metrics['error_rate'] > 0.05 or
            metrics['p95_latency'] > 5.0):
            self.degrade()
    
    def degrade(self):
        """Degrade to baseline model"""
        logger.warning("🚨 Anomaly detected, degrading to baseline model")
        self.current_model = self.baseline_model
        self.degraded = True
        send_alert("Model degradation notification")
```

---

## 5. Evaluation Metrics

### 5.1 Core Metrics

| Category | Metric Name | Baseline | Target | Weight | Calculation |
|----------|------------|----------|--------|--------|-------------|
| **User Experience** | User satisfaction (avg rating) | 4.2 | 4.5+ | ⭐⭐⭐⭐⭐ | AVG(user_rating) |
| **Accuracy** | Tool chain accuracy | 75% | 85%+ | ⭐⭐⭐⭐⭐ | Correct / Total |
| **Efficiency** | Response time (P95) | 3.5s | < 4.0s | ⭐⭐⭐⭐ | Percentile(95) |
| **Cost** | API cost reduction | Baseline | 30%+ | ⭐⭐⭐⭐ | (Baseline - Current) / Baseline |
| **Intelligence** | Few-Shot hit rate | 55% | 70%+ | ⭐⭐⭐ | Hits / Total |

---

## 6. Risk Assessment and Mitigation

### 6.1 Risk List

| Risk Type | Description | Probability | Impact | Mitigation |
|-----------|-------------|-------------|--------|------------|
| **Data Quality** | Noise in cleaned data | Medium | High | Manual review of 100 samples + strict validation rules |
| **Overfitting** | Good on training set, poor on test set | Medium | Medium | Early stopping + validation monitoring + Dropout |
| **Performance Degradation** | Increased response time after fine-tuning | Low | High | Model quantization + inference optimization + A/B testing |
| **Scenario Coverage Insufficient** | Too little data for some scenarios | Medium | Medium | Actively generate supplementary data + transfer learning |
| **Catastrophic Forgetting** | General capability decline after fine-tuning | Low | Medium | LoRA method + retain base task data |
| **GPU Resource Shortage** | GPU unavailable/cost overrun during training | Low | Medium | Rent cloud GPU (AWS/Azure) + training queue |
| **Model Leakage** | Fine-tuned model stolen | Low | High | Access control + model encryption + audit logs |

---

## 7. Implementation Timeline

### 7.1 Milestones

| Stage | Time Point | Key Tasks | Deliverables |
|-------|-----------|-----------|--------------|
| **P0** | Framework stable +2 weeks | Implement data cleaning pipeline | Cleaning script + first cleaning report |
| **P1** | +1 month | Continuous data accumulation | Weekly cleaning reports |
| **P2** | +3 months | Check fine-tuning readiness | Readiness assessment report |
| **P3** | +3.5 months | Train LoRA adapter | Fine-tuned model + evaluation report |
| **P4** | +4 months | A/B testing | A/B test report + decision |
| **P5** | +4.5 months | Gradual rollout | Rollout report |
| **P6** | +5 months | Full replacement | Launch announcement + monitoring dashboard |

---

## 8. References

### 8.1 Papers

1. **LoRA**: ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
2. **PEFT**: ["Parameter-Efficient Fine-Tuning of Large-Scale Pre-Trained Language Models"](https://arxiv.org/abs/2110.04366) (Liu et al., 2021)
3. **Instruction Tuning**: ["Finetuned Language Models Are Zero-Shot Learners"](https://arxiv.org/abs/2109.01652) (Wei et al., 2021)

### 8.2 Tools

- **Hugging Face Transformers**: https://huggingface.co/docs/transformers
- **PEFT Library**: https://github.com/huggingface/peft
- **Sentence Transformers**: https://www.sbert.net/
- **Weights & Biases (Training Monitoring)**: https://wandb.ai/

---

**Maintainer**: DAML-RAG Framework Team  
**Last Review**: 2025-11-07

<div align="center">
<strong>📊 Data-Driven · 🎯 Precise Optimization · 🚀 Continuous Evolution</strong>
</div>

---

## 📖 Application-Specific Implementation

This generic architecture has been implemented in:

- **BUILD_BODY Fitness Application**: See [数据清洗与微调架构规划.md](../../build_body/mcp-servers/meta-learning-mcp/docs/02-核心架构/数据清洗与微调架构规划.md) for fitness-specific implementation details.

To adapt this architecture to your domain:
1. Customize domain-specific keywords in `classify_domain()`
2. Define your domain-specific evaluation metrics
3. Adjust trigger thresholds based on your data scale
4. Select appropriate base model for your use case

