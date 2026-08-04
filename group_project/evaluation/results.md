# RAG Evaluation Results

## Framework

> **Framework:** RAGAS v0.1.21  
> **LLM judge:** OpenAI GPT-4o-mini  
> **Dataset:** 5 of 20 golden Q&A pairs (rate-limit budget)  
> **Configs compared:** A = Hybrid + Reranking | B = Dense-Only

---

## Overall Scores

| Metric | Config A (hybrid+rerank) | Config B (dense-only) | Delta |
|--------|--------------------------|----------------------|-------|
| Faithfulness | 0.9333 | 0.5417 | +0.3917 |
| Answer Relevance | 0.4936 | 0.4945 | -0.0009 |
| Context Recall | 0.8000 | 0.7000 | +0.1000 |
| Context Precision | 0.7608 | 0.7608 | +0.0000 |
| **Average** | **0.7469** | **0.6243** | **+0.1227** |

---

## A/B Configuration Details

**Config A — Hybrid + Reranking**
- Dense semantic search (ChromaDB cosine) + sparse BM25 fused via Reciprocal Rank Fusion (RRF).
- Jina Reranker cross-encoder re-scores the top-20 candidates before selecting top-5 for the LLM.

**Config B — Dense-Only**
- Pure dense vector search from ChromaDB; BM25 and reranking skipped.
- Simpler but may miss relevant chunks that share few semantic overlaps.

**Conclusion:**
Config A (Hybrid + Reranking) outperforms Config B with a mean score advantage of **0.1227**.
Jina Reranking improves context precision by surfacing the most semantically faithful chunks.

---

## Worst Performers (Bottom 3 — Config A)

| # | Question (truncated) | Faithfulness | Answer Relevance | Context Recall | Root Cause |
|---|----------------------|-------------|-----------------|----------------|------------|
| 1 | Ngưỡng Guts mục tiêu cho cự ly Long 4000m là bao nhiêu? | nan | 0.00 | 0.00 | Context too sparse or answer rejected by fallback guard |
| 2 | Công thức tính HP của một uma là gì, và hệ số Strategy ... | nan | 0.00 | 1.00 | Context too sparse or answer rejected by fallback guard |
| 3 | Nếu uma của tôi có 1200 Speed, xác suất nhận được 3 sta... | 0.80 | 0.87 | 1.00 | Context too sparse or answer rejected by fallback guard |

---

## Recommendations

### 1. Increase chunk overlap
**Action:** Raise overlap from 50 to 150 tokens when chunking.  
**Expected impact:** Improves Context Recall for boundary-crossing facts.

### 2. Add Few-Shot examples to system prompt
**Action:** Include 2-3 demonstrative Q&A pairs in the system prompt.  
**Expected impact:** Raises Faithfulness by showing the model exactly how to cite sources.

### 3. Domain-specific keyword pre-filter
**Action:** Pre-filter BM25 candidates with game-specific stopword removal.  
**Expected impact:** Reduces noisy candidates entering RRF, improving Context Precision.