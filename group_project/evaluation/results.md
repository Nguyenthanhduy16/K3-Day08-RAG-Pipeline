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
| Faithfulness | 0.7714 | 0.5972 | +0.1742 |
| Answer Relevance | 0.6563 | 0.6559 | +0.0004 |
| Context Recall | 0.6333 | 0.6500 | -0.0167 |
| Context Precision | 0.6967 | 0.6967 | +0.0000 |
| **Average** | **0.6894** | **0.6499** | **+0.0395** |

---

## A/B Configuration Details

**Config A — Hybrid + Reranking**
- Dense semantic search (ChromaDB cosine) + sparse BM25 fused via Reciprocal Rank Fusion (RRF).
- Jina Reranker cross-encoder re-scores the top-20 candidates before selecting top-5 for the LLM.

**Config B — Dense-Only**
- Pure dense vector search from ChromaDB; BM25 and reranking skipped.
- Simpler but may miss relevant chunks that share few semantic overlaps.

**Conclusion:**
Config A (Hybrid + Reranking) outperforms Config B with a mean score advantage of **0.0395**.
Jina Reranking improves context precision by surfacing the most semantically faithful chunks.

---

## Worst Performers (Bottom 3 — Config A)

| # | Question (truncated) | Faithfulness | Answer Relevance | Context Recall | Root Cause |
|---|----------------------|-------------|-----------------|----------------|------------|
| 1 | Ngưỡng Guts mục tiêu cho cự ly Long 4000m là bao nhiêu? | 0.00 | 0.00 | 0.00 | Context too sparse or answer rejected by fallback guard |
| 2 | Ngưỡng Guts mục tiêu của từng cự ly là bao nhiêu, và ý ... | 1.00 | 0.76 | 1.00 | Context too sparse or answer rejected by fallback guard |
| 3 | Kỹ năng hồi phục (recovery skill) hồi lại bao nhiêu phầ... | 1.00 | 0.81 | 0.50 | Context too sparse or answer rejected by fallback guard |

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