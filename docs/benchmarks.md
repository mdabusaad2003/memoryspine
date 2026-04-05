# MemorySpine Benchmarks

## Test Environment

| Property | Value |
|:---|:---|
| CPU | Consumer-grade (CPU-only inference) |
| OS | Windows 11 |
| Embedding Model | nomic-embed-text-v1.5 (768-dim, f16 GGUF, 274 MB) |
| Generation Model | LLaMA 3 8B Instruct (Q4_K_M GGUF, ~4.9 GB) |
| MemorySpine | 27M slots, 768-dim, 2-bit quantized |
| Memory Footprint | 4.95 GB (constant) |
| Inference | llama.cpp v0.0.x, CPU |

---

## Phase 29: Standalone Recall Test

Tests MemorySpine's standalone retrieval fidelity using synthetic random embeddings. Each embedding is stored, then queried to verify retrieval quality.

| Context Depth | Slots Used | Avg Cosine Sim | Recall@0.9 |
|:---:|:---:|:---:|:---:|
| 50 | 50 / 10M | 0.941 | 50/50 (100%) |
| 128 | 128 / 10M | 0.940 | 100/100 (100%) |
| 500 | 500 / 10M | 0.940 | 100/100 (100%) |
| 1,000 | 1K / 10M | 0.939 | 100/100 (100%) |
| 5,000 | 5K / 10M | 0.940 | 100/100 (100%) |
| 10,000 | 10K / 10M | 0.940 | 100/100 (100%) |
| 100,000 | 100K / 10M | 0.931 | 99/100 (99%) |
| **1,000,000** | **1M / 10M** | **0.889** | **95/100 (95%)** |

**Key findings**:
- Cosine similarity remains above 0.88 even at 1M stored patterns
- 100% recall up to 10K patterns
- Slight degradation at high occupancy due to hash collisions

---

## Phase 30: LLM Integration Test

A synthetic document of ~240,000 characters (~60K tokens) spanning 10 topical domains, with 200 chapters (20 repetitions × 10 topics).

### Ingestion Statistics

| Metric | Value |
|:---|:---|
| Document size | ~240,000 characters |
| Chunks generated | 191 |
| Chunk size | 1,500 chars with 200 overlap |
| Unique slots used | 191 / 10M (0% collision) |
| Embedding throughput | 3.3 chunks/second (CPU) |
| Total ingestion time | ~58 seconds |

### Factual Recall (10 Questions)

| # | Question | Target Domain | Retrieved? | Correct? | Top Cosine Sim |
|:---:|:---|:---:|:---:|:---:|:---:|
| 1 | How tall is Olympus Mons? | Space | ✅ | ✅ 72,000 ft | 0.608 |
| 2 | When was the Mona Lisa painted? | Art | ✅ | ✅ 1503-1519 | 0.733 |
| 3 | Base pairs in human genome? | Genome | ✅ | ✅ 3 billion | 0.771 |
| 4 | What is the Maillard reaction? | Gastronomy | ✅ | ✅ Amino acids + sugars >140°C | 0.680 |
| 5 | Amazon River discharge? | Rainforest | ✅ | ✅ More than next 7 rivers | 0.656 |
| 6 | Maximum Bitcoin supply? | Crypto | ✅ | ✅ 21 million | 0.701 |
| 7 | Colosseum spectators? | Rome | ✅ | ✅ 50K-80K | 0.697 |
| 8 | Rumelhart et al. 1986? | ML | ✅ | ✅ Backpropagation | 0.560 |
| 9 | Octopus hearts? | Marine | ✅ | ✅ Three hearts | 0.674 |
| 10 | Heisenberg uncertainty? | QM | ✅ | ✅ Position & momentum | 0.727 |

**Result: 10/10 correct (100% accuracy), zero hallucination**

### Cosine Similarity Distribution

| Range | Count | Percentage |
|:---:|:---:|:---:|
| 0.50 – 0.60 | 1 | 10% |
| 0.60 – 0.70 | 4 | 40% |
| 0.70 – 0.80 | 4 | 40% |
| > 0.80 | 1 | 10% |

Average top-1 cosine similarity: **0.681**

---

## Ablation: LLM Embeddings vs Dedicated Embedding Model

| Metric | LLaMA 3 8B Embeddings | nomic-embed-text |
|:---|:---:|:---:|
| Correct answers | 5/10 (50%) | **10/10 (100%)** |
| Unique slots | 39 / 191 (80% collisions) | **191 / 191 (0% collisions)** |
| Retrieval diversity | Same 5 slots all queries | Different slots per query |
| Hallucination | Severe | **None** |
| Embedding speed | 0.4 chunks/sec | **3.3 chunks/sec** |

**Conclusion**: A dedicated embedding model is essential. LLM hidden states have high inter-embedding correlation, causing hash clustering and collisions.

---

## Memory Scaling

| System | 4K context | 32K | 128K | 1M | 10M |
|:---|:---:|:---:|:---:|:---:|:---:|
| LLaMA 3 8B KV-cache | 128 MB | 1.0 GB | 4.0 GB | 32 GB | 320 GB |
| Llama 3 70B KV-cache | 320 MB | 2.5 GB | 10 GB | 80 GB | 800 GB |
| FAISS (float32) | 3 KB | 24 KB | 98 KB | 3 GB | 30.7 GB |
| **MemorySpine** | **4.95 GB** | **4.95 GB** | **4.95 GB** | **4.95 GB** | **4.95 GB** |

---

## Retrieval Latency

| Stored Patterns | Top-5 Retrieval | Notes |
|:---:|:---:|:---|
| 191 | ~1.5 ms | Our test case |
| 1,000 | ~8 ms | Interactive |
| 10,000 | ~80 ms | Interactive |
| 100,000 | ~800 ms | Near real-time |
| 1,000,000 | ~8 sec | Needs ANN indexing |

---

## Compression

| Metric | Float32 | MemorySpine (2-bit) | Ratio |
|:---|:---:|:---:|:---:|
| Per-vector storage (768-dim) | 3,072 bytes | 197 bytes | **15.6×** |
| 10M vectors | 30.7 GB | 1.84 GB | **16.7×** |
| Per-vector (384-dim) | 1,536 bytes | 101 bytes | **15.2×** |
| Per-vector (1024-dim) | 4,096 bytes | 261 bytes | **15.7×** |

---

*Benchmarks performed as part of the MemorySpine Phase 29/30 validation.*
*Author: Abu Saad (mdabusaad2003@gmail.com)*
