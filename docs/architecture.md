# MemorySpine Architecture

## Overview

MemorySpine is a constant-memory context extension system designed to augment any LLM with long-context retrieval capabilities. Unlike KV-cache approaches that scale linearly with context length, MemorySpine uses a fixed-size embedding store with O(1) memory complexity.

## Core Components

### 1. Orthogonal Rotation Matrix (Ω)

**Purpose**: Transform raw embeddings into a basis where dimensions are statistically independent, enabling uniform hash distribution.

**Construction**: Modified Gram-Schmidt orthogonalization on a Gaussian random matrix with a deterministic seed.

```
Initialization:
  1. Generate D×D matrix G with entries ~ N(0,1) using seed=42
  2. For each row i = 0 to D-1:
     a. Subtract projection onto all previous rows:
        Ω[i] = G[i] - Σ_{k<i} <G[i], Ω[k]> · Ω[k]
     b. Normalize: Ω[i] = Ω[i] / ||Ω[i]||
  3. Result: Ω^T · Ω = I (orthogonal)
```

**Properties**:
- **Norm-preserving**: ||Ωx|| = ||x|| for all x
- **Cosine-preserving**: cos(Ωx, Ωy) = cos(x, y)
- **Decorrelating**: Breaks structured correlations in neural network embeddings
- **Deterministic**: Same seed → same Ω → reproducible hashing

**Inspiration**: TurboQuant (Ashkboos et al., 2025) Lemma 1 proves that after rotation by a Haar-random orthogonal matrix, each coordinate follows a Beta distribution independent of input structure.

### 2. Content-Addressable Hash

**Purpose**: Map each embedding to one of S slots deterministically.

```
hash(x):
  1. x_rot = Ω · x                              // rotate
  2. h = 0
  3. for i = 0 to 15:                            // use first 16 dims
       h ^= bits(x_rot[i]) + φ + (h << 6) + (h >> 2)
  4. return h mod S
```

Where `φ = 0x9E3779B9` (golden ratio constant, optimal for bit mixing).

**Why only 16 dimensions?** The orthogonal rotation distributes information uniformly — after rotation, any 16 dimensions capture a representative projection of the full D-dimensional space. Using more dimensions increases hash computation cost without improving distribution quality.

### 3. 2-Bit Lloyd-Max Quantization

**Purpose**: Compress each D-dimensional embedding from D×4 bytes to D/4 bytes (15.6× compression).

```
Quantization of v ∈ ℝ^D:
  1. Compute norm: σ = ||v||₂
  2. Normalize: v̂ = v / σ
  3. For each dimension i:
       if v̂[i] < -0.98/√D  → code 0 (centroid: -1.51/√D)
       if v̂[i] < 0          → code 1 (centroid: -0.45/√D)
       if v̂[i] < +0.98/√D  → code 2 (centroid: +0.45/√D)
       else                  → code 3 (centroid: +1.51/√D)
  4. Pack 4 codes per byte
  5. Store σ as float32 scale factor

Dequantization:
  out[i] = centroid[code[i]] × σ
```

**Reconstruction quality**: Expected cosine similarity between original and dequantized vectors: ~0.94.

### 4. Slot Storage Layout

Each slot uses exactly `⌈D/4⌉ + 5` bytes:

| Field | Size | Description |
|:---|:---:|:---|
| Quantized vector | D/4 bytes | 2-bit packed codes |
| Scale factor (σ) | 4 bytes | float32 L2 norm |
| Occupied flag | 1 byte | uint8 (0 or 1) |
| **Total** | **D/4 + 5** | **197 bytes for D=768** |

### 5. Top-K Retrieval

Linear scan over all occupied slots:

```
retrieve_topk(query, K):
  1. For each occupied slot s:
       a. Dequantize: v_s = dequant(slot_s)
       b. Compute: sim(s) = cos(v_s, query)
  2. Return K slots with highest similarity
```

**Complexity**: O(|occupied| × D). For 191 chunks: ~1.5 ms on modern CPUs.

### 6. Persistence (Sparse Binary Format)

```
File layout:
  [4 bytes]  "MSPN" magic
  [4 bytes]  D (dimension)
  [4 bytes]  S (slot count)
  [4 bytes]  N_occ (number of occupied slots)
  [D²×4 bytes]  Ω rotation matrix
  For each occupied slot:
    [4 bytes]  slot index
    [D/4 bytes]  quantized embedding
    [4 bytes]  scale factor
```

Only occupied slots are serialized, making the file compact: 2.3 MB for 191 stored chunks.

## Dual-Server Architecture

MemorySpine uses two separate llama.cpp servers:

```
┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│  Embedding    │     │   MemorySpine  │     │  Generation   │
│  Server       │────→│   Store        │────→│  Server       │
│  port 8091    │     │   10M × 197B   │     │  port 8090    │
│  768-dim      │     │   1.84 GB      │     │  Any LLM      │
└──────────────┘     └────────────────┘     └──────────────┘
```

**Why two servers?**
- Embedding models (e.g., nomic-embed-text) are trained specifically for semantic similarity
- Generation models (e.g., Qwen, Llama) are trained for text generation
- Using a dedicated embedding model dramatically improves retrieval quality (100% vs 50% accuracy)

## Data Flow

### Ingestion
```
Document
  → Split into 1500-char chunks (200-char overlap)
  → For each chunk:
      → Add "search_document: " prefix
      → POST to embedding server → 768-dim vector
      → hash(vector) → slot index
      → quantize(vector) → 197 bytes
      → store in slot + save text in ChunkStore
```

### Query
```
Question
  → Add "search_query: " prefix
  → POST to embedding server → 768-dim vector
  → Scan all occupied slots, compute cosine similarity
  → Return top-5 most similar chunks
  → Inject into system prompt:
      "<|im_start|>system
       [retrieved context]
       <|im_end|>"
  → POST to generation server → answer
```

## Design Decisions

### Why Embedding-Level Rather Than Attention-Level?

KV-cache stores per-layer, per-head attention states — enabling fine-grained token-level attention. MemorySpine stores embedding-level semantic fingerprints — enabling chunk-level semantic retrieval. This trade-off means:

- ✅ **Model-agnostic**: Works with any LLM without modification
- ✅ **Extreme compression**: 15.6× over float32
- ✅ **Zero architectural coupling**: No fine-tuning or retraining needed
- ❌ **Chunk granularity**: Cannot retrieve individual tokens
- ❌ **No attention-head specificity**: Loses per-layer information

For factual recall tasks, this trade-off is highly favorable.

### Why Write-Once Instead of EMA?

Earlier versions used Exponential Moving Average (EMA) blending on collisions. At scale (>10K patterns), EMA causes exponential decay of stored information: after N collisions, the original pattern's energy is 0.9^N → near zero. Write-once preserves each stored pattern cleanly (one quantization pass, no compounding error).

### Why nomic-embed-text?

Ablation experiments show that using the LLM's own embeddings causes **80% hash collision rate** due to high inter-embedding correlation. Dedicated embedding models trained with contrastive learning produce more uniformly distributed vectors, achieving **0% collision rate** on the same data.
