// ============================================================================
// MEMSPINE.H — MemorySpine: O(1) Memory Context Extension for LLMs
//
// A fixed-size, 2-bit quantized embedding store for constant-memory
// semantic retrieval. Works with any embedding model and any LLM.
//
// Configuration (compile-time):
//   SPINE_DIM   — Embedding dimension (default: 768 for nomic-embed-text)
//   SPINE_SLOTS — Number of storage slots (default: 10,000,000)
//
// Memory footprint: SLOTS × (DIM/4 + 5) + DIM² × 4 bytes
//   Default: 10M × 197 + 2.36M ≈ 1.84 GB
//
// Author: Abu Saad (mdabusaad2003@gmail.com)
// Part of the KHND (Knowledge-Hamiltonian Neural Dynamics) project.
// ============================================================================
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================================
// Configuration — override at compile time with -DSPINE_DIM=384, etc.
// ============================================================================
#ifndef SPINE_DIM
#define SPINE_DIM 768 // Must match your embedding model output dimension
#endif

#ifndef SPINE_SLOTS
#define SPINE_SLOTS 27000000 // 27M slots
#endif

static constexpr int MEMSPINE_DIM = SPINE_DIM;
static constexpr int MEMSPINE_SLOTS = SPINE_SLOTS;
static constexpr int MEMSPINE_QUANT_BYTES = MEMSPINE_DIM / 4; // 2 bits per dim

// ============================================================================
// Deterministic RNG — reproducible Omega initialization across platforms
// ============================================================================
struct SpineRNG {
    uint32_t state;

    explicit SpineRNG(uint32_t seed = 42) : state(seed) {}

    /// Generate uniform random float in [0, 1)
    float uniform() {
        state = state * 1103515245u + 12345u;
        return static_cast<float>((state >> 16) & 0x7FFF) / 32767.0f;
    }

    /// Generate standard normal variate via Box-Muller transform
    float normal() {
        float u1 = uniform();
        if (u1 < 1e-7f) u1 = 1e-7f;
        float u2 = uniform();
        return std::sqrt(-2.0f * std::log(u1)) *
               std::cos(2.0f * 3.14159265358979f * u2);
    }
};

// ============================================================================
// ChunkStore — maps slot IDs to original text chunks for retrieval
// ============================================================================
struct ChunkStore {
    std::unordered_map<int, std::string> chunks;

    void add(int slot_id, const std::string &text) {
        chunks[slot_id] = text;
    }

    std::string get(int slot_id) const {
        auto it = chunks.find(slot_id);
        return (it != chunks.end()) ? it->second : "";
    }

    int size() const { return static_cast<int>(chunks.size()); }

    /// Save chunk store to binary file
    bool save(const char *path) const {
        FILE *f = fopen(path, "wb");
        if (!f) return false;

        int n = static_cast<int>(chunks.size());
        if (fwrite(&n, sizeof(int), 1, f) != 1) { fclose(f); return false; }

        for (const auto &kv : chunks) {
            if (fwrite(&kv.first, sizeof(int), 1, f) != 1) { fclose(f); return false; }
            int len = static_cast<int>(kv.second.size());
            if (fwrite(&len, sizeof(int), 1, f) != 1) { fclose(f); return false; }
            if (fwrite(kv.second.data(), 1, len, f) != static_cast<size_t>(len)) {
                fclose(f); return false;
            }
        }
        fclose(f);
        return true;
    }

    /// Load chunk store from binary file
    bool load(const char *path) {
        FILE *f = fopen(path, "rb");
        if (!f) return false;

        chunks.clear();
        int n;
        if (fread(&n, sizeof(int), 1, f) != 1) { fclose(f); return false; }

        for (int i = 0; i < n; i++) {
            int slot, len;
            if (fread(&slot, sizeof(int), 1, f) != 1) { fclose(f); return false; }
            if (fread(&len, sizeof(int), 1, f) != 1) { fclose(f); return false; }
            std::string text(len, '\0');
            if (fread(&text[0], 1, len, f) != static_cast<size_t>(len)) {
                fclose(f); return false;
            }
            chunks[slot] = std::move(text);
        }
        fclose(f);
        return true;
    }
};

// ============================================================================
// MemSpine — Core data structure
//   S slots × D dimensions, 2-bit quantized, write-once
// ============================================================================
struct MemSpine {
    uint8_t *M_quant;                      // [S × QUANT_BYTES] quantized storage
    float   *M_scale;                      // [S] per-slot scale factors
    uint8_t *M_occupied;                   // [S] occupancy bitmap
    float  (*Omega)[MEMSPINE_DIM];         // [D × D] orthogonal rotation matrix
    std::vector<int> occupied_slots;       // indices of occupied slots
    bool initialized;

    MemSpine()
        : M_quant(nullptr), M_scale(nullptr), M_occupied(nullptr),
          Omega(nullptr), initialized(false) {}

    ~MemSpine() { cleanup(); }

    // ========================================================================
    // Initialization — allocates memory and computes Omega
    // ========================================================================
    bool init(uint32_t seed = 42) {
        size_t total_bytes = static_cast<size_t>(MEMSPINE_SLOTS) *
                             (MEMSPINE_QUANT_BYTES + sizeof(float) + 1);

        printf("  [MemSpine] Allocating %dM slots x %d bytes = %.2f GB...\n",
               MEMSPINE_SLOTS / 2700000, MEMSPINE_QUANT_BYTES + 5,
               static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0));
        fflush(stdout);

        M_quant   = new (std::nothrow) uint8_t[static_cast<size_t>(MEMSPINE_SLOTS) * MEMSPINE_QUANT_BYTES]();
        M_scale   = new (std::nothrow) float[MEMSPINE_SLOTS]();
        M_occupied = new (std::nothrow) uint8_t[MEMSPINE_SLOTS]();
        Omega     = new (std::nothrow) float[MEMSPINE_DIM][MEMSPINE_DIM];

        if (!M_quant || !M_scale || !M_occupied || !Omega) {
            printf("  [MemSpine] ERROR: Failed to allocate memory!\n");
            cleanup();
            return false;
        }

        printf("  [MemSpine] Initializing Omega (%dx%d, Modified Gram-Schmidt)...\n",
               MEMSPINE_DIM, MEMSPINE_DIM);
        fflush(stdout);
        SpineRNG rng(seed);
        init_omega(rng);

        initialized = true;
        printf("  [MemSpine] Ready. %dM slots, %d-dim, 2-bit quantized, %.2f GB\n",
               MEMSPINE_SLOTS / 2700000, MEMSPINE_DIM,
               static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0));
        return true;
    }

    // ========================================================================
    // Cleanup — free all allocated memory
    // ========================================================================
    void cleanup() {
        delete[] M_quant;   M_quant   = nullptr;
        delete[] M_scale;   M_scale   = nullptr;
        delete[] M_occupied; M_occupied = nullptr;
        delete[] Omega;     Omega     = nullptr;
        occupied_slots.clear();
        initialized = false;
    }

    // ========================================================================
    // Modified Gram-Schmidt orthogonalization
    //
    // Constructs a D×D orthogonal matrix Omega from a Gaussian random matrix.
    // Properties:
    //   - Omega^T * Omega = I (orthogonal)
    //   - Preserves norms and cosine similarities
    //   - Decorrelates input dimensions for uniform hash distribution
    //   - Deterministic given RNG seed
    // ========================================================================
    void init_omega(SpineRNG &rng) {
        // Fill with Gaussian random values
        for (int i = 0; i < MEMSPINE_DIM; i++)
            for (int j = 0; j < MEMSPINE_DIM; j++)
                Omega[i][j] = rng.normal();

        // Modified Gram-Schmidt: orthogonalize rows
        for (int i = 0; i < MEMSPINE_DIM; i++) {
            // Subtract projections onto all previous rows
            for (int k = 0; k < i; k++) {
                float dot = 0;
                for (int j = 0; j < MEMSPINE_DIM; j++)
                    dot += Omega[i][j] * Omega[k][j];
                for (int j = 0; j < MEMSPINE_DIM; j++)
                    Omega[i][j] -= dot * Omega[k][j];
            }
            // Normalize
            float norm = 0;
            for (int j = 0; j < MEMSPINE_DIM; j++)
                norm += Omega[i][j] * Omega[i][j];
            norm = std::sqrt(norm) + 1e-12f;
            for (int j = 0; j < MEMSPINE_DIM; j++)
                Omega[i][j] /= norm;
        }
    }

    // ========================================================================
    // Content-addressable hash
    //
    // 1. Rotate input by Omega (decorrelates dimensions)
    // 2. Apply golden-ratio FNV-style hash on first 16 rotated dimensions
    // 3. Map to slot index via modulo
    //
    // The golden ratio constant 0x9E3779B9 = floor(2^32 / phi) provides
    // optimal bit mixing (Knuth, 1997).
    // ========================================================================
    int hash(const float *x) const {
        float x_rot[MEMSPINE_DIM];
        for (int i = 0; i < MEMSPINE_DIM; i++) {
            float sum = 0;
            for (int j = 0; j < MEMSPINE_DIM; j++)
                sum += Omega[i][j] * x[j];
            x_rot[i] = sum;
        }

        uint32_t h = 0;
        for (int i = 0; i < 16; i++) {
            uint32_t bits;
            std::memcpy(&bits, &x_rot[i], sizeof(bits));
            h ^= bits + 0x9E3779B9u + (h << 6) + (h >> 2);
        }
        return static_cast<int>(h % static_cast<uint32_t>(MEMSPINE_SLOTS));
    }

    // ========================================================================
    // 2-bit Lloyd-Max quantize and store (write-once)
    //
    // Centroids for unit-normal distribution:
    //   c0 = -1.51/sqrt(D), c1 = -0.45/sqrt(D)
    //   c2 = +0.45/sqrt(D), c3 = +1.51/sqrt(D)
    //
    // Decision boundaries:
    //   b1 = -0.98/sqrt(D), b2 = 0, b3 = +0.98/sqrt(D)
    //
    // Each dimension is encoded as 2 bits; 4 dims packed per byte.
    // Scale factor (L2 norm) stored separately for dequantization.
    // ========================================================================
    int store(const float *vec) {
        int slot = hash(vec);

        // Compute L2 norm for scale factor
        float norm = 0;
        for (int i = 0; i < MEMSPINE_DIM; i++)
            norm += vec[i] * vec[i];
        norm = std::sqrt(norm) + 1e-8f;

        float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(MEMSPINE_DIM));
        float b01 = -0.98f * inv_sqrt_d;
        float b23 =  0.98f * inv_sqrt_d;

        uint8_t *q = &M_quant[static_cast<size_t>(slot) * MEMSPINE_QUANT_BYTES];
        std::memset(q, 0, MEMSPINE_QUANT_BYTES);

        for (int i = 0; i < MEMSPINE_DIM; i++) {
            float v = vec[i] / norm;
            int code;
            if (v < b01)       code = 0;
            else if (v < 0.0f) code = 1;
            else if (v < b23)  code = 2;
            else               code = 3;
            q[i / 4] |= static_cast<uint8_t>(code << ((i % 4) * 2));
        }

        if (!M_occupied[slot]) {
            M_occupied[slot] = 1;
            occupied_slots.push_back(slot);
        }
        M_scale[slot] = norm;
        return slot;
    }

    // ========================================================================
    // Dequantize a stored slot back to float vector
    // ========================================================================
    void dequantize(int slot, float *out) const {
        const uint8_t *q = &M_quant[static_cast<size_t>(slot) * MEMSPINE_QUANT_BYTES];
        float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(MEMSPINE_DIM));
        float centroids[4] = {
            -1.51f * inv_sqrt_d,
            -0.45f * inv_sqrt_d,
            +0.45f * inv_sqrt_d,
            +1.51f * inv_sqrt_d
        };
        float scale = M_scale[slot];
        for (int i = 0; i < MEMSPINE_DIM; i++) {
            int code = (q[i / 4] >> ((i % 4) * 2)) & 0x3;
            out[i] = centroids[code] * scale;
        }
    }

    // ========================================================================
    // Cosine similarity between a query vector and a stored slot
    // ========================================================================
    float cosine_sim(int slot, const float *query) const {
        float stored[MEMSPINE_DIM];
        dequantize(slot, stored);
        float dot = 0, ns = 0, nq = 0;
        for (int i = 0; i < MEMSPINE_DIM; i++) {
            dot += stored[i] * query[i];
            ns  += stored[i] * stored[i];
            nq  += query[i]  * query[i];
        }
        return dot / (std::sqrt(ns * nq) + 1e-8f);
    }

    // ========================================================================
    // Top-K retrieval — returns K most similar stored patterns
    // ========================================================================
    struct ScoredSlot {
        int   slot;
        float score;
    };

    std::vector<ScoredSlot> retrieve_topk(const float *query, int k) const {
        std::vector<ScoredSlot> results;
        results.reserve(occupied_slots.size());

        for (int s : occupied_slots) {
            float sim = cosine_sim(s, query);
            results.push_back({s, sim});
        }

        if (static_cast<int>(results.size()) > k) {
            std::partial_sort(
                results.begin(), results.begin() + k, results.end(),
                [](const ScoredSlot &a, const ScoredSlot &b) {
                    return a.score > b.score;
                });
            results.resize(k);
        } else {
            std::sort(results.begin(), results.end(),
                      [](const ScoredSlot &a, const ScoredSlot &b) {
                          return a.score > b.score;
                      });
        }
        return results;
    }

    // ========================================================================
    // Save to binary (sparse format — only occupied slots + Omega)
    //
    // Format: [MSPN magic][dim][slots][n_occ][Omega][per-slot data...]
    // ========================================================================
    bool save(const char *path) const {
        FILE *f = fopen(path, "wb");
        if (!f) return false;

        int dim = MEMSPINE_DIM, slots = MEMSPINE_SLOTS;
        int n_occ = static_cast<int>(occupied_slots.size());

        if (fwrite("MSPN", 4, 1, f) != 1) { fclose(f); return false; }
        if (fwrite(&dim, 4, 1, f) != 1) { fclose(f); return false; }
        if (fwrite(&slots, 4, 1, f) != 1) { fclose(f); return false; }
        if (fwrite(&n_occ, 4, 1, f) != 1) { fclose(f); return false; }
        if (fwrite(Omega, sizeof(float) * MEMSPINE_DIM * MEMSPINE_DIM, 1, f) != 1) {
            fclose(f); return false;
        }

        for (int s : occupied_slots) {
            if (fwrite(&s, 4, 1, f) != 1) { fclose(f); return false; }
            if (fwrite(&M_quant[static_cast<size_t>(s) * MEMSPINE_QUANT_BYTES],
                        MEMSPINE_QUANT_BYTES, 1, f) != 1) { fclose(f); return false; }
            if (fwrite(&M_scale[s], 4, 1, f) != 1) { fclose(f); return false; }
        }

        fclose(f);
        printf("  [MemSpine] Saved %d patterns to %s\n", n_occ, path);
        return true;
    }

    // ========================================================================
    // Load from binary
    // ========================================================================
    bool load(const char *path) {
        FILE *f = fopen(path, "rb");
        if (!f) return false;

        char magic[4];
        int dim, slots, n_occ;

        if (fread(magic, 4, 1, f) != 1) { fclose(f); return false; }
        if (memcmp(magic, "MSPN", 4) != 0) {
            printf("  [MemSpine] ERROR: Invalid file format (bad magic)\n");
            fclose(f);
            return false;
        }

        if (fread(&dim, 4, 1, f) != 1) { fclose(f); return false; }
        if (fread(&slots, 4, 1, f) != 1) { fclose(f); return false; }
        if (fread(&n_occ, 4, 1, f) != 1) { fclose(f); return false; }

        if (dim != MEMSPINE_DIM || slots != MEMSPINE_SLOTS) {
            printf("  [MemSpine] ERROR: Dimension mismatch — file: %dx%d, expected: %dx%d\n",
                   dim, slots, MEMSPINE_DIM, MEMSPINE_SLOTS);
            fclose(f);
            return false;
        }

        if (fread(Omega, sizeof(float) * MEMSPINE_DIM * MEMSPINE_DIM, 1, f) != 1) {
            fclose(f); return false;
        }

        occupied_slots.clear();
        occupied_slots.reserve(n_occ);

        for (int i = 0; i < n_occ; i++) {
            int s;
            if (fread(&s, 4, 1, f) != 1) { fclose(f); return false; }
            if (fread(&M_quant[static_cast<size_t>(s) * MEMSPINE_QUANT_BYTES],
                       MEMSPINE_QUANT_BYTES, 1, f) != 1) { fclose(f); return false; }
            if (fread(&M_scale[s], 4, 1, f) != 1) { fclose(f); return false; }
            M_occupied[s] = 1;
            occupied_slots.push_back(s);
        }

        fclose(f);
        printf("  [MemSpine] Loaded %d patterns from %s\n", n_occ, path);
        return true;
    }

    /// Number of currently stored patterns
    int n_stored() const { return static_cast<int>(occupied_slots.size()); }
};

// ============================================================================
// C API for shared library (DLL / .so) — used by Python bindings
// ============================================================================
extern "C" {

#ifdef _WIN32
#define MS_EXPORT __declspec(dllexport)
#else
#define MS_EXPORT __attribute__((visibility("default")))
#endif

MS_EXPORT inline void* ms_create() {
    MemSpine* spine = new MemSpine();
    if (!spine->init(42)) {
        delete spine;
        return nullptr;
    }
    return spine;
}

MS_EXPORT inline void ms_destroy(void* spine_ptr) {
    if (spine_ptr) {
        MemSpine* spine = static_cast<MemSpine*>(spine_ptr);
        delete spine;
    }
}

MS_EXPORT inline int ms_store(void* spine_ptr, const float* vec) {
    if (!spine_ptr || !vec) return -1;
    MemSpine* spine = static_cast<MemSpine*>(spine_ptr);
    return spine->store(vec);
}

MS_EXPORT inline int ms_retrieve(void* spine_ptr, const float* query, int k, int* out_slots, float* out_scores) {
    if (!spine_ptr || !query || !out_slots || !out_scores) return 0;
    MemSpine* spine = static_cast<MemSpine*>(spine_ptr);
    auto results = spine->retrieve_topk(query, k);
    int count = static_cast<int>(results.size());
    for (int i = 0; i < count; i++) {
        out_slots[i] = results[i].slot;
        out_scores[i] = results[i].score;
    }
    return count;
}

MS_EXPORT inline bool ms_save(void* spine_ptr, const char* path) {
    if (!spine_ptr || !path) return false;
    MemSpine* spine = static_cast<MemSpine*>(spine_ptr);
    return spine->save(path);
}

MS_EXPORT inline bool ms_load(void* spine_ptr, const char* path) {
    if (!spine_ptr || !path) return false;
    MemSpine* spine = static_cast<MemSpine*>(spine_ptr);
    return spine->load(path);
}

MS_EXPORT inline int ms_n_stored(void* spine_ptr) {
    if (!spine_ptr) return 0;
    MemSpine* spine = static_cast<MemSpine*>(spine_ptr);
    return spine->n_stored();
}

MS_EXPORT inline void* ms_chunkstore_create() {
    return new ChunkStore();
}

MS_EXPORT inline void ms_chunkstore_destroy(void* store_ptr) {
    if (store_ptr) {
        ChunkStore* store = static_cast<ChunkStore*>(store_ptr);
        delete store;
    }
}

MS_EXPORT inline void ms_chunkstore_add(void* store_ptr, int slot_id, const char* text) {
    if (store_ptr && text) {
        ChunkStore* store = static_cast<ChunkStore*>(store_ptr);
        store->add(slot_id, text);
    }
}

// Thread-local buffer for returning strings
static thread_local std::string tl_chunk_ret;

MS_EXPORT inline const char* ms_chunkstore_get(void* store_ptr, int slot_id) {
    if (!store_ptr) return "";
    ChunkStore* store = static_cast<ChunkStore*>(store_ptr);
    tl_chunk_ret = store->get(slot_id);
    return tl_chunk_ret.c_str();
}

MS_EXPORT inline int ms_chunkstore_size(void* store_ptr) {
    if (!store_ptr) return 0;
    ChunkStore* store = static_cast<ChunkStore*>(store_ptr);
    return store->size();
}

MS_EXPORT inline bool ms_chunkstore_save(void* store_ptr, const char* path) {
    if (!store_ptr || !path) return false;
    ChunkStore* store = static_cast<ChunkStore*>(store_ptr);
    return store->save(path);
}

MS_EXPORT inline bool ms_chunkstore_load(void* store_ptr, const char* path) {
    if (!store_ptr || !path) return false;
    ChunkStore* store = static_cast<ChunkStore*>(store_ptr);
    return store->load(path);
}

} // extern "C"
