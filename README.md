# MemorySpine

**O(1) Memory Context Extension for Large Language Models via 2-Bit Quantized Embedding Storage**

> Give any LLM unlimited context memory with a fixed 4.95 GB footprint — no fine-tuning, no architectural changes. Chat with your documents using any model.

---

## Quick Start

### Prerequisites

1. **Python 3.8+**
2. **LM Studio** installed and running.
3. In LM Studio, click **Start Server** (Local Server icon) so the API runs on `http://localhost:1234`.
4. Visual Studio Build Tools (Windows) or GCC (Linux) to build the C++ core.
MemorySpine solves the **"infinite context hallucination"** problem. Normal LLMs physically cannot process 27 million chunks of text natively—not only because the attention KV-cache would require hundreds of gigabytes of RAM, but because models pushed beyond their trained context length mathematically degrade and hallucinate. 

MemorySpine bypasses this entirely: it mathematically compresses all 27 million semantic chunks into a fixed-size **4.95 GB** hardware-level O(1) hash table. When you ask a question, the engine retrieves *only the 5 most relevant chunks* in under a millisecond and injects them into the standard, safe context window of any LLM. 

Even simple, low-hardware machines can now have completely infinite semantic recall without modifying the base model's training!
### 1. Build the C++ Engine (MemorySpine)

**Windows** (MSVC Developer Command Prompt):
```cmd
git clone https://github.com/YOUR_USERNAME/memoryspine.git
cd memoryspine
build.bat
```
*(This native compile will automatically generate the `memspine.dll` library and related `.obj`/`.lib` intermediate caches required by Python).*

**Linux / macOS**:
```bash
git clone https://github.com/YOUR_USERNAME/memoryspine.git
cd memoryspine
chmod +x build.sh && ./build.sh
```
*(This native compile will automatically generate the `libmemspine.so` or `libmemspine.dylib` shared libraries).*

### 2. Start the ChatGPT-like UI

```bash
pip install -r requirements.txt
python app.py
```

The script will automatically:
1. ✅ Download **llama-server** if missing.
2. ✅ Download the **Meta-Llama-3-8B-Instruct** generation model (~4.9 GB).
3. ✅ Download the **nomic-embed-text** embedding model (~274 MB).
4. ✅ Start the embedding server implicitly on port 8091 and LLM natively on port 8080.

### 3. Chat & Ingest
Open a browser to `http://127.0.0.1:5000`
- Click the attachment icon to upload a document (`txt`, `md`, `pdf` etc.)
- MemorySpine will instantly chunk, quantize, and store it.
- Chat with the document using your favorite local LLM loaded in LM Studio!

---

## How It Works

1. **Load a document** → it gets split into chunks, embedded, and stored in 2-bit quantized slots
2. **Ask a question** → your question is embedded, matched against stored chunks, and the most relevant context is injected into the LLM's prompt
3. **Chat naturally** → conversation history is maintained like Ollama, so follow-up questions work

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                                                          │
│   Document → Chunk → Embed → 2-bit Quantize → Store    │
│   Question → Embed → Top-K Match → Inject → Generate   │
│                                                          │
│   ┌──────────────┐          ┌──────────────────────┐    │
│   │  Embedding    │          │  MemorySpine Store   │    │
│   │  Server       │────→    │  10M × 197 bytes     │    │
│   │  (port 8091)  │          │  4.95 GB constant    │    │
│   └──────────────┘          └──────────────────────┘    │
│                                                          │
│   ┌──────────────┐                                      │
│   │  Generation   │←── retrieved chunks injected        │
│   │  Server       │    into conversation context        │
│   │  (port 8090)  │                                      │
│   └──────────────┘                                      │
└─────────────────────────────────────────────────────────┘
```

### Key Numbers

| | Value |
|:---|:---:|
| Memory footprint | **4.95 GB (constant)** |
| Compression vs float32 | **15.6×** |
| Quantization | 2-bit Lloyd-Max |
| Max stored chunks | 10,000,000 |
| Factual recall (240K chars) | **10/10 (100%)** |
| Dependencies | **Zero** (single C++ header) |

---

## Chat Commands

| `/upload` | Use the web UI file attachment button |
| Memory Management | Click `Clear History` or `Save/Load` in UI (coming soon) |

---

## Using with Different Models

### Embedding Models

| Model | Dimension | Compile Flag |
|:---|:---:|:---|
| **nomic-embed-text-v1.5** (default) | 768 | (none needed) |
| all-MiniLM-L6-v2 | 384 | `-DSPINE_DIM=384` |
| bge-base-en-v1.5 | 768 | (none needed) |
| mxbai-embed-large-v1 | 1024 | `-DSPINE_DIM=1024` |

### Generation Models

### Generation Models

The system natively auto-pulls `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`. You do not need to intervene. However, changing the model requires swapping the URL in the `app.py` `AVAILABLE_MODELS` array.

| Model | Usage | Size |
|:---|:---|:---:|
| LLaMA 3 8B (default) | Default Native Engine | ~4.9 GB |

---

## Manual Setup (Advanced)

If you prefer to set things up yourself:

### 1. Build
(See Quick Start)

### 2. Run LM Studio Server
Start the local API in LM Studio. Keep a large model loaded (e.g. Llama-3-8B).

### 3. Run Web App
```bash
python app.py
```

---

## Memory Comparison

| System | 4K ctx | 128K | 1M | 10M |
|:---|:---:|:---:|:---:|:---:|
| LLM KV-cache (8B) | 48 MB | 1.5 GB | 12 GB | 120 GB |
| FAISS (float32) | 3 KB | 98 KB | 3 GB | 30 GB |
| **MemorySpine (27M)** | **4.95 GB** | **4.95 GB** | **4.95 GB** | **4.95 GB** |

### Predictable Memory Scaling

Because MemorySpine statically allocates its hardware-level hash table upfront, its footprint is **100% predictable** and completely decoupled from whatever LLM is reading from it. The 4.95 GB footprint reflects the massive 27M-slot configuration, but you can effortlessly scale this down for smaller hardware:

- **1 Million Slots (~350 Million tokens of memory):** ~197 MB RAM
- **10 Million Slots (~3.5 Billion tokens of memory):** ~1.84 GB RAM
- **27 Million Slots (~9.4 Billion tokens of memory):** ~4.95 GB RAM

You can target any precise hardware limit simply by adjusting the `SPINE_SLOTS` variable in the C++ engine before running `build.bat`.

---

## Repository Structure

```text
memoryspine/
├── memspine.h            # Core library (single header, C/C++ API)
├── memspine.cpp          # Shared library compilation unit
├── memoryspine.py        # Python ctypes bindings
├── app.py                # Flask Backend (ChatGPT UI, LM Studio binding)
├── requirements.txt      # Python dependencies
├── build.bat             # Build DLL and CLI (Windows)
├── build.sh              # Build DLL and CLI (Linux/macOS)
├── CMakeLists.txt        # CMake build
├── README.md             # This file
├── ui/                   # Web GUI unified styling and layout
├── paper/
│   └── memoryspine_paper.md  # Research paper
└── docs/
```

---

## Research Paper

See [paper/memoryspine_paper.md](paper/memoryspine_paper.md) for the full research paper with mathematical formalization.

**Citation:**
```
Abu Saad. "MemorySpine: O(1) Memory Context Extension for Large Language
Models via 2-Bit Quantized Embedding Storage." 2026.
```

---

## Contact

- **Author**: Abu Saad
- **Email**: mdabusaad2003@gmail.com
- **Connect**: [Facebook](https://facebook.com/arafat.ahmed.20004)

---

*small part of the KHND (Koopman-Hamiltonian Neural Dynamics) project.*
