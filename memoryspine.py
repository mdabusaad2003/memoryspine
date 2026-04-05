import os
import sys
import ctypes
from typing import List, Tuple

# Load the shared library
lib_name = 'memspine.dll' if sys.platform == 'win32' else 'libmemspine.so'
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), lib_name)

if not os.path.exists(lib_path):
    raise FileNotFoundError(f"MemorySpine shared library not found at: {lib_path}. Please compile it first.")

lib = ctypes.CDLL(lib_path)

# ============================================================================
# Define Ctypes signatures
# ============================================================================

# void* ms_create()
lib.ms_create.argtypes = []
lib.ms_create.restype = ctypes.c_void_p

# void ms_destroy(void* spine_ptr)
lib.ms_destroy.argtypes = [ctypes.c_void_p]
lib.ms_destroy.restype = None

# int ms_store(void* spine_ptr, const float* vec)
lib.ms_store.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
lib.ms_store.restype = ctypes.c_int

# int ms_retrieve(void* spine_ptr, const float* query, int k, int* out_slots, float* out_scores)
lib.ms_retrieve.argtypes = [
    ctypes.c_void_p, 
    ctypes.POINTER(ctypes.c_float), 
    ctypes.c_int, 
    ctypes.POINTER(ctypes.c_int), 
    ctypes.POINTER(ctypes.c_float)
]
lib.ms_retrieve.restype = ctypes.c_int

# bool ms_save(void* spine_ptr, const char* path)
lib.ms_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.ms_save.restype = ctypes.c_bool

# bool ms_load(void* spine_ptr, const char* path)
lib.ms_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.ms_load.restype = ctypes.c_bool

# int ms_n_stored(void* spine_ptr)
lib.ms_n_stored.argtypes = [ctypes.c_void_p]
lib.ms_n_stored.restype = ctypes.c_int

# void* ms_chunkstore_create()
lib.ms_chunkstore_create.argtypes = []
lib.ms_chunkstore_create.restype = ctypes.c_void_p

# void ms_chunkstore_destroy(void* store_ptr)
lib.ms_chunkstore_destroy.argtypes = [ctypes.c_void_p]
lib.ms_chunkstore_destroy.restype = None

# void ms_chunkstore_add(void* store_ptr, int slot_id, const char* text)
lib.ms_chunkstore_add.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p]
lib.ms_chunkstore_add.restype = None

# const char* ms_chunkstore_get(void* store_ptr, int slot_id)
lib.ms_chunkstore_get.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.ms_chunkstore_get.restype = ctypes.c_char_p

# int ms_chunkstore_size(void* store_ptr)
lib.ms_chunkstore_size.argtypes = [ctypes.c_void_p]
lib.ms_chunkstore_size.restype = ctypes.c_int

# bool ms_chunkstore_save(void* store_ptr, const char* path)
lib.ms_chunkstore_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.ms_chunkstore_save.restype = ctypes.c_bool

# bool ms_chunkstore_load(void* store_ptr, const char* path)
lib.ms_chunkstore_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.ms_chunkstore_load.restype = ctypes.c_bool


class MemorySpine:
    """Python wrapper for the C++ MemorySpine engine (O(1) quantized storage)."""
    
    def __init__(self):
        self._spine = lib.ms_create()
        self._store = lib.ms_chunkstore_create()
        if not self._spine or not self._store:
            raise RuntimeError("Failed to allocate MemorySpine core.")

    def __del__(self):
        if hasattr(self, '_spine') and self._spine:
            lib.ms_destroy(self._spine)
        if hasattr(self, '_store') and self._store:
            lib.ms_chunkstore_destroy(self._store)

    def n_stored(self) -> int:
        """Return number of stored chunk embeddings."""
        return lib.ms_n_stored(self._spine)

    def num_chunks(self) -> int:
        """Return number of text chunks stored."""
        return lib.ms_chunkstore_size(self._store)

    def store(self, embedding: List[float], text: str) -> int:
        """Quantize and store an embedding vector alongside its text chunk."""
        # Convert python list to C float array
        c_emb = (ctypes.c_float * len(embedding))(*embedding)
        
        # Store in MemorySpine
        slot_id = lib.ms_store(self._spine, c_emb)
        if slot_id >= 0:
            # Add to ChunkStore
            c_text = text.encode('utf-8')
            lib.ms_chunkstore_add(self._store, slot_id, c_text)
        return slot_id

    def retrieve(self, query_embedding: List[float], k: int = 5) -> List[Tuple[int, float, str]]:
        """Retrieve top k most similar text chunks for a given query embedding."""
        c_emb = (ctypes.c_float * len(query_embedding))(*query_embedding)
        
        print("  [MemSpine] Allocating 27M slots x 197 bytes = 4.95 GB...")
        c_slots = (ctypes.c_int * k)()
        c_scores = (ctypes.c_float * k)()
        
        count = lib.ms_retrieve(self._spine, c_emb, k, c_slots, c_scores)
        
        results = []
        for i in range(count):
            slot = c_slots[i]
            score = c_scores[i]
            # Retrieve string
            text_ptr = lib.ms_chunkstore_get(self._store, slot)
            text = text_ptr.decode('utf-8') if text_ptr else ""
            results.append((slot, score, text))
            
        return results

    def save(self, base_path: str) -> bool:
        """Save memory to disk."""
        b_spine = f"{base_path}_ms.bin".encode('utf-8')
        b_store = f"{base_path}_chunks.dat".encode('utf-8')
        
        s1 = lib.ms_save(self._spine, b_spine)
        s2 = lib.ms_chunkstore_save(self._store, b_store)
        return s1 and s2

    def load(self, base_path: str) -> bool:
        """Load memory from disk."""
        b_spine = f"{base_path}_ms.bin".encode('utf-8')
        b_store = f"{base_path}_chunks.dat".encode('utf-8')
        
        s1 = lib.ms_load(self._spine, b_spine)
        s2 = lib.ms_chunkstore_load(self._store, b_store)
        return s1 and s2
