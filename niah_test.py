import os
import sys
import json
import time
import random
import urllib.request
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from memoryspine import MemorySpine

EMBEDDING_API = "http://127.0.0.1:8091"
QWEN_API = "http://127.0.0.1:8080/v1"

# The Needle
NEEDLE_TEXT = "\n\n--- THE SECRET PASSCODE FOR THE UNDERGROUND VAULT IS: BANANA_MANGO_42 ---\n\n"
QUERY_TEXT = "What is the secret passcode for the underground vault?"

def get_embedding(text):
    data = json.dumps({
        "input": text,
        "model": "nomic-embed-text-v1.5.f16.gguf"
    }).encode('utf-8')
    req = urllib.request.Request(
        f"{EMBEDDING_API}/v1/embeddings",
        data=data,
        headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            res = json.loads(response.read().decode('utf-8'))
            if 'data' in res and len(res['data']) > 0:
                emb = res['data'][0]['embedding']
                if isinstance(emb, list) and len(emb) == 1 and isinstance(emb[0], list):
                    return emb[0]
                return emb 
    except Exception as e:
        print(f"Embedding failed: {e}")
    return None

def ask_llm(context_str, query):
    prompt_text = f"You are an expert analyst. Answer the user's question directly and concisely using ONLY the provided context.\n\n### CONTEXT:\n{context_str}\n\n### QUESTION:\n{query}\n\n### ANSWER:"
    
    final_messages = [{"role": "user", "content": prompt_text}]

    payload = json.dumps({
        "model": "llama3",
        "messages": final_messages,
        "temperature": 0.5,
        "max_tokens": 512,
        "stream": False
    }).encode('utf-8')
    
    req = urllib.request.Request(
        f"{QWEN_API}/chat/completions",
        data=payload,
        headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"LLM Error: {e}"

def generate_haystack(target_tokens):
    # ~4 chars per token roughly
    target_chars = target_tokens * 4
    
    doc_path = os.path.join(os.path.dirname(BASE_DIR), "test_longdoc.txt")
    if not os.path.exists(doc_path):
        raise Exception(f"Haystack source not found at {doc_path}!")
        
    with open(doc_path, "r", encoding="utf-8") as f:
        base_text = f.read()
        
    haystack = ""
    while len(haystack) < target_chars:
        haystack += base_text + "\n\n"
        
    return haystack[:target_chars]

def run_niah(token_depths):
    print("=== MemorySpine Needle In A Haystack (NIAH) Test ===")
    
    # Check if servers are running
    try:
        urllib.request.urlopen(f"{EMBEDDING_API}/health", timeout=2)
    except:
        print("ERROR: Formatting embedding server on 8091 is not running!")
        return

    results = []
    
    for tokens in token_depths:
        print(f"\n--- Testing {tokens} Tokens ---")
        haystack = generate_haystack(tokens)
        
        # Sentence-aligned chunking based on Section 3.7 of Architecture Paper
        CHUNK_SIZE = 1500
        OVERLAP = 200
        chunks = []
        
        i = 0
        while i < len(haystack):
            end = i + CHUNK_SIZE
            if end >= len(haystack):
                chunks.append(haystack[i:end])
                break
            
            # Align boundary backward to nearest punctuation within the second half of the chunk
            search_start = max(i, end - (CHUNK_SIZE // 2))
            last_punc = -1
            for p in ['.', '!', '?', '\n']:
                pos = haystack.rfind(p, search_start, end)
                if pos > last_punc:
                    last_punc = pos
                    
            if last_punc != -1:
                end = last_punc + 1 # Include the punctuation
                
            chunks.append(haystack[i:end])
            i = end - OVERLAP
            
        # GUARANTEED NEEDLE ISOLATION: Inject exactly into the middle of a random chunk
        if len(chunks) > 5:
            target_idx = random.randint(len(chunks) // 10, len(chunks) - (len(chunks) // 10))
            mid = len(chunks[target_idx]) // 2
            chunks[target_idx] = chunks[target_idx][:mid] + NEEDLE_TEXT + chunks[target_idx][mid:]
            
        print(f"Generated {len(chunks)} chunks.")
        
        # Instantiate fresh MemorySpine
        print("Allocating MemorySpine C++ (27M)...")
        spine = MemorySpine()
        
        print("Embedding and Storing...")
        start_time = time.time()
        for idx, chunk in enumerate(chunks):
            emb = get_embedding("search_document: " + chunk)
            if emb:
                spine.store(emb, chunk)
            if idx % 100 == 0 and idx > 0:
                print(f"  ...processed {idx}/{len(chunks)} chunks")
                
        emb_time = time.time() - start_time
        print(f"Stored {spine.num_chunks()} unique chunks in {emb_time:.1f}s")
        
        # Retrieval Phase
        print("Retrieving...")
        q_emb = get_embedding("search_query: " + QUERY_TEXT)
        retrieved_data = spine.retrieve(q_emb, k=15)
        
        needle_found_rank = -1
        context_block = ""
        for rank, (slot, score, rchunk) in enumerate(retrieved_data):
            if score > 0.15:
                context_block += f"\n{rchunk}\n"
            if "BANANA_MANGO_42" in rchunk:
                needle_found_rank = rank + 1
                
        if needle_found_rank > 0:
            print(f"✅ Needle SUCCESS: Retained at Rank {needle_found_rank}")
            retrieval_success = True
        else:
            print(f"❌ Needle FAILURE: Dropped from Top-15 due to quantization.")
            retrieval_success = False
            
        # Generation Phase
        print("Querying LLM...")
        llm_reply = ask_llm(context_block, QUERY_TEXT)
        print(f"LLM Answer: {llm_reply.strip()}")
        
        llm_reply_lower = llm_reply.lower().replace("_", " ")
        llm_pass = "banana mango 42" in llm_reply_lower or "banana_mango_42" in llm_reply_lower
        
        results.append({
            "DepthTokens": tokens,
            "Chunks": len(chunks),
            "RetrievalRank": needle_found_rank,
            "RetrievalSuccess": retrieval_success,
            "LLMSuccess": llm_pass,
            "LLMOutput": llm_reply.strip().replace("\n", " ")
        })
        
        # Free memory before next run
        del spine
        print("Freed MemorySpine.")
        
    print("\n=== Final Results ===")
    for r in results:
        print(f"[{r['DepthTokens']} Tokens] -> Retrieved: {r['RetrievalSuccess']} | LLM: {r['LLMSuccess']}")
        
    # Write to CSV
    with open("niah_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["DepthTokens", "Chunks", "RetrievalRank", "RetrievalSuccess", "LLMSuccess", "LLMOutput"])
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    depths = [10000, 50000, 100000, 250000, 500000, 1000000]
    run_niah(depths)
