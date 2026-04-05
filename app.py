import os
import sys
import json
import time
import urllib.request
import urllib.error
import subprocess
import threading
from urllib.error import URLError

from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename
from memoryspine import MemorySpine

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
EMBED_DIR = os.path.join(BASE_DIR, 'embeddings')
SESSIONS_DIR = os.path.join(BASE_DIR, 'sessions')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')

DEFAULT_EMBED_URL = "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf"
EMBED_MODEL_NAME = "nomic-embed-text-v1.5.f16.gguf"

AVAILABLE_MODELS = {
    "llama3_8b": {
        "url": "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        "name": "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        "title": "LLaMA 3 8B Instruct (Q4_K_M)"
    }
}

CHAT_API = "http://127.0.0.1:8080/v1"
EMBEDDING_API = "http://127.0.0.1:8091"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EMBED_DIR, exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

app = Flask(__name__, template_folder='ui', static_folder='ui')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB max upload

# Global MemorySpine instance
spine = None
llm_proc = None
current_model_id = None
turn_counter = 0

# ============================================================================
# HELPERS: Background Server Management
# ============================================================================
def download_file(url, out_path, desc):
    """Download a file with a simple progress print."""
    print(f"[Setup] Downloading {desc} ...")
    def reporthook(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar * 1e2 / totalsize
            sys.stdout.write(f"\rDownloading: {percent:5.1f}%")
            sys.stdout.flush()
    try:
        urllib.request.urlretrieve(url, out_path, reporthook)
        print(f"\n[Setup] Successfully downloaded {desc}")
    except Exception as e:
        print(f"\n[Setup] Failed to download {desc}: {e}")

def check_and_download_dependencies():
    """Ensure llama-server and embedding model exist."""
    print("[Setup] Checking dependencies...")
    # 1. Check/Download llama-server
    srv_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    srv_bin_dir = os.path.join(BASE_DIR, "llama-bin")
    srv_path = os.path.join(srv_bin_dir, srv_name)
    embed_path = os.path.join(MODELS_DIR, EMBED_MODEL_NAME)
    
    # Check old root location for backwards compatibility
    if not os.path.exists(srv_path) and os.path.exists(os.path.join(BASE_DIR, srv_name)):
        srv_path = os.path.join(BASE_DIR, srv_name)

    if not os.path.exists(srv_path):
        print(f"[Setup] WARNING: {srv_name} not found in {srv_bin_dir}.")
        print("[Setup] Attempting to auto-download latest llama-server release from GitHub...")
        import platform, zipfile, tarfile
        
        system = platform.system().lower()
        arch = platform.machine().lower()
        
        # GitHub API for latest release
        api_url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
        try:
            req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=50) as response:
                release_data = json.loads(response.read().decode())
                
            download_url = None
            filename = None
            for asset in release_data.get('assets', []):
                name = asset['name'].lower()
                # Exclude GPU-specific builds to ensure base compatibility for embedding
                if 'cuda' in name or 'vulkan' in name or 'sycl' in name or 'rocm' in name:
                    continue
                
                if system == "windows" and "win" in name and "x64" in name and name.endswith(".zip"):
                    download_url = asset['browser_download_url']
                    filename = asset['name']
                    break
                elif system == "darwin" and "macos" in name and (("arm64" in name and arch in ["arm64", "aarch64"]) or ("x64" in name and arch not in ["arm64", "aarch64"])) and name.endswith(".tar.gz"):
                    download_url = asset['browser_download_url']
                    filename = asset['name']
                    break
                elif system == "linux" and "linux" in name and (("aarch64" in name and arch in ["arm64", "aarch64"]) or ("x64" in name and arch not in ["arm64", "aarch64"])) and name.endswith(".tar.gz"):
                    download_url = asset['browser_download_url']
                    filename = asset['name']
                    break
                    
            if download_url:
                arc_path = os.path.join(BASE_DIR, filename)
                download_file(download_url, arc_path, f"llama.cpp ({filename})")
                print("\n[Setup] Extracting...")
                
                os.makedirs(srv_bin_dir, exist_ok=True)
                
                if filename.endswith(".zip"):
                    with zipfile.ZipFile(arc_path, 'r') as zip_ref:
                        # We ONLY want llama-server.exe and any .dll files
                        for file_info in zip_ref.infolist():
                            fname = os.path.basename(file_info.filename).lower()
                            if fname == "llama-server.exe" or fname.endswith(".dll"):
                                file_info.filename = fname
                                zip_ref.extract(file_info, srv_bin_dir)
                elif filename.endswith(".tar.gz"):
                    with tarfile.open(arc_path, 'r:gz') as t:
                        for member in t.getmembers():
                            if member.isfile():
                                member.name = os.path.basename(member.name)
                                t.extract(member, srv_bin_dir)
                
                os.remove(arc_path)
                if not sys.platform == "win32":
                    os.chmod(srv_path, 0o755)
            else:
                print("\n[Setup] ERROR: Could not find suitable release artifact. Please download manually.")
        except Exception as e:
            print(f"\n[Setup] ERROR during auto-download: {e}")
            print("\n[Setup] Please download llama-server executable manually and place it in the same directory.")
    embed_path = os.path.join(MODELS_DIR, EMBED_MODEL_NAME)
    
    # 2. Check/Download Embedding Model
    embed_path = os.path.join(EMBED_DIR, EMBED_MODEL_NAME)
    if not os.path.exists(embed_path):
        download_file(DEFAULT_EMBED_URL, embed_path, EMBED_MODEL_NAME)
        
    # 3. Check/Download Generative LLM
    llm_info = AVAILABLE_MODELS["llama3_8b"]
    llm_path = os.path.join(MODELS_DIR, llm_info["name"])
    if not os.path.exists(llm_path):
        download_file(llm_info["url"], llm_path, llm_info["name"])

def start_server(port, model_name, is_embedding):
    """Start llama.cpp silently."""
    srv_name = "llama-server.exe" if sys.platform == "win32" else "./llama-server"
    srv_bin_dir = os.path.join(BASE_DIR, "llama-bin")
    srv_path = os.path.join(srv_bin_dir, srv_name)
    
    if not os.path.exists(srv_path) and os.path.exists(os.path.join(BASE_DIR, srv_name)):
        srv_path = os.path.join(BASE_DIR, srv_name)
    
    if is_embedding:
        model_path = os.path.join(EMBED_DIR, model_name)
    else:
        model_path = os.path.join(MODELS_DIR, model_name)
    
    if not os.path.exists(srv_path) or not os.path.exists(model_path):
        print(f"[Server] Missing executable or model for {model_name}. Skipping.")
        return None

    try:
        r = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
        if r.getcode() == 200:
            print(f"[Server] Server on port {port} is already running.")
            return None
    except:
        pass

    print(f"[Server] Starting server on http://127.0.0.1:{port} with {model_name}...")
    cmd = [srv_path, "-m", model_path, "-c", "8192", "-b", "8192", "--port", str(port)]
    if sys.platform != "win32":
        cmd.extend(["-ngl", "99"])  # enable GPU explicitly if not Windows CPU
    if is_embedding:
        cmd.append("--embedding")
    
    startupinfo = None
    if sys.platform == 'win32':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, startupinfo=startupinfo)
    return proc

def get_embedding(text):
    """Retrieve 768-dim embedding from llama-server using OpenAI API spec."""
    # Safety truncation for mega-queries exceeding Nomic's 8192-token window
    if len(text) > 2000:
        # Tightly constrain to 2000 chars. Perfect for semantic accuracy & tokenizer safety.
        text = text[:1000] + "\n...\n" + text[-1000:]
        
    data = json.dumps({
        "input": text,
        "model": EMBED_MODEL_NAME
    }).encode('utf-8')
    
    req = urllib.request.Request(
        f"{EMBEDDING_API}/v1/embeddings", 
        data=data, 
        headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req, timeout=240) as response:
            res = json.loads(response.read().decode('utf-8'))
            if 'data' in res and len(res['data']) > 0:
                emb = res['data'][0]['embedding']
                # Sometimes raw embeddings structure places it in a nested list
                if isinstance(emb, list) and len(emb) == 1 and isinstance(emb[0], list):
                    return emb[0]
                return emb
            return None
    except Exception as e:
        print(f"Embedding error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# FLASK WEB ENDPOINTS
# ============================================================================

@app.route("/")
def index():
    return render_template("index.html")



@app.route("/api/save_session", methods=["POST"])
def save_session():
    ok = spine.save(os.path.join(SESSIONS_DIR, "session"))
    return jsonify({"success": ok})

@app.route("/api/load_session", methods=["POST"])
def load_session():
    ok = spine.load(os.path.join(SESSIONS_DIR, "session"))
    return jsonify({"success": ok, "chunks": spine.n_stored()})

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOADS_DIR, filename)
    file.save(filepath)
    
    # 1. Read & chunk file
    text = ""
    if filename.lower().endswith('.pdf'):
        try:
            import pypdf
            with open(filepath, 'rb') as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        except ImportError:
            os.remove(filepath)
            return jsonify({"error": "pypdf library is not installed. Run 'pip install pypdf'."}), 500
        except Exception as e:
            os.remove(filepath)
            return jsonify({"error": f"Failed to read PDF: {str(e)}"}), 400
    else:
        # Handles .md, .txt, .json, .cpp, .py, .bat, etc.
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except Exception as e:
            os.remove(filepath)
            return jsonify({"error": f"Failed to read text file: {str(e)}"}), 400
    
    CHUNK_SIZE = 1500
    OVERLAP = 200
    chunks = []
    pos = 0
    while pos < len(text):
        end = min(pos + CHUNK_SIZE, len(text))
        if end < len(text):
            # find good boundary
            for i in range(end, pos + CHUNK_SIZE//2, -1):
                if text[i] in ['.', '!', '?', '\n']:
                    end = i + 1
                    break
        chunks.append(text[pos:end])
        pos = max(pos + 1, end - OVERLAP)
        if end >= len(text): break

    # 2. Embed and store
    ok_count = 0
    for chunk in chunks:
        emb = get_embedding("search_document: " + chunk)
        if emb and len(emb) == 768:
            spine.store(emb, chunk)
            ok_count += 1
            
    # Remove temporary file
    os.remove(filepath)
    return jsonify({"success": True, "chunks": ok_count})

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Turn 1: User says X, Assistant says Y
    Turn 2: User asks Z
    
    To maintain sequential clarity, we track 'turn_counter' and inject it natively.
    """
    global turn_counter
    turn_counter += 1
    
    data = request.json
    messages = data.get("messages", [])
    model = data.get("model", "")
    
    if not messages:
        return jsonify({"error": "No messages array provided"}), 400
        
    user_query = messages[-1]["content"]

    # 1. Retrieve Context if MemorySpine has data
    context_str = ""
    if spine.n_stored() > 0:
        q_emb = get_embedding("search_query: " + user_query)
        if q_emb and len(q_emb) == 768:
            results = spine.retrieve(q_emb, k=15)
            # Filter and sort by Turn order if it's conversation history (to prevent chronological blurring)
            filtered = []
            for slot, score, chunk in results:
                if score > 0.15:
                    filtered.append((score, chunk))
                    
            # Basic heuristic to sort conversational turns chronologically
            def extract_turn(text):
                if "[Turn " in text:
                    try:
                        return int(text.split("[Turn ")[1].split("]")[0])
                    except:
                        return 0
                return 0
                
            filtered.sort(key=lambda x: extract_turn(x[1]))
            
            print(f"Retrieval results for query '{user_query}':")
            for score, chunk in filtered:
                print(f" - Score: {score:.2f} | {chunk[:30]}...")
                context_str += f"\n{chunk}\n"
    
    if context_str:
        print("-> Context injected successfully.")
    else:
        print("-> No adequate context found to inject.")
    
    # Inject context gracefully as conversational pre-loading
    system_prompt = "You are a concise, helpful assistant. Follow user instructions and rules perfectly."
    
    final_messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    if context_str:
        final_messages.append({"role": "user", "content": f"[Recalled Memory/Documents]:\n{context_str}"})
        final_messages.append({"role": "assistant", "content": "I have reviewed these recovered memories. How can I help you?"})
        
    final_messages.append({"role": "user", "content": user_query})

    payload = json.dumps({
        "model": model,
        "messages": final_messages,
        "temperature": 0.7,
        "frequency_penalty": 1.2,
        "presence_penalty": 1.2,
        "repeat_penalty": 1.1,
        "max_tokens": 4096,
        "stream": True
    }).encode('utf-8')
    
    def generate():
        req = urllib.request.Request(
            f"{CHAT_API}/chat/completions",
            data=payload,
            headers={'Content-Type': 'application/json'}
        )
        assistant_reply = ""
        import time
        eval_start_time = time.time()
        start_time = None
        token_count = 0
        prefill_time = 0.0
        
        try:
            with urllib.request.urlopen(req, timeout=1200) as resp:
                for line in resp:
                    if line:
                        yield line
                        try:
                            l_dec = line.decode('utf-8').strip()
                            if l_dec.startswith('data: ') and not l_dec.endswith('[DONE]'):
                                d = json.loads(l_dec[6:])
                                if 'choices' in d and len(d['choices']) > 0:
                                    delta = d['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        if start_time is None:
                                            start_time = time.time()
                                            prefill_time = start_time - eval_start_time
                                        assistant_reply += delta['content']
                                        token_count += 1
                        except:
                            pass
                            
                # When stream completes successfully, compute physics and yield final badge
                if start_time:
                    elapsed = time.time() - start_time
                    tps = token_count / elapsed if elapsed > 0 else 0
                    metrics = f"\n\n<div style='font-size:0.8em; color:#64748b; font-style:italic;'>[Diagnostics] Prompt Context Eval: {prefill_time:.2f}s | Generation Speed: {tps:.2f} tokens/sec ({token_count} tkn)</div>"
                    final_chunk = {"choices": [{"delta": {"content": metrics}}]}
                    yield f"data: {json.dumps(final_chunk)}\n\n".encode('utf-8')
                    assistant_reply += metrics
            
        except GeneratorExit:
            print("-> Client stream aborted by STOP button.")
            pass
        except urllib.error.HTTPError as he:
            body = he.read().decode('utf-8', errors='ignore')
            err = {"error": f"LLM Server HTTP {he.code}: {body}"}
            print(f"LLM Error {he.code}:", body)
            yield f"data: {json.dumps(err)}\n\n".encode('utf-8')
        except Exception as e:
            err = {"error": f"LLM Server API Error: {str(e)}"}
            print("LLM Error:", str(e))
            yield f"data: {json.dumps(err)}\n\n".encode('utf-8')
        finally:
            # --- AUTO MEMORY INGESTION EXECUTED EVEN IF STOPPED ---
            if assistant_reply:
                mem_block = f"[Turn {turn_counter}] User Message:\n{user_query}\n\nAssistant Reply:\n{assistant_reply}"
                
                CHUNK_SIZE = 1500
                OVERLAP = 200
                chunks = []
                pos = 0
                while pos < len(mem_block):
                    end = min(pos + CHUNK_SIZE, len(mem_block))
                    if end < len(mem_block):
                        for i in range(end, pos + CHUNK_SIZE//2, -1):
                            if mem_block[i] in ['.', '!', '?', '\n']:
                                end = i + 1
                                break
                    chunks.append(mem_block[pos:end])
                    pos = max(pos + 1, end - OVERLAP)
                    if end >= len(mem_block): break

                ok_count = 0
                for chunk in chunks:
                    emb = get_embedding("search_document: " + chunk)
                    if emb and len(emb) == 768:
                        spine.store(emb, chunk)
                        ok_count += 1
                        
                print(f"-> Conversation pair [Turn {turn_counter}] autonomously ingested as {ok_count} chunks into MemorySpine.")

    return Response(generate(), mimetype='text/event-stream')

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("\n=== MemorySpine ChatGPT-like Local GUI ===\n")
    check_and_download_dependencies()
    
    embed_proc = start_server(8091, EMBED_MODEL_NAME, True)
    
    # Auto-boot LLM so users don't need manual UI intervention anymore.
    llm_info = AVAILABLE_MODELS["llama3_8b"]
    llm_proc = start_server(8080, llm_info["name"], False)
    
    print("[Spine] Initializing O(1) Memory Engine...")
    spine = MemorySpine()
    
    print("\n   => Server ready at: http://127.0.0.1:5000\n")
    
    try:
        app.run(host="127.0.0.1", port=5000, threaded=True)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        if embed_proc:
            embed_proc.terminate()
        if llm_proc:
            llm_proc.terminate()
