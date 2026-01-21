from fastapi import FastAPI, HTTPException, UploadFile, File, Security, Depends, Header
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from dotenv import load_dotenv
import os
import re
import json
from datetime import datetime
from pypdf import PdfReader

# LlamaIndex & ChromaDB Imports
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Document, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.mistralai import MistralAI

# ========================
# 1. SETUP ENV & CONSTANTS
# ========================
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "rahasia_admin")
DB_FILE = "divisions_db.json"
def slugify(text: str) -> str:
    """Mengubah string menjadi format ID yang aman (contoh: 'Human Capital' -> 'human_capital')"""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unnamed_division"

def load_divisions_from_db():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_divisions_to_db(divisions_data):
    try:
        with open(DB_FILE, "w") as f:
            json.dump(divisions_data, f, indent=4)
    except Exception as e:
        print(f"Gagal menyimpan database JSON: {e}")


UNANSWERED_DB_FILE = "unanswered_db.json"
def load_unanswered_from_db():
    if os.path.exists(UNANSWERED_DB_FILE):
        try:
            with open(UNANSWERED_DB_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_unanswered_to_db(data):
    try:
        with open(UNANSWERED_DB_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Gagal menyimpan unanswered db: {e}")

STATS_DB_FILE = "stats_db.json"
DEFAULT_STATS = {
    "Jan": 0, "Feb": 0, "Mar": 0, "Apr": 0,
    "May": 0, "Jun": 0, "Jul": 0, "Aug": 0,
    "Sep": 0, "Oct": 0, "Nov": 0, "Dec": 0
}
def load_stats_from_db():
    if os.path.exists(STATS_DB_FILE):
        try:
            with open(STATS_DB_FILE, "r") as f:
                data = json.load(f)
                return {**DEFAULT_STATS, **data}
        except Exception:
            return DEFAULT_STATS.copy()
    return DEFAULT_STATS.copy()

def save_stats_to_db(data):
    try:
        with open(STATS_DB_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Gagal menyimpan stats db: {e}")

DOCUMENTS_DB_FILE = "documents_db.json"
FAQS_DB_FILE = "faqs_db.json"

def load_documents_from_db():
    if os.path.exists(DOCUMENTS_DB_FILE):
        try:
            with open(DOCUMENTS_DB_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_documents_to_db(data):
    try:
        with open(DOCUMENTS_DB_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Gagal menyimpan documents db: {e}")

def load_faqs_from_db():
    if os.path.exists(FAQS_DB_FILE):
        try:
            with open(FAQS_DB_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_faqs_to_db(data):
    try:
        with open(FAQS_DB_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Gagal menyimpan faqs db: {e}")

if not MISTRAL_API_KEY or not CHROMA_API_KEY:
    raise ValueError("MISTRAL_API_KEY dan CHROMA_API_KEY wajib diisi di .env")

# ========================
# 3. INITIALIZE CLIENTS
# ========================
llm = MistralAI(api_key=MISTRAL_API_KEY, model="mistral-large-latest")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

try:
    client = chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
    )
except Exception as e:
    print(f"Warning: Koneksi ChromaDB gagal saat startup. {e}")
    client = None

# ========================
# 4. SYNC DATA (STARTUP)
# ========================
# Load dari JSON Local
saved_divisions = load_divisions_from_db()

# Ambil list ID dari Chroma (Sumber Kebenaran Fisik)
try:
    _chroma_colls = client.list_collections() if client else []
    valid_ids = [c.name for c in _chroma_colls]
except Exception:
    valid_ids = []

DIVISIONS = []

# --- BAGIAN INI DIUBAH AGAR MULAI DARI NOL ---
if not saved_divisions and not valid_ids:
    # Mulai dengan kosong (User harus input manual)
    DIVISIONS = []
    save_divisions_to_db(DIVISIONS)
else:
    # 1. Prioritaskan data JSON (karena ada nama panjang & deskripsi)
    for div in saved_divisions:
        DIVISIONS.append(div)
    
    # 2. Cek "Orphaned" Collections di Chroma (Ada di cloud, gak ada di JSON)
    existing_ids = [d["id"] for d in DIVISIONS]
    for cid in valid_ids:
        if cid not in existing_ids:
            new_entry = {"id": cid, "name": cid.replace("_", " ").title(), "description": ""}
            DIVISIONS.append(new_entry)
            
    save_divisions_to_db(DIVISIONS)

# Global Variables
FAQS: List[Dict] = load_faqs_from_db()
DOCUMENTS: List[Dict] = load_documents_from_db()
FAQ_COUNTER = (max([f["id"] for f in FAQS]) + 1) if FAQS else 1
UNANSWERED: List[Dict] = load_unanswered_from_db()
UNANSWERED_COUNTER = (max([u["id"] for u in UNANSWERED]) + 1) if UNANSWERED else 1
MONTHLY_HITS = load_stats_from_db()

# ========================
# 5. PROMPT TEMPLATE
# ========================
def get_dynamic_prompt_template(current_div_id: str):
    other_divisions = [d for d in DIVISIONS if d["id"] != current_div_id]
    
    redirect_list_str = ""
    for d in other_divisions:
        desc = d.get("description", "Layanan Divisi")
        redirect_list_str += f"- Topik: {d['name']} ({desc}) -> [[REDIRECT:{d['id']}]]\n"

    # Jika tidak ada divisi lain, kosongkan list redirect
    if not redirect_list_str:
        redirect_list_str = "(Tidak ada divisi lain yang tersedia saat ini)"

    template_str = f"""
Anda adalah asisten virtual profesional untuk PT Pindad di Divisi: {current_div_id}.
Tugas Anda adalah menjawab pertanyaan pengguna dengan gaya bahasa natural, ramah, dan langsung pada intinya.

ATURAN KRUSIAL:
1. Jawablah seolah-olah Anda memiliki pengetahuan tersebut sendiri.
2. Jawaban harus sopan, formal, dan membantu.

LOGIKA PENANGANAN PERTANYAAN:

1. **JAWABAN LANGSUNG**
   Jika pertanyaan RELEVAN dengan divisi ini ({current_div_id}) dan informasinya ada di konteks:
   - Jawab langsung pertanyaannya secara lengkap.

2. **SALAH DIVISI (REDIRECT)**
   Jika pertanyaan TIDAK RELEVAN, cek daftar berikut:
   {redirect_list_str}
   Contoh: "Mohon maaf, layanan tersebut ditangani divisi lain. [[REDIRECT:nama_divisi_lain]]"

3. **KONTAK MANUAL**
   Jika relevan tapi tidak ada jawaban detail di dokumen, arahkan ke kontak PIC terkait.

Context information is below.
---------------------
{{context_str}}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {{query_str}}
Answer:
"""
    return PromptTemplate(template_str)

# ========================
# 6. FASTAPI APP SETUP
# ========================
app = FastAPI(title="PINDAD Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Models
class ChatRequest(BaseModel):
    session_id: str
    division: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    answer: str

class NewDivision(BaseModel):
    name: str
    description: str = ""

class UpdateDivisionDescription(BaseModel):
    description: str

class NewFaq(BaseModel):
    division_id: str
    question: str
    answer: str

# Helper: Get Index
def get_index_for_division(division_id: str) -> VectorStoreIndex:
    try:
        coll = client.get_collection(division_id)
    except Exception:
        coll = client.create_collection(division_id)

    vector_store = ChromaVectorStore(chroma_collection=coll)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    return index

# Auth Helper
def get_admin_token(x_admin_secret: str = Header(None)):
    if x_admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

# ========================
# 7. ENDPOINTS
# ========================

@app.get("/divisions")
async def list_divisions():
    return DIVISIONS

@app.post("/division")
async def create_division(data: NewDivision):
    global DIVISIONS
    
    div_id = slugify(data.name)

    if any(d["id"] == div_id for d in DIVISIONS):
        raise HTTPException(status_code=400, detail="Divisi sudah ada")

    try:
        try:
            client.get_collection(div_id)
        except Exception:
            client.create_collection(div_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    new_entry = {"id": div_id, "name": data.name, "description": data.description}
    DIVISIONS.append(new_entry)
    save_divisions_to_db(DIVISIONS)

    return new_entry

@app.put("/division/{div_id}")
async def update_division_description(div_id: str, data: UpdateDivisionDescription):
    global DIVISIONS
    div_found = None
    for d in DIVISIONS:
        if d["id"] == div_id:
            d["description"] = data.description
            div_found = d
            break
            
    if not div_found:
        raise HTTPException(status_code=404, detail="Divisi tidak ditemukan")
    
    save_divisions_to_db(DIVISIONS)
    return div_found

@app.delete("/division/{div_id}")
async def delete_division(div_id: str):
    global DIVISIONS, FAQS, DOCUMENTS
    try:
        client.delete_collection(div_id)
    except Exception:
        pass 
        
    DIVISIONS = [d for d in DIVISIONS if d["id"] != div_id]
    FAQS = [f for f in FAQS if f["division_id"] != div_id]
    DOCUMENTS = [doc for doc in DOCUMENTS if doc["division_id"] != div_id]
    
    save_divisions_to_db(DIVISIONS)
    save_faqs_to_db(FAQS)
    save_documents_to_db(DOCUMENTS)
    return {"detail": "Divisi dihapus"}

@app.post("/upload/{division_id}")
async def upload_file(division_id: str, file: UploadFile = File(...)):
    global DOCUMENTS 
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Hanya file .pdf yang didukung")

    if not os.path.exists("data"):
        os.makedirs("data")
        
    file_path = os.path.join("data", file.filename)
    content_bytes = await file.read()
    with open(file_path, "wb") as f:
        f.write(content_bytes)

    try:
        coll = client.get_collection(division_id)
        pass 
    except Exception:
        pass

    text_content = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            extract = page.extract_text()
            if extract:
                text_content += extract + "\n"
    except Exception:
        raise HTTPException(status_code=500, detail="Gagal memproses PDF")

    if not text_content.strip():
        raise HTTPException(status_code=400, detail="PDF kosong/gambar scan")

    doc = Document(text=text_content, metadata={"filename": file.filename, "division": division_id})
    index = get_index_for_division(division_id)
    index.insert(doc)

    DOCUMENTS.append({
        "filename": file.filename,
        "division_id": division_id,
        "uploaded_at": datetime.now().isoformat()
    })

    save_documents_to_db(DOCUMENTS)

    return {"detail": "Upload berhasil dan diindex."}

@app.get("/admin/download/pdf/{filename}", dependencies=[Depends(get_admin_token)])
async def download_pdf_admin(filename: str):
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    file_path = os.path.join("data", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="application/pdf", filename=filename)

@app.get("/faqs")
async def get_faqs():
    doc_faqs = []
    for idx, doc in enumerate(DOCUMENTS):
        doc_faqs.append({
            "id": f"doc_{idx}",
            "division_id": doc["division_id"],
            "question": f"File: {doc['filename']}",
            "answer": f"Uploaded: {doc['uploaded_at']}"
        })
    return FAQS + doc_faqs

@app.post("/faq")
async def add_faq(data: NewFaq):
    global FAQ_COUNTER, FAQS
    faq = {
        "id": FAQ_COUNTER,
        "division_id": data.division_id,
        "question": data.question,
        "answer": data.answer,
    }
    FAQ_COUNTER += 1
    FAQS.append(faq)
    
    text = f"Pertanyaan: {data.question}\nJawaban: {data.answer}"
    doc = Document(text=text, metadata={"division": data.division_id, "type": "faq"})
    index = get_index_for_division(data.division_id)
    index.insert(doc)
    
    save_faqs_to_db(FAQS)

    return faq

@app.get("/stats")
async def get_stats():
    monthly_list = [{"month": k, "count": v} for k, v in MONTHLY_HITS.items()]
    return {
        "monthly": monthly_list,
        "total_faqs": len(FAQS) + len(DOCUMENTS), 
        "total_unanswered": len(UNANSWERED),
        "total_documents": len(DOCUMENTS)
    }

@app.get("/unanswered")
async def get_unanswered():
    return UNANSWERED

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    global UNANSWERED, UNANSWERED_COUNTER, MONTHLY_HITS

    # 1. Update Statistik Hits Bulanan
    current_month = datetime.now().strftime("%b") 
    if current_month in MONTHLY_HITS:
        MONTHLY_HITS[current_month] += 1
    else:
        MONTHLY_HITS[current_month] = 1
    
    save_stats_to_db(MONTHLY_HITS)

    # 2. Siapkan Index & Engine
    division_id = req.division
    index = get_index_for_division(division_id)
    dynamic_prompt = get_dynamic_prompt_template(division_id)
    query_engine = index.as_query_engine(llm=llm, text_qa_template=dynamic_prompt)

    # 3. Eksekusi Query
    answer = ""
    try:
        res = query_engine.query(req.message)
        answer = str(res)
    except Exception as e:
        print(f"Error LLM: {e}")
        answer = "Mohon maaf, terjadi gangguan pada sistem AI kami."
    
    # 4. Deteksi Kegagalan atau Redirect
    ans_lower = answer.lower()
    failure_keywords = [
        "mohon maaf", "tidak dapat dipahami", "informasi spesifik", 
        "belum tersedia", "tidak menemukan jawaban", "saya tidak tahu",
        "silakan beralih", "silakan ajukan"
    ]
    
    is_redirect = "[[redirect:" in ans_lower
    is_failure = any(k in ans_lower for k in failure_keywords)
    
    # LOGIKA: APPEND ALWAYS (Tanpa Cek Duplikat)
    # Jika gagal atau redirect, langsung buat entry baru.
    if is_failure or is_redirect:
        
        new_entry = {
            "id": UNANSWERED_COUNTER,
            "division_id": division_id, 
            "question": req.message,    
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "status": "Redirected" if is_redirect else "Unanswered",
            "hit_count": 1 # Selalu 1, karena setiap kejadian dicatat sebagai baris baru
        }
        
        UNANSWERED.append(new_entry)
        UNANSWERED_COUNTER += 1
        
        # Simpan ke database JSON agar aman saat restart
        save_unanswered_to_db(UNANSWERED)
    
    return ChatResponse(session_id=req.session_id, answer=answer)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)