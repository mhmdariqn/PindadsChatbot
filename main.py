from fastapi import FastAPI, HTTPException, UploadFile, File, Security, Depends, Header
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from dotenv import load_dotenv
import os
import re
import json
from datetime import datetime
from pypdf import PdfReader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import hashlib
import uuid

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
SEED_ADMIN_PASSWORD = os.getenv("ADMIN_SECRET", "rahasia_admin")

USERS_DB_FILE = "users_db.json"
ACTIVE_SESSIONS = {}  # session_token -> email

def hash_string(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def load_users():
    if os.path.exists(USERS_DB_FILE):
        try:
            with open(USERS_DB_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_users(users):
    try:
        with open(USERS_DB_FILE, "w") as f:
            json.dump(users, f, indent=4)
    except Exception as e:
        print(f"Gagal menyimpan users db: {e}")

# Seeder
def seed_users():
    users = load_users()
    if not users:
        default_user = {
            "email": "admin@pindad.com",
            "password_hash": hash_string(SEED_ADMIN_PASSWORD),
            "security_question": "Apa nama divisi utama PT Pindad?",
            "security_answer_hash": hash_string("HCM")
        }
        users.append(default_user)
        save_users(users)
        print("Database pengguna di-seed secara otomatis.")

seed_users()

# Optional env for max file upload limit in MB (defaults to 10)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_SIZE = MAX_UPLOAD_MB * 1024 * 1024

DB_FILE = "divisions_db.json"
def slugify(text: str) -> str:
    text = text.strip().upper()  
    text = re.sub(r"[^A-Z0-9]+", "_", text) 
    return text.strip("_") or "UNNAMED_DIVISION"

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
            new_entry = {"id": cid, "name": cid.replace("_", " ").upper(), "description": ""}
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
# ========================
# 5. PROMPT TEMPLATE (DIMODIFIKASI)
# ========================
def get_dynamic_prompt_template(current_div_id: str):
    # Ambil divisi lain selain divisi saat ini
    other_divisions = [d for d in DIVISIONS if d["id"] != current_div_id]
    
    # Buat string daftar divisi beserta deskripsinya (KEYWORD DIAMBIL DARI SINI)
    redirect_list_str = ""
    for d in other_divisions:
        # Fallback jika deskripsi kosong
        desc = d.get("description", "").strip()
        if not desc:
            desc = "Tidak ada deskripsi spesifik."
            
        # Format: - [Nama Divisi]: [Deskripsi/Keywords] -> [[REDIRECT:ID]]
        redirect_list_str += f"- Divisi {d['name']} (Lingkup: {desc}) -> [[REDIRECT:{d['id']}]]\n"

    # Jika tidak ada divisi lain
    if not redirect_list_str:
        redirect_list_str = "(Tidak ada divisi lain yang tersedia saat ini)"

    template_str = f"""
Anda adalah asisten virtual profesional untuk PT Pindad di Divisi: {{current_div_id}}.
Tugas Anda adalah menjawab pertanyaan pengguna dengan gaya bahasa natural, ramah, dan langsung pada intinya.

ATURAN KRUSIAL (GAYA BAHASA):
1. DILARANG KERAS menggunakan frasa: "berdasarkan dokumen", "menurut konteks", atau "informasi yang tersedia".
2. Jawaban harus sopan, formal, dan sangat membantu.

LOGIKA PENANGANAN PERTANYAAN (Ikuti Prioritas 1-3):

1. **PRIORITAS UTAMA: JAWABAN LANGSUNG**
   Jika pertanyaan RELEVAN dengan divisi ini ({{current_div_id}}) dan informasinya ada di konteks dokumen:
   - Jawab langsung pertanyaannya secara lengkap.

2. **PRIORITAS KEDUA: SALAH DIVISI (REDIRECT OTOMATIS)**
   Analisa pertanyaan pengguna. Jika topik pertanyaan TIDAK sesuai dengan divisi ini, tetapi COCOK dengan deskripsi divisi lain di bawah, lakukan redirect.
   
   **DAFTAR KOMPETENSI DIVISI LAIN (Gunakan ini sebagai acuan):**
   {redirect_list_str}

   **Instruksi Redirect:**
   - Berikan jawaban singkat: "Mohon maaf, hal tersebut ditangani oleh [Nama Divisi Tujuuan]."
   - AKHIRI jawaban dengan TAG: [[REDIRECT:ID_DIVISI]].
   - DILARANG menampilkan kontak manual jika melakukan redirect.

3. **PRIORITAS KETIGA: DIVISI BENAR TAPI DATA KURANG (MANUAL CONTACT)**
   Jika pertanyaan RELEVAN dengan divisi ini ({{current_div_id}}) tapi jawaban detail TIDAK DITEMUKAN di dokumen:
   - Anda WAJIB memberikan kontak PIC khusus divisi ini (Jika tersedia di database kontak).
   - JANGAN gunakan tag redirect untuk kasus ini.

---
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

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Menghalangi clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = "frame-ancestors 'none'"
        # Mencegah MIME sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        # Memaksa koneksi HTTPS (HSTS)
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        # Sembunyikan/override versi server asli
        response.headers["Server"] = "Pindad-Chatbot-Gateway"
        return response

app.add_middleware(SecurityHeadersMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False, # Wildcard * + credentials = security risk
)

# Models
class ChatRequest(BaseModel):
    session_id: str = Field(..., max_length=128)
    division: str = Field(..., max_length=64)
    message: str = Field(..., max_length=4096)

class ChatResponse(BaseModel):
    session_id: str
    answer: str

class NewDivision(BaseModel):
    name: str = Field(..., max_length=100)
    description: str = Field("", max_length=500)

class UpdateDivisionDescription(BaseModel):
    description: str = Field(..., max_length=500)

class NewFaq(BaseModel):
    division_id: str = Field(..., max_length=64)
    question: str = Field(..., max_length=1000)
    answer: str = Field(..., max_length=5000)

class LoginRequest(BaseModel):
    email: str = Field(..., max_length=128)
    password: str = Field(..., max_length=128)

class ForgotPasswordQuestionRequest(BaseModel):
    email: str = Field(..., max_length=128)

class ResetPasswordRequest(BaseModel):
    email: str = Field(..., max_length=128)
    security_answer: str = Field(..., max_length=256)
    new_password: str = Field(..., max_length=128)

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
def get_admin_token(x_admin_token: str = Header(None)):
    if not x_admin_token or x_admin_token not in ACTIVE_SESSIONS:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return ACTIVE_SESSIONS[x_admin_token]

# ========================
# 7. ENDPOINTS
# ========================

@app.post("/api/auth/login")
async def auth_login(req: LoginRequest):
    users = load_users()
    pwd_hash = hash_string(req.password)
    user = next((u for u in users if u["email"].lower() == req.email.lower() and u["password_hash"] == pwd_hash), None)
    if not user:
        raise HTTPException(status_code=401, detail="Email atau password salah")
    
    token = str(uuid.uuid4())
    ACTIVE_SESSIONS[token] = req.email.lower()
    return {"token": token}

@app.post("/api/auth/forgot-password/question")
async def auth_forgot_password_question(req: ForgotPasswordQuestionRequest):
    users = load_users()
    user = next((u for u in users if u["email"].lower() == req.email.lower()), None)
    if not user:
        raise HTTPException(status_code=404, detail="Email tidak terdaftar")
    return {"security_question": user["security_question"]}

@app.post("/api/auth/forgot-password/reset")
async def auth_forgot_password_reset(req: ResetPasswordRequest):
    users = load_users()
    user = next((u for u in users if u["email"].lower() == req.email.lower()), None)
    if not user:
        raise HTTPException(status_code=404, detail="Email tidak terdaftar")
    
    ans_hash = hash_string(req.security_answer.strip())
    if user["security_answer_hash"] != ans_hash:
        raise HTTPException(status_code=400, detail="Jawaban pertanyaan keamanan salah")
    
    user["password_hash"] = hash_string(req.new_password)
    save_users(users)
    return {"detail": "Password berhasil direset"}

@app.get("/divisions")
async def list_divisions():
    return DIVISIONS

@app.post("/division", dependencies=[Depends(get_admin_token)])
async def create_division(data: NewDivision):
    global DIVISIONS
    
    div_id = slugify(data.name)

    if any(d["id"] == div_id for d in DIVISIONS):
        raise HTTPException(status_code=400, detail=f"Divisi '{data.name}' sudah ada")

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

@app.put("/division/{div_id}", dependencies=[Depends(get_admin_token)])
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

@app.delete("/division/{div_id}", dependencies=[Depends(get_admin_token)])
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

@app.post("/upload/{division_id}", dependencies=[Depends(get_admin_token)])
async def upload_file(division_id: str, file: UploadFile = File(...)):
    global DOCUMENTS
    safe_filename = os.path.basename(file.filename)
    if not safe_filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Hanya file .pdf yang didukung")

    if not os.path.exists("data"):
        os.makedirs("data")

    file_path = os.path.join("data", safe_filename)
    content_bytes = await file.read()

    if len(content_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail=f"Ukuran file melebihi batas maksimal {MAX_UPLOAD_MB} MB")

    with open(file_path, "wb") as f:
        f.write(content_bytes)

    try:
        coll = client.get_collection(division_id)
        coll.delete(where={"division": division_id})
    except Exception:
        pass
        
    #2. PROSES EKSTRAKSI PDF
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

    doc = Document(text=text_content, metadata={"filename": safe_filename, "division": division_id})
    index = get_index_for_division(division_id)
    index.insert(doc)

    DOCUMENTS.append({
        "filename": safe_filename,
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

@app.post("/faq", dependencies=[Depends(get_admin_token)])
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
    uvicorn.run(app, host="0.0.0.0", port=8000, server_header=False)
