from fastapi import FastAPI, HTTPException, UploadFile, File, Security, Depends, Header
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv
import os
import re
from datetime import datetime
from pypdf import PdfReader
from io import BytesIO
from fastapi import FastAPI, HTTPException, UploadFile, File, Security, Depends, Header

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.mistralai import MistralAI
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext, Document, SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext, Document, SimpleDirectoryReader, PromptTemplate
# ========================
# ENV + CLIENT SETUP
# ========================
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "rahasia_admin")

if not MISTRAL_API_KEY or not CHROMA_API_KEY:
  raise ValueError("MISTRAL_API_KEY dan CHROMA_API_KEY wajib diisi di .env")

llm = MistralAI(api_key=MISTRAL_API_KEY, model="mistral-large-latest")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE,
)

# ========================
# CUSTOM PROMPT (LOGIKA REDIRECT LEBIH KETAT)
# ========================
# ========================
# CUSTOM PROMPT (LOGIKA REDIRECT & GAYA BAHASA)
# ========================
# ========================
# CUSTOM PROMPT (LOGIKA REDIRECT & KONTAK PIC)
# ========================
qa_template_str = """
Anda adalah asisten virtual profesional untuk PT Pindad. 
Tugas Anda adalah menjawab pertanyaan pengguna dengan gaya bahasa natural, ramah, dan langsung pada intinya.

ATURAN KRUSIAL (GAYA BAHASA & KONTEN):
1. DILARANG KERAS menggunakan frasa: "berdasarkan dokumen", "menurut konteks", "informasi yang tersedia", atau sejenisnya. Jawablah seolah-olah Anda memiliki pengetahuan tersebut sendiri.
2. Jawaban harus sopan, formal, dan membantu.

LOGIKA PENANGANAN PERTANYAAN (Ikuti Prioritas 1-3):

1. **PRIORITAS UTAMA: JAWABAN LANGSUNG**
   Jika pertanyaan relevan dengan divisi saat ini dan informasinya ada di konteks:
   - Jawab langsung pertanyaannya secara lengkap.

2. **PRIORITAS KEDUA: SALAH DIVISI (REDIRECT)**
   Jika pertanyaan menyangkut wewenang divisi lain:
   - Berikan jawaban singkat bahwa hal tersebut ditangani divisi terkait.
   - DILARANG MENAMPILKAN Nama PIC, Nomor HP, atau Email milik divisi tujuan redirect.
   - AKHIRI jawaban dengan TAG REDIRECT PENDEK.

   Daftar Mapping Tag:
   - Alat Berat, Bengkel, MRO, Penjualan Produk (Sales), Layanan purna jual, produk, senjata, munisi, garansi, service, antar jemput, pembelian produk -> [[REDIRECT:MRO]]
   - Rekrutmen, Karir, Magang, HRD -> [[REDIRECT:HCM]]
   - Vendor, Tender, Pengadaan, Rantai Pasok -> [[REDIRECT:SCM]]
   - Mutu, Kualitas, K3LH, ISO -> [[REDIRECT:K3LH]]
   - CSR, Bantuan Proposal, UMKM, Lingkungan -> [[REDIRECT:TJSL]]

3. **PRIORITAS KETIGA: DIVISI BENAR TAPI DATA KURANG (MANUAL CONTACT)**
   Jika pertanyaan relevan dengan divisi ini tapi jawaban detail tidak ditemukan di tabel FAQ:
   - Anda WAJIB memberikan Nama PIC, Email, atau Nomor Kontak KHUSUS untuk divisi ini yang tertera di dokumen.
   - Gunakan data kontak berikut jika sesuai dengan Divisi Room saat ini:
     * HCM/Rekrutmen: Vania Avviantari (ecareer@pindad.com / 0851-1720-5177)
     * SCM/Rantai Pasok: Juliandre Caesar Evanda (andre@pindad.com / 0813-1223-8553)
     * TJSL/CSR: Dwi Sumeitri (dsumeitri@pindad.com / Ext 2243)
     * MRO/Sales: Email defense@pindad.com atau sales@pindad.com
   - JANGAN gunakan tag redirect.

---
CONTOH INTERAKSI YANG DIHARAPKAN:

Kasus A (Salah Kamar - Redirect):
User (di Room HCM): "Bagaimana cara servis traktor dan beli senjata?"
Bot: "Mohon maaf, layanan perbaikan alat berat dan pembelian produk pertahanan ditangani oleh Divisi Pemasaran & MRO. Silakan beralih ke room divisi terkait melalui tombol di bawah. [[REDIRECT:MRO]]"
(Perhatikan: Bot TIDAK menyebutkan email sales@pindad.com di sini, biarkan tombol yang bekerja).

Kasus B (Kamar Benar - Info Kurang - Kasih Kontak):
User (di Room HCM): "Apakah ada lowongan spesifik untuk lulusan Sastra Jepang?" (Tidak ada info spesifik di dokumen).
Bot: "Saat ini informasi spesifik mengenai formasi tersebut belum tersedia. Anda dapat menanyakan ketersediaan formasi secara langsung kepada PIC Rekrutmen, Ibu Vania Avviantari melalui email ecareer@pindad.com atau WhatsApp 0851-1720-5177."
(Perhatikan: Bot memberikan kontak karena topiknya benar di HCM).

Kasus C (Jawaban Tersedia):
User: "Apa syarat magang?"
Bot: "Syarat administrasi magang meliputi Surat Pengantar dari sekolah/kampus, CV, Transkrip Nilai, Pas Foto, dan SKCK."
(Perhatikan: Langsung jawab tanpa embel-embel 'berdasarkan dokumen').
---

Context information is below.
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer:
"""

QA_PROMPT = PromptTemplate(qa_template_str)

# ========================
# DATA IN-MEMORY (DEMO)
# ========================

def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "divisi"

# ambil daftar collection awal dari Chroma sebagai divisi awal
_divs = client.list_collections()
DIVISIONS: List[Dict] = [{"id": c.name, "name": c.name} for c in _divs] or [
    {"id": "MRO", "name": "Maintenance Repair & Overhaul", "description": ""},
    {"id": "TJSL", "name": "Tanggung Jawab Sosial & Lingkungan", "description": ""},
    {"id": "HCM", "name": "Human Capital Management", "description": ""},
    {"id": "SCM", "name": "Supply Chain / Rantai Pasok", "description": ""},
    {"id": "K3LH", "name": "Penjaminan Mutu / Quality Assurance", "description": ""},
]

# FAQ & unanswered & stats disimpan in memory (cukup untuk demo)
FAQS: List[Dict] = []       # {id, division_id, question, answer}
FAQ_COUNTER = 1

UNANSWERED: List[Dict] = [] # {id, division_id, question, created_at}
UNANSWERED_COUNTER = 1

MONTHLY_HITS = {
    "Dec": 0, "Jan": 0, "Feb": 0, "Mar": 0
}

DOCUMENTS: List[Dict] = []  # {filename, division_id, uploaded_at}


# ========================
# FASTAPI APP
# ========================
app = FastAPI(title="PINDAD Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# ========================
# MODELS
# ========================
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


# ========================
# HELPER: DAPATKAN INDEX DARI COLLECTION
# ========================
def get_index_for_division(division_id: str) -> VectorStoreIndex:
    try:
        coll = client.get_collection(division_id)
    except Exception:
        # kalau belum ada, buat collection baru
        coll = client.create_collection(division_id)

    vector_store = ChromaVectorStore(chroma_collection=coll)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    return index


# ========================
# HELPER: ADMIN AUTH
# ========================
def get_admin_token(x_admin_secret: str = Header(None)):
    if x_admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True



# ========================
# ENDPOINT: DIVISIONS (untuk GreetingPage & Dashboard)
# ========================
@app.get("/divisions")
async def list_divisions():
    return DIVISIONS


@app.post("/division")
async def create_division(data: NewDivision):
    global DIVISIONS
    div_id = slugify(data.name)

    # cek duplikat
    if any(d["id"] == div_id for d in DIVISIONS):
        raise HTTPException(status_code=400, detail="Divisi sudah ada")

    # buat collection di Chroma
    client.create_collection(div_id)

    DIVISIONS.append({"id": div_id, "name": data.name, "description": data.description})
    return {"id": div_id, "name": data.name, "description": data.description}

# ========================
# ENDPOINT: UPLOAD FILE 
# ========================
@app.put("/division/{div_id}")
async def update_division_description(div_id: str, data: UpdateDivisionDescription):
    global DIVISIONS
    
    # cari divisi by id
    div_found = None
    for d in DIVISIONS:
        if d["id"] == div_id:
            div_found = d
            break
            
    if not div_found:
        raise HTTPException(status_code=404, detail="Divisi tidak ditemukan")
        
    div_found["description"] = data.description
    return div_found


@app.delete("/division/{div_id}")
async def delete_division(div_id: str):
    global DIVISIONS, FAQS

    # hapus dari Chroma
    try:
        client.delete_collection(div_id)
    except Exception:
        # kalau collection tidak ada, abaikan saja
        pass

    # hapus dari list divisions
    DIVISIONS = [d for d in DIVISIONS if d["id"] != div_id]

    # hapus FAQ yang terkait
    FAQS = [f for f in FAQS if f["division_id"] != div_id]

    return {"detail": "Divisi dihapus"}


@app.post("/upload/{division_id}")
async def upload_file(division_id: str, file: UploadFile = File(...)):
    """
    Menerima file, MENGHAPUS isi collection lama (jika ada), 
    lalu mengupload dan mengindex file baru.
    """
    global DOCUMENTS # Kita perlu akses global variable untuk update list dokumen di memori

    # 1. Validasi Ekstensi
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Demo ini hanya mendukung file .pdf"
        )

    # 2. Baca File
    content_bytes = await file.read()
    
    # 3. Simpan file fisik (Opsional: overwrite file lama jika nama sama)
    if not os.path.exists("data"):
        os.makedirs("data")
        
    file_path = os.path.join("data", file.filename)
    with open(file_path, "wb") as f:
        f.write(content_bytes)
        
    # 4. Decode text
    try:
        text = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = content_bytes.decode("latin-1")

    # ====================================================
    # LOGIKA BARU: HAPUS ISI LAMA (RESET COLLECTION)
    # ====================================================
    try:
        # Ambil collection berdasarkan division_id
        coll = client.get_collection(division_id)
        
        # Ambil semua data (kita butuh ID-nya untuk menghapus)
        existing_data = coll.get()
        existing_ids = existing_data.get('ids', [])

        if existing_ids:
            # Jika ada isinya, hapus berdasarkan ID
            print(f"Menghapus {len(existing_ids)} dokumen lama dari divisi {division_id}...")
            coll.delete(ids=existing_ids)
            
            # PENTING: Hapus juga metadata file lama dari variable global DOCUMENTS
            # supaya data di dashboard 'Stats' sinkron dengan isi Chroma
            DOCUMENTS = [doc for doc in DOCUMENTS if doc["division_id"] != division_id]

    except Exception as e:
        # Jika collection belum ada (error), tidak masalah, lanjut buat baru di bawah
        print(f"Collection belum ada atau error saat clear data: {e}")
        pass

    # ====================================================
    # LANJUTKAN PROSES UPLOAD SEPERTI BIASA
    # ====================================================

    # Buat Document LlamaIndex
    doc = Document(text=text, metadata={"filename": file.filename, "division": division_id})
    
    # Dapatkan index (ini akan memakai collection yang sudah dikosongkan tadi)
    index = get_index_for_division(division_id)
    
    # Masukkan data baru
    index.insert(doc)

    # Update Global Documents List dengan file baru
    DOCUMENTS.append({
        "filename": file.filename,
        "division_id": division_id,
        "uploaded_at": datetime.now().isoformat()
    })

    return {"detail": f"Data lama dihapus. File {file.filename} berhasil diupload dan diindex."}


# ========================
# ENDPOINT: DOWNLOAD PDF (ADMIN ONLY)
# ========================
@app.get("/admin/download/pdf/{filename}", dependencies=[Depends(get_admin_token)])
async def download_pdf_admin(filename: str):
    # Sanitize filename to prevent directory traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
        
    file_path = os.path.join("data", filename)
    
    if not os.path.exists(file_path):
        # DEBUG: Show what path was checked
        raise HTTPException(status_code=404, detail=f"File not found: {file_path} (cwd: {os.getcwd()})")
        
    return FileResponse(file_path, media_type="application/pdf", filename=filename)



# ========================
# ENDPOINT: FAQ (untuk Dashboard)
# ========================
@app.get("/faqs")
async def get_faqs():
    # Gabungkan FAQ manual + Document (formatted as FAQ)
    doc_faqs = []
    for idx, doc in enumerate(DOCUMENTS):
        doc_faqs.append({
            "id": f"doc_{idx}",
            "division_id": doc["division_id"],
            "question": f"File: {doc['filename']}",
            "answer": f"Document uploaded on {doc['uploaded_at']}"
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

    # masukkan FAQ ke Chroma juga sebagai knowledge
    text = f"Pertanyaan: {data.question}\nJawaban: {data.answer}"
    doc = Document(text=text, metadata={"division": data.division_id, "type": "faq"})
    index = get_index_for_division(data.division_id)
    index.insert(doc)

    return faq


# ========================
# ENDPOINT: STATS & UNANSWERED (untuk Dashboard)
# ========================
@app.get("/stats")
async def get_stats():
    # Convert MONTHLY_HITS dict back to list format expected by UI
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


# ========================
# ENDPOINT: CHATBOT (untuk ChatPage)
# ========================
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    global UNANSWERED, UNANSWERED_COUNTER, MONTHLY_HITS

    current_month = datetime.now().strftime("%b")
    
    # Cek apakah bulan tersebut ada di dictionary kita
    if current_month in MONTHLY_HITS:
        MONTHLY_HITS[current_month] += 1
    else:
        MONTHLY_HITS[current_month] = 1

    division_id = req.division
    index = get_index_for_division(division_id)
    
    query_engine = index.as_query_engine(
        llm=llm,
        text_qa_template=QA_PROMPT 
    )

    answer = ""
    try:
        res = query_engine.query(req.message)
        answer = str(res)
    except Exception as e:
        print(f"Error: {e}")
        answer = "Terjadi kesalahan pada sistem."
   
    ans_lower = answer.lower()
    
    failure_keywords = [
        "mohon maaf", 
        "tidak dapat dipahami", 
        "informasi spesifik", 
        "belum tersedia",
        "tidak menemukan jawaban",
        "saya tidak tahu",
        "silakan beralih", 
        "silakan ajukan"
    ]
    
    is_redirect = "[[redirect:" in ans_lower
    
    if any(k in ans_lower for k in failure_keywords) or is_redirect:
        
        is_duplicate = any(u["question"] == req.message for u in UNANSWERED)
        
        if not is_duplicate:
            UNANSWERED.append({
                "id": UNANSWERED_COUNTER,
                "division_id": division_id, 
                "question": req.message,    
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M") 
            })
            UNANSWERED_COUNTER += 1
    
    return ChatResponse(session_id=req.session_id, answer=answer)
