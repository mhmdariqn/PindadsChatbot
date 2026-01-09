# PindadsChatbot - Installation & Deployment Guide

This guide covers the steps to deploy the PindadsChatbot server from scratch. Ideally, run this on a fresh Ubuntu 22.04 LTS server or a Windows environment.

## 1. Prerequisites (OS Preparation)

### Linux (Ubuntu)
Update the package list and install Python 3 and pip.
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip python3-venv git -y
```

### Windows
1.  Download and install **Python 3.10+** from [python.org](https://www.python.org/downloads/).
2.  During installation, check **"Add Python to PATH"**.
3.  Install Git for Windows from [git-scm.com](https://git-scm.com/download/win).

---

## 2. Clone the Repository

Clone the project to your server or local machine.

```bash
git clone https://github.com/mhmdariqn/PindadsChatbot.git
cd PindadsChatbot
```

---

## 3. Environment Setup

It is recommended to use a virtual environment to manage dependencies.

### Create Virtual Environment
**Linux/Mac:**
```bash
python3 -m venv env
source env/bin/activate
```

**Windows:**
```bash
python -m venv env
.\env\Scripts\activate
```

### Install Library Dependencies
Install all required Python packages using `pip`.

```bash
pip install -r requirements.txt
```

---

## 4. Configuration

Create a `.env` file in the root directory. You can copy the example below.

```bash
touch .env  # On Linux
# Or create manually on Windows
```

**Required `.env` Variables:**
```properties
# Mistral AI API Key (Get from console.mistral.ai)
MISTRAL_API_KEY=your_mistral_api_key

# ChromaDB Remote Settings
CHROMA_API_KEY=your_chroma_api_key
CHROMA_TENANT=your_chroma_tenant
CHROMA_DATABASE=your_chroma_database

# Admin Security for PDF Downloads
ADMIN_SECRET=rahasia_admin
```

> [!IMPORTANT]
> Change `ADMIN_SECRET` to a strong, unique password in production.

---

## 5. Running the Application

Start the FastAPI server using Uvicorn.

**Development Mode (Auto-reload):**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Production Mode:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000`.
- API Docs: `http://localhost:8000/docs`
- Redoc: `http://localhost:8000/redoc`

---

## 6. Basic Troubleshooting

### 1. "ModuleNotFoundError"
Ensure you successfully activated the virtual environment (`source env/bin/activate` or `.\env\Scripts\activate`) before running `uvicorn`. Re-run `pip install -r requirements.txt` if needed.

### 2. "500 Internal Server Error" during Chat
Check your `.env` file. Ensure `MISTRAL_API_KEY` and `CHROMA_API_KEY` are correct and have sufficient quota.

### 3. File Download 404
Check if the file physically exists in the `data/` directory on the server. Old uploads (before file persistence fix) might be indexed in the database but missing from the disk. Re-upload the file via the API.

---

## Folder Structure
- `main.py`: Main application entry point.
- `data/`: Directory where uploaded PDF files are stored.
- `env/`: Virtual environment directory (excluded from git).
- `requirements.txt`: Python package dependencies.
