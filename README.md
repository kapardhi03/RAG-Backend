# RAG Microservices API

FastAPI-based RAG system with ChromaDB, OpenAI, PostgreSQL, and AWS S3 for document processing and intelligent querying.

## Prerequisites

- Python 3.10+
- PostgreSQL 13+
- OpenAI API Key
- AWS S3 Bucket

## Setup

### 1. Clone Repository
```bash
git clone https://gitlab.aveosoft.com/ai-product/rag-module.git
cd rag-microservices
```

### 2. Create Virtual Environment

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip setuptools wheel ( Optional )
pip install -r requirements-python3.txt

or

pip install --upgrade pip setuptools wheel && pip install -r requirements-python3.txt

```

### 4. Setup PostgreSQL

**Windows:**
```cmd
# Install PostgreSQL from https://postgresql.org
# Create database
psql -U postgres -c "CREATE DATABASE rag_microservices;"
```

**macOS:**
```bash
brew install postgresql
brew services start postgresql
createdb rag_microservices
```

**Linux:**
```bash
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo -u postgres createdb rag_microservices
```

### 5. Configure Environment

Update `.env` file:
```env
# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=rag_microservices
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/rag_microservices

# OpenAI
OPENAI_API_KEY=sk-your-openai-api-key-here

# AWS S3
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
S3_BUCKET_NAME=your-bucket-name
AWS_REGION=ap-south-1

# JWT
JWT_SECRET_KEY=your_super_secret_key_here
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=4320

# Paths
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

### 6. Initialize Database
```bash
python -c "
import asyncio
from app.db.session import init_db
asyncio.run(init_db())
"
```

### 7. Start Application
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Testing

### API Documentation
Access the interactive Swagger UI for testing and documentation:
- **Swagger UI**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

You can use the Swagger interface to test all endpoints interactively.

### 1. Register User
```bash
curl -X POST "http://localhost:8000/api/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@company.com", "password": "password123", "name": "User"}'
```

### 2. Login
```bash
curl -X POST "http://localhost:8000/api/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@company.com&password=password123"
```
Save the `access_token`.

### 3. Create Knowledge Base
```bash
curl -X POST "http://localhost:8000/api/kb/create" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "Test KB", "description": "Test knowledge base"}'
```
Save the `kb_id`.

### 4. Upload Document
```bash
curl -X POST "http://localhost:8000/api/kb/KB_ID/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@document.pdf"
```

### 5. Add URL
```bash
curl -X POST "http://localhost:8000/api/urls/KB_ID/add-url" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

### 6. Query (Choose One)

**Semantic Search:**
```bash
curl -X POST "http://localhost:8000/api/kb/KB_ID" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the features?", "top_k": 5}'
```

**RAG Response:**
```bash
curl -X POST "http://localhost:8000/api/kb/KB_ID/rag" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the content", "top_k": 5}'
```

**Chat Interface:**
```bash
curl -X POST "http://localhost:8000/api/kb/KB_ID/chat" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "What documents do you have?", "top_k": 5}'
```

## AWS S3 Commands

# S3 Bucket Structure

This outlines the structure of documents stored in the S3 bucket. The hierarchy is organized by tenant and knowledge base (KB), down to individual document files.

```

s3://your-bucket-name/
├── tenant-uuid-1/
│   ├── kb-uuid-1/
│   │   ├── doc-uuid-1/
│   │   │   ├── original.pdf         # Original uploaded file
│   │   │   ├── processed.txt        # Extracted or processed content
│   │   │   └── metadata.json        # Metadata about the document
│   │   └── doc-uuid-2/
│   │       ├── url\_content.html     # Fetched content from URL (if applicable)
│   │       └── processed.txt        # Processed text from HTML
│   └── kb-uuid-2/
│       └── ...
└── tenant-uuid-2/
└── ...

```

## Description of Components

- **tenant-uuid-*/:** Top-level directory for each tenant (client or organization).
- **kb-uuid-*/:** A unique knowledge base under a tenant.
- **doc-uuid-*/:** Each document uploaded or processed under a knowledge base.
  - `original.pdf`: Original document uploaded by the user.
  - `url_content.html`: HTML content fetched from a URL, if applicable.
  - `processed.txt`: Cleaned and extracted text used for downstream processing (e.g., vector embedding).
  - `metadata.json`: Metadata including upload date, source, document type, etc.

> ⚠️ UUIDs are placeholders and should be replaced by actual dynamically generated identifiers in the application.

---



# AWS S3 Operations for RAG Microservice

This guide provides essential AWS S3 commands for managing tenants, knowledge bases, and documents in the CloneifyAI RAG system.

---

## Basic S3 Operations

### List Bucket Contents

```bash
# List all tenant folders in the root of the bucket
aws s3 ls s3://cloneifyai/

# List all knowledge base folders for a specific tenant
aws s3 ls s3://cloneifyai/<tenant-id>/

# List all document folders in a knowledge base
aws s3 ls s3://cloneifyai/<tenant-id>/<kb-id>/

# List all files in a document folder
aws s3 ls s3://cloneifyai/<tenant-id>/<kb-id>/<document-id>/
````

---

## File Operations

```bash
# Download a file from S3 to your local machine
aws s3 cp s3://cloneifyai/<tenant-id>/<kb-id>/<document-id>/original.docx ./downloaded-file.docx

# Upload a local file to S3
aws s3 cp ./local-file.docx s3://cloneifyai/<tenant-id>/<kb-id>/<document-id>/original.docx

# Copy a file within S3
aws s3 cp s3://cloneifyai/source-path/file.docx s3://cloneifyai/destination-path/file.docx

# Delete a file from S3
aws s3 rm s3://cloneifyai/<tenant-id>/<kb-id>/<document-id>/original.docx
```

---

## Check if a File Exists

```bash
# Method 1: Using ls (no output means file not found)
aws s3 ls s3://cloneifyai/<tenant-id>/<kb-id>/<document-id>/original.docx

# Method 2: Using head-object (more reliable)
aws s3api head-object \
  --bucket cloneifyai \
  --key <tenant-id>/<kb-id>/<document-id>/original.docx
```

---

## Batch Operations

```bash
# Sync a local directory to S3
aws s3 sync ./local-dir s3://cloneifyai/<destination-path>

# Sync an S3 directory to a local folder
aws s3 sync s3://cloneifyai/<source-path> ./local-dir

# Copy all files from one folder to another
aws s3 cp s3://cloneifyai/<source-folder>/ s3://cloneifyai/<destination-folder>/ --recursive

# Delete all files in a folder
aws s3 rm s3://cloneifyai/<folder-path>/ --recursive
```

---

## Generate a Pre-signed URL

```bash
# Generate a temporary URL valid for 1 hour (3600 seconds)
aws s3 presign s3://cloneifyai/<tenant-id>/<kb-id>/<document-id>/original.docx --expires-in 3600
```

---

## Advanced Operations

```bash
# List all objects in the bucket (flat list)
aws s3api list-objects-v2 --bucket cloneifyai --max-items 100

# Get metadata about a file
aws s3api get-object-attributes \
  --bucket cloneifyai \
  --key <tenant-id>/<kb-id>/<document-id>/original.docx \
  --object-attributes ETag,Checksum,ObjectSize,StorageClass,ObjectParts

# Change the storage class of an object
aws s3api copy-object \
  --copy-source cloneifyai/<path-to-file.docx> \
  --bucket cloneifyai \
  --key <path-to-file.docx> \
  --storage-class STANDARD_IA
```

---

## Working with File Types

```bash
# Word document
aws s3 ls s3://cloneifyai/<path-to-document>/original.docx

# PDF file
aws s3 ls s3://cloneifyai/<path-to-document>/original.pdf

# HTML file (e.g., from a URL)
aws s3 ls s3://cloneifyai/<path-to-document>/url_content.html

# Plain text file
aws s3 ls s3://cloneifyai/<path-to-document>/original.txt
```

---

## Tenant and Knowledge Base Operations

```bash
# Count all files for a specific tenant
aws s3 ls s3://cloneifyai/<tenant-id>/ --recursive | wc -l

# Find files newer than a specific date (example: 2025-05-12)
aws s3 ls s3://cloneifyai/<tenant-id>/ --recursive | grep "2025-05-12"

# Download all files from a knowledge base
aws s3 cp s3://cloneifyai/<tenant-id>/<kb-id>/ ./local-dir --recursive
```

---

## Important Notes

* Replace placeholders like `<tenant-id>`, `<kb-id>`, and `<document-id>` with actual values.
* Use double quotes for paths containing spaces, e.g., `"path/with spaces/file.docx"`.
* Be cautious with `rm` and `--recursive`; they will permanently delete files.
* Public S3 object URLs follow this format:

  ```
  https://cloneifyai.s3.ap-south-1.amazonaws.com/<path-to-file>
  ```

---

## Error Troubleshooting

| Error           | Cause                                  |
| --------------- | -------------------------------------- |
| 404 Not Found   | The file or path does not exist        |
| 403 Forbidden   | Insufficient permissions               |
| ValidationError | Incorrect syntax or invalid parameters |
---




## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Register user |
| POST | `/api/auth/login` | Login |
| POST | `/api/kb/create` | Create knowledge base |
| GET | `/api/kb/` | List knowledge bases |
| POST | `/api/kb/{kb_id}/upload` | Upload document |
| POST | `/api/urls/{kb_id}/add-url` | Add URL |
| POST | `/api/kb/{kb_id}` | Semantic search |
| POST | `/api/kb/{kb_id}/rag` | RAG query |
| POST | `/api/kb/{kb_id}/chat` | Chat query |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI documentation |
---