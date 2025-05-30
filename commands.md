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
