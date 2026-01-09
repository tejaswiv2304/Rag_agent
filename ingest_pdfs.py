"""
PDF Ingestion Script for Weaviate RAG System
=============================================
This script adds PDF files to the "test_collection" in Weaviate.
Supports both single file and folder ingestion.

Usage:
    # Single file:
    python ingest_pdfs.py --file /path/to/document.pdf
    
    # Multiple files in a folder:
    python ingest_pdfs.py --folder /path/to/pdf_folder
    
    # With custom collection name:
    python ingest_pdfs.py --folder /path/to/pdfs --collection my_collection
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.data import DataObject
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# ====================
# CONFIGURATION
# ====================
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_COLLECTION = "test_collection"

# Chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def get_weaviate_client() -> weaviate.WeaviateClient:
    """Create and return a Weaviate client."""
    client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=50051,
        headers={
            "X-OpenAI-Api-Key": OPENAI_API_KEY
        }
    )
    return client


def create_collection_if_not_exists(client: weaviate.WeaviateClient, collection_name: str) -> None:
    """Create the collection if it doesn't exist."""
    try:
        # Check if collection exists
        if client.collections.exists(collection_name):
            print(f"✓ Collection '{collection_name}' already exists.")
            return
        
        # Create collection with OpenAI vectorizer
        client.collections.create(
            name=collection_name,
            vectorizer_config=Configure.Vectorizer.text2vec_openai(
                model="text-embedding-3-small",
                dimensions=1536
            ),
            generative_config=Configure.Generative.openai(
                model="gpt-4o-mini"
            ),
            properties=[
                Property(
                    name="content",
                    data_type=DataType.TEXT,
                    description="The text content of the chunk"
                ),
                Property(
                    name="source_file",
                    data_type=DataType.TEXT,
                    description="Original PDF filename"
                ),
                Property(
                    name="page_number",
                    data_type=DataType.INT,
                    description="Page number in the PDF"
                ),
                Property(
                    name="chunk_index",
                    data_type=DataType.INT,
                    description="Index of the chunk within the document"
                ),
                Property(
                    name="file_hash",
                    data_type=DataType.TEXT,
                    description="Hash of the source file for deduplication"
                ),
                Property(
                    name="ingested_at",
                    data_type=DataType.TEXT,
                    description="Timestamp when the chunk was ingested"
                ),
            ]
        )
        print(f"✓ Created collection '{collection_name}' with OpenAI vectorizer.")
        
    except Exception as e:
        print(f"✗ Error creating collection: {e}")
        raise


def extract_text_from_pdf(pdf_path: str) -> List[dict]:
    """Extract text from PDF with page information."""
    pages_data = []
    
    try:
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages_data.append({
                    "page_number": page_num,
                    "content": text.strip()
                })
    except Exception as e:
        print(f"✗ Error reading PDF '{pdf_path}': {e}")
        return []
    
    return pages_data


def chunk_document(pages_data: List[dict], source_file: str, file_hash: str) -> List[dict]:
    """Split document into chunks while preserving metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    chunk_index = 0
    timestamp = datetime.utcnow().isoformat()
    
    for page_data in pages_data:
        page_chunks = text_splitter.split_text(page_data["content"])
        
        for chunk_text in page_chunks:
            chunks.append({
                "content": chunk_text,
                "source_file": source_file,
                "page_number": page_data["page_number"],
                "chunk_index": chunk_index,
                "file_hash": file_hash,
                "ingested_at": timestamp
            })
            chunk_index += 1
    
    return chunks


def get_file_hash(file_path: str) -> str:
    """Calculate MD5 hash of a file for deduplication."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def check_file_exists_in_collection(client: weaviate.WeaviateClient, collection_name: str, file_hash: str) -> bool:
    """Check if a file with the given hash already exists in the collection."""
    try:
        collection = client.collections.get(collection_name)
        response = collection.query.fetch_objects(
            filters=weaviate.classes.query.Filter.by_property("file_hash").equal(file_hash),
            limit=1
        )
        return len(response.objects) > 0
    except Exception:
        return False


def ingest_pdf(client: weaviate.WeaviateClient, pdf_path: str, collection_name: str, skip_existing: bool = True) -> int:
    """Ingest a single PDF file into Weaviate."""
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        print(f"✗ File not found: {pdf_path}")
        return 0
    
    if not pdf_path.suffix.lower() == ".pdf":
        print(f"✗ Not a PDF file: {pdf_path}")
        return 0
    
    # Calculate file hash
    file_hash = get_file_hash(str(pdf_path))
    
    # Check for duplicates
    if skip_existing and check_file_exists_in_collection(client, collection_name, file_hash):
        print(f"⊘ Skipping (already exists): {pdf_path.name}")
        return 0
    
    # Extract text from PDF
    print(f"📄 Processing: {pdf_path.name}")
    pages_data = extract_text_from_pdf(str(pdf_path))
    
    if not pages_data:
        print(f"✗ No text extracted from: {pdf_path.name}")
        return 0
    
    # Chunk the document
    chunks = chunk_document(pages_data, pdf_path.name, file_hash)
    
    if not chunks:
        print(f"✗ No chunks created from: {pdf_path.name}")
        return 0
    
    # Insert chunks into Weaviate
    collection = client.collections.get(collection_name)
    
    with collection.batch.dynamic() as batch:
        for chunk in chunks:
            batch.add_object(properties=chunk)
    
    print(f"✓ Ingested {len(chunks)} chunks from: {pdf_path.name}")
    return len(chunks)


def ingest_folder(client: weaviate.WeaviateClient, folder_path: str, collection_name: str, skip_existing: bool = True) -> dict:
    """Ingest all PDF files from a folder into Weaviate."""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"✗ Folder not found: {folder_path}")
        return {"files_processed": 0, "total_chunks": 0, "errors": 1}
    
    if not folder.is_dir():
        print(f"✗ Not a directory: {folder_path}")
        return {"files_processed": 0, "total_chunks": 0, "errors": 1}
    
    # Find all PDF files
    pdf_files = list(folder.glob("*.pdf")) + list(folder.glob("*.PDF"))
    
    if not pdf_files:
        print(f"✗ No PDF files found in: {folder_path}")
        return {"files_processed": 0, "total_chunks": 0, "errors": 0}
    
    print(f"\n📁 Found {len(pdf_files)} PDF files in '{folder_path}'")
    print("=" * 50)
    
    stats = {
        "files_processed": 0,
        "total_chunks": 0,
        "skipped": 0,
        "errors": 0
    }
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            chunks_added = ingest_pdf(client, str(pdf_file), collection_name, skip_existing)
            if chunks_added > 0:
                stats["files_processed"] += 1
                stats["total_chunks"] += chunks_added
            elif chunks_added == 0:
                stats["skipped"] += 1
        except Exception as e:
            print(f"✗ Error processing {pdf_file.name}: {e}")
            stats["errors"] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDF files into Weaviate for RAG workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a single PDF file:
  python ingest_pdfs.py --file ./documents/report.pdf
  
  # Ingest all PDFs in a folder:
  python ingest_pdfs.py --folder ./documents/
  
  # Use a custom collection name:
  python ingest_pdfs.py --folder ./docs --collection my_docs
  
  # Force re-ingestion of existing files:
  python ingest_pdfs.py --folder ./docs --no-skip-existing
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--file", "-f",
        type=str,
        help="Path to a single PDF file to ingest"
    )
    input_group.add_argument(
        "--folder", "-d",
        type=str,
        help="Path to a folder containing PDF files to ingest"
    )
    
    # Optional arguments
    parser.add_argument(
        "--collection", "-c",
        type=str,
        default=DEFAULT_COLLECTION,
        help=f"Name of the Weaviate collection (default: {DEFAULT_COLLECTION})"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-ingest files even if they already exist in the collection"
    )
    
    args = parser.parse_args()
    
    # Validate OpenAI API key
    if not OPENAI_API_KEY:
        print("✗ Error: OPENAI_API_KEY not set in environment or .env file")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🚀 Weaviate PDF Ingestion Tool")
    print("=" * 50)
    print(f"Weaviate URL: {WEAVIATE_URL}")
    print(f"Collection: {args.collection}")
    print(f"Skip existing: {not args.no_skip_existing}")
    print("=" * 50 + "\n")
    
    # Connect to Weaviate
    try:
        client = get_weaviate_client()
        print("✓ Connected to Weaviate")
    except Exception as e:
        print(f"✗ Failed to connect to Weaviate: {e}")
        sys.exit(1)
    
    try:
        # Ensure collection exists
        create_collection_if_not_exists(client, args.collection)
        
        # Process input
        if args.file:
            chunks_added = ingest_pdf(
                client, 
                args.file, 
                args.collection, 
                skip_existing=not args.no_skip_existing
            )
            print(f"\n✓ Done! Added {chunks_added} chunks.")
        else:
            stats = ingest_folder(
                client, 
                args.folder, 
                args.collection, 
                skip_existing=not args.no_skip_existing
            )
            print("\n" + "=" * 50)
            print("📊 Ingestion Summary")
            print("=" * 50)
            print(f"Files processed: {stats['files_processed']}")
            print(f"Total chunks added: {stats['total_chunks']}")
            print(f"Files skipped (existing): {stats.get('skipped', 0)}")
            print(f"Errors: {stats['errors']}")
            print("=" * 50)
    
    finally:
        client.close()
        print("\n✓ Disconnected from Weaviate")


if __name__ == "__main__":
    main()
