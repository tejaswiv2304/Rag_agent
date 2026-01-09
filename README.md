# RAG System with Weaviate and Mistral AI

A Retrieval-Augmented Generation (RAG) system that lets you chat with your PDF documents using Weaviate vector database and Mistral AI.

## What is RAG?

RAG combines the power of:
- **Retrieval**: Finding relevant information from your documents
- **Generation**: Using AI to generate natural language answers based on that information

This means you can ask questions about your PDFs and get accurate, context-aware answers!

## Files Overview

| File | Purpose |
|------|---------|
| `ingest_pdfs.py` | Extracts text from PDFs and stores them in Weaviate vector database |
| `rag_weaviate_mistral.py` | Main RAG workflow - connects everything and lets you ask questions |

## How It Works

```
PDF Documents → Extract Text → Create Embeddings → Store in Weaviate
                                                          ↓
User Question → Find Similar Chunks → Send to Mistral AI → Answer
```

## Prerequisites

- Python 3.8+
- Weaviate Cloud account ([sign up free](https://console.weaviate.cloud/))
- Mistral AI API key ([get one here](https://console.mistral.ai/))

## Installation

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/ragagent.git
cd ragagent
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux
```

3. Install dependencies:
```bash
pip install weaviate-client langchain langchain-core langchain-huggingface
pip install langchain-weaviate langchain-mistralai langchain-text-splitters
pip install langchain-community pypdf sentence-transformers python-dotenv
```

4. Create a `.env` file with your API keys:
```env
WEAVIATE_CLUSTER_URL=your-cluster-url.weaviate.cloud
WEAVIATE_API_KEY=your-weaviate-api-key
MISTRAL_API_KEY=your-mistral-api-key
```

## Usage

### Step 1: Ingest your PDF documents

```bash
python ingest_pdfs.py --file your_document.pdf
```

Or ingest an entire folder:
```bash
python ingest_pdfs.py --folder ./documents/
```

### Step 2: Start asking questions

```bash
python rag_weaviate_mistral.py
```

This will:
1. Connect to your Weaviate Cloud instance
2. Load the PDF and create embeddings
3. Start an interactive Q&A session

## Example

```
Your question: What is RAG?

Answer: RAG (Retrieval-Augmented Generation) is a technique that combines 
information retrieval with text generation. It first retrieves relevant 
documents from a knowledge base, then uses that context to generate 
accurate and informed responses...
```

## Tech Stack

- **Vector Database**: [Weaviate Cloud](https://weaviate.io/)
- **LLM**: [Mistral AI](https://mistral.ai/)
- **Embeddings**: HuggingFace sentence-transformers
- **Framework**: [LangChain](https://langchain.com/)

## License

MIT License - feel free to use and modify!

---

Made using Weaviate and Mistral AI
