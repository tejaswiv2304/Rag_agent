

import os
from dotenv import load_dotenv

t
load_dotenv()


WEAVIATE_CLUSTER_URL = os.getenv("WEAVIATE_CLUSTER_URL", "xv0g7lpascu7nnpxiqqyq.c0.asia-southeast1.gcp.weaviate.cloud")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


if not WEAVIATE_API_KEY:
    print(" Error: WEAVIATE_API_KEY not found in .env file")
    exit(1)
if not MISTRAL_API_KEY:
    print(" Error: MISTRAL_API_KEY not found in .env file")
    exit(1)

print("API keys loaded from .env file")

from langchain_community.document_loaders import PyPDFLoader


PDF_PATH = "rag.pdf"  

print("Loading PDF documents...")
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()
print(f"Loaded {len(pages)} pages from the PDF")


from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,  
    separators=["\n\n", "\n", " ", ""]
)

docs = text_splitter.split_documents(pages)
print(f"Split into {len(docs)} document chunks")

def sanitize_metadata(metadata: dict) -> dict:
    """Replace dots with underscores in metadata keys for Weaviate compatibility."""
    return {
        k.replace(".", "_"): v
        for k, v in metadata.items()
    }

for doc in docs:
    doc.metadata = sanitize_metadata(doc.metadata)

print("Sanitized document metadata")


from langchain_huggingface import HuggingFaceEmbeddings

print("Loading embedding model...")
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
    encode_kwargs={'normalize_embeddings': True}
)
print(f"Embedding model loaded: {embedding_model_name}")


import weaviate
from weaviate.classes.init import Auth

print("Connecting to Weaviate Cloud...")
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_CLUSTER_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
)


if client.is_ready():
    print(" Successfully connected to Weaviate Cloud!")
else:
    print(" Failed to connect to Weaviate Cloud")
    exit(1)


from langchain_weaviate.vectorstores import WeaviateVectorStore

INDEX_NAME = "DocsIndex"


print(f"Creating vector store with index: {INDEX_NAME}")
vector_db = WeaviateVectorStore(
    client=client,
    index_name=INDEX_NAME,
    text_key="text",
    embedding=embeddings
)


print("Adding documents to vector store (this may take a while)...")
vector_db.add_documents(docs)
print(f" Successfully added {len(docs)} documents to Weaviate!")

print("\n" + "="*50)
print("Testing similarity search...")
query = "What is RAG (Retrieval-Augmented Generation)?"
search_results = vector_db.similarity_search(query, k=3)

print(f"\nQuery: {query}")
print(f"\nTop 3 relevant document chunks:")
for i, result in enumerate(search_results, 1):
    print(f"\n--- Result {i} ---")
    print(result.page_content[:300] + "...")

from langchain_mistralai import ChatMistralAI

print("\n" + "="*50)
print("Initializing Mistral AI...")

llm = ChatMistralAI(
    model="mistral-small",
    mistral_api_key=MISTRAL_API_KEY,
    temperature=0.3,  # Lower for more focused responses
    max_tokens=1024
)

print("Mistral AI initialized!")


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say you don't know.
Keep the answer concise and use a maximum of 5 sentences.

Question: {question}

Context: {context}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)


retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4} 
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(" RAG chain created!")


print("\n" + "="*50)
print("TESTING RAG SYSTEM")
print("="*50)

test_questions = [
    "What is RAG (Retrieval-Augmented Generation)?",
    "What are the two types of RAG models mentioned?",
    "What is the difference between RAG-Token and RAG-Sequence?",
]

for question in test_questions:
    print(f"\n Question: {question}")
    print("-" * 40)
    
    try:
        answer = rag_chain.invoke(question)
        print(f" Answer: {answer}")
    except Exception as e:
        print(f" Error: {str(e)}")
    
    print()

def interactive_qa():
    """Run an interactive Q&A session."""
    print("\n" + "="*50)
    print("INTERACTIVE Q&A MODE")
    print("Type 'quit' or 'exit' to stop")
    print("="*50)
    
    while True:
        question = input("\n Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! ")
            break
        
        if not question:
            print("Please enter a question.")
            continue
        
        try:
            answer = rag_chain.invoke(question)
            print(f"\n Answer: {answer}")
        except Exception as e:
            print(f" Error: {str(e)}")


interactive_qa()


print("\n" + "="*50)
print("Closing Weaviate connection...")
client.close()
print(" Connection closed. Script complete!")
