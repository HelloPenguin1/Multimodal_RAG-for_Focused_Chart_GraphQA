
from typing import List, Dict, Any
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from utils.config import EMBEDDING_MODEL, FAISS_INDEX_PATH


def initialize_embeddings(model_name: str = EMBEDDING_MODEL):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings



def build_vectorstore(extracted_data: List[Dict[str, Any]], embedding_model=None):
    """Build FAISS vectorstore from extracted chart/table data."""
    if embedding_model is None:
        embedding_model = initialize_embeddings()

    documents: List[Document] = []
    for item in extracted_data:
        page_content = f"{item.get('type', '')} {item.get('title', '')} {item.get('context', '')}"
        if not page_content.strip():
            page_content = str(item.get('extracted_text', 'No text available'))

        metadata = {
            "page": item.get("page"),
            "type": item.get("type"),
            "chart_title": item.get("title", ""),
        }

        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

    vectorstore = FAISS.from_documents(documents, embedding_model)
    print(f"✓ Created FAISS index with {len(documents)} documents")
    return vectorstore


def get_retriever_from_data(extracted_data: List[Dict[str, Any]], embedding_model=None, k: int = 3):
    """Build vectorstore and return a LangChain retriever."""
    vectorstore = build_vectorstore(extracted_data, embedding_model=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    print(f"✓ Created retriever with k={k}")
    return retriever


def save_vectorstore(vectorstore, index_path: str = FAISS_INDEX_PATH) -> None:
    """Save FAISS vectorstore to disk."""
    vectorstore.save_local(index_path)
    print(f"✓ Saved FAISS index to {index_path}")


def load_vectorstore(index_path: str = FAISS_INDEX_PATH, embedding_model=None, as_retriever: bool = True, k: int = 3):
    """Load FAISS vectorstore from disk and optionally return retriever."""
    if not os.path.exists(index_path):
        print(f"✗ Index path does not exist: {index_path}")
        return None

    if embedding_model is None:
        embedding_model = initialize_embeddings()

    vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    print(f"✓ Loaded FAISS index from {index_path}")

    if as_retriever:
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        print(f"✓ Created retriever with k={k}")
        return retriever
    
    return vectorstore