
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.schema import Document
from utils.config import EMBEDDING_MODEL, FAISS_INDEX_PATH


def initialize_embeddings(model_name: str = EMBEDDING_MODEL):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print(f"Loaded embeddings: {model_name}")
    return embeddings


def create_retriever(extracted_data: List[Dict], embedding_model=None):
    """
    Create FAISS retriever from extracted chart/table data.
    LangChain handles index creation automatically.
    
    Args:
        extracted_data: List of extracted data dicts from extractor.py
        embedding_model: HuggingFaceEmbeddings instance (create if None)
        
    Returns:
        FAISS vectorstore ready to search
        
    Example:
        >>> extracted = extract_from_document(images)
        >>> vectorstore = create_retriever(extracted)
        >>> results = vectorstore.similarity_search("What was Q2 revenue?", k=3)
    """
    if embedding_model is None:
        embedding_model = initialize_embeddings()
    
    documents = []
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


def search_retriever(vectorstore, query: str, k: int = 3) -> List[Dict]:
    """
    Search retriever for similar documents.
    
    Args:
        vectorstore: FAISS vectorstore
        query: Search query
        k: Number of results
        
    Returns:
        List of similar documents with metadata
        
    Example:
        >>> results = search_retriever(vectorstore, "revenue", k=3)
        >>> for doc in results:
        ...     print(doc.page_content)
    """
    docs = vectorstore.similarity_search(query, k=k)
    print(f"✓ Retrieved {len(docs)} results")
    return docs


def save_retriever(vectorstore, index_path: str = FAISS_INDEX_PATH) -> None:
    vectorstore.save_local(index_path)
    print(f"Saved FAISS index to {index_path}")


def load_retriever(index_path: str = FAISS_INDEX_PATH, embedding_model=None):
    """
    Load FAISS index from disk.
    
    Args:
        index_path: Path to FAISS index
        embedding_model: HuggingFaceEmbeddings instance (create if None)
        
    Returns:
        Loaded FAISS vectorstore or None if index not found
        
    Example:
        >>> vectorstore = load_retriever()
        >>> results = vectorstore.similarity_search("query")
    """
    import os
    
    if not os.path.exists(index_path):
        print(f"✗ Index path does not exist: {index_path}")
        return None
    
    if embedding_model is None:
        embedding_model = initialize_embeddings()
    
    vectorstore = FAISS.load_local(
        index_path, 
        embedding_model,
        allow_dangerous_deserialization=True
    )
    print(f"✓ Loaded FAISS index from {index_path}")
    return vectorstore