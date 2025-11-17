"""
utils/text_processor.py
Text processing using unstructured library - optimized for multimodal RAG.
Parses PDFs into structured elements and chunks by title.

References:
- Unstructured Library: https://unstructured-io.github.io/unstructured/
"""

from typing import List, Dict, Any
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition


def partition_pdf_document(pdf_path: str) -> List[Any]:
    """
    Parse PDF into structured elements (text, tables, images, etc.).
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of document elements with type and metadata
        
    Example:
        >>> elements = partition_pdf_document("sample.pdf")
        >>> print(f"Found {len(elements)} elements")
    """
    elements = partition(pdf_path)
    print(f"Partitioned PDF: {len(elements)} elements")
    return elements


def create_chunks_by_title(elements):
    """
    Chunk elements by title/heading structure.
    Preserves document hierarchy and keeps related content together.
    Perfect for keeping chart captions with chart data.
    
    Args:
        elements: Document elements from partition_pdf_document()
        max_characters: Absolute max characters per chunk
        new_after_n_chars: Soft limit (algo aims for this)
        combine_text_under_n_chars: Merge tiny chunks smaller than this
        
    Returns:
        List of chunked elements preserving structure
        
    Example:
        >>> elements = partition_pdf_document("sample.pdf")
        >>> chunks = create_chunks_by_title(elements)
        >>> print(f"Created {len(chunks)} chunks")
    """
    chunks = chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500,
    )
    
    print(f"Created {len(chunks)} chunks by title")
    return chunks


def chunks_to_dict(chunks):
    """
    Convert chunks to dictionary format for processing and storage.
    Extracts text, type, and metadata.
    
    Args:
        chunks: List of chunk elements
        
    Returns:
        List of dictionaries with text, type, metadata
        
    Example:
        >>> chunks = create_chunks_by_title(elements)
        >>> chunk_dicts = chunks_to_dict(chunks)
        >>> for chunk in chunk_dicts:
        ...     print(chunk["text"][:50])
    """
    chunk_dicts = []
    
    for idx, chunk in enumerate(chunks):
        chunk_dict = {
            "chunk_id": idx,
            "text": chunk.text if hasattr(chunk, 'text') else str(chunk),
            "type": chunk.type if hasattr(chunk, 'type') else "unknown",
        }
        
        # Add metadata if available
        if hasattr(chunk, 'metadata'):
            chunk_dict["metadata"] = {
                "page_number": getattr(chunk.metadata, 'page_number', None),
                "coordinates": getattr(chunk.metadata, 'coordinates', None),
            }
        
        chunk_dicts.append(chunk_dict)
    
    print(f"Converted {len(chunk_dicts)} chunks to dictionary format")
    return chunk_dicts


def filter_by_type(chunks: List[Dict], include_types: List[str]) -> List[Dict]:
    """
    Filter chunks by element type (text, table, image, etc.).
    Useful for focusing on specific content.
    
    Args:
        chunks: List of chunk dictionaries
        include_types: Types to keep (e.g., ["text", "table"])
        
    Returns:
        Filtered chunks
        
    Example:
        >>> filtered = filter_by_type(chunk_dicts, ["text", "table"])
    """
    filtered = [c for c in chunks if c.get("type") in include_types]
    print(f"✓ Filtered to {len(filtered)} chunks (types: {include_types})")
    return filtered


def clean_chunk_text(chunk: Dict) -> Dict:
    chunk["text"] = " ".join(chunk["text"].split())
    return chunk


def clean_all_chunks(chunks: List[Dict]) -> List[Dict]:
    return [clean_chunk_text(chunk) for chunk in chunks]