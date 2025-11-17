
import os
import fitz  # PyMuPDF
from typing import List, Tuple, Dict


def pdf_to_images(pdf_path: str, output_dir: str = "temp_images", dpi_multiplier: int = 2) -> List[Tuple[str, int]]:
  
  
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open PDF
        pdf_document = fitz.open(pdf_path)
        print(f"Opened PDF: {pdf_path} ({pdf_document.page_count} pages)")
        
        image_paths = []
        
        # Extract each page as image
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            
            # Render page to image with zoom for better OCR
            matrix = fitz.Matrix(dpi_multiplier, dpi_multiplier)
            pix = page.get_pixmap(matrix=matrix)
            
            # Save image
            image_path = os.path.join(output_dir, f"page_{page_num + 1}.png")
            pix.save(image_path)
            
            image_paths.append((image_path, page_num + 1))
            print(f"Extracted page {page_num + 1}")
        
        pdf_document.close()
        print(f"Extracted {len(image_paths)} images from PDF\n")
        
        return image_paths
    
    except Exception as e:
        print(f"Error converting PDF to images: {str(e)}")
        raise


def extract_pdf_metadata(pdf_path: str) -> Dict:
    """
    Extract metadata from PDF (title, author, creation date, etc.).
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary with PDF metadata
        
    Example:
        >>> metadata = extract_pdf_metadata("sample.pdf")
        >>> print(metadata.get("title"))
    """
    try:
        pdf_document = fitz.open(pdf_path)
        metadata = pdf_document.metadata
        pdf_document.close()
        
        print(f"Extracted metadata from {pdf_path}")
        return metadata or {}
    
    except Exception as e:
        print(f"Error extracting PDF metadata: {str(e)}")
        return {}


def get_pdf_page_count(pdf_path: str) -> int:
    """
    Get total number of pages in PDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Number of pages
    """
    try:
        pdf_document = fitz.open(pdf_path)
        page_count = pdf_document.page_count
        pdf_document.close()
        return page_count
    except Exception as e:
        print(f"Error getting page count: {str(e)}")
        return 0


def extract_text_from_pdf(pdf_path: str) -> Dict[int, str]:
    """
    Extract raw text from all PDF pages.
    Useful for fallback or text-only processing.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary mapping page_number to text content
    """
    try:
        pdf_document = fitz.open(pdf_path)
        text_content = {}
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text = page.get_text()
            text_content[page_num + 1] = text
        
        pdf_document.close()
        print(f"Extracted text from {len(text_content)} pages\n")
        return text_content
    
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return {}