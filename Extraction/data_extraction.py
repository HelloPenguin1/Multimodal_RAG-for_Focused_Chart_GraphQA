"""
extractor.py
Extract structured data from charts/tables using Chandra OCR and LayoutLMv3.
Reads chart/table images and extracts values, captions, and surrounding text.

"""

from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import json
import os
from chandra import ChandraOCR
from transformers import AutoProcessor, AutoModelForTokenClassification
import torch
import os



# Initialize Chandra OCR
def initialize_chandra() -> Optional[Any]:
    ocr = ChandraOCR()
    return ocr

# Initialize LayoutLMv3
def initialize_layoutlm():
    """
    Initialize LayoutLMv3 processor and model.
    
    Returns:
        Tuple of (processor, model) or (None, None)
    """
    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")
    return processor, model

#Extract structured data from chart/table image using Chandra OCR. 
def extract_chart_data(image_path, ocr) -> Dict[str, Any]:
    """
    Args:
        image_path: Path to chart/table image
        ocr: Initialized Chandra OCR instance
        
    Returns:
        Dictionary with extracted chart data (values, labels, type, title)
        
    Example:
        >>> ocr = initialize_chandra()
        >>> chart_data = extract_chart_data("page_1_chart.png", ocr)
        >>> print(chart_data["type"])  # "bar_chart"
    """
    if ocr is None:
        print("OCR not initialized")
        return {}
    
    try:
        # Run Chandra OCR on image
        result = ocr.extract(image_path)
        
        # Parse result into structured format
        chart_data = {
            "source_image": image_path,
            "type": result.get("chart_type", "unknown"),
            "title": result.get("title", ""),
            "extracted_text": result.get("text", ""),
            "data": result.get("data", {}),
            "values": result.get("values", []),
            "labels": result.get("labels", []),
        }
        
        print(f"Extracted chart data from {image_path}")
        return chart_data
    
    except Exception as e:
        print(f"Error extracting chart data: {str(e)}")
        return {}

#Extract text around/near the chart image using LayoutLMv3.
def extract_contextual_text(image_path: str, processor, model) -> str:
    """
    
    
    Args:
        image_path: Path to image
        processor: LayoutLMv3 processor
        model: LayoutLMv3 model
        
    Returns:
        Extracted contextual text
        
    Example:
        >>> processor, model = initialize_layoutlm()
        >>> text = extract_contextual_text("page_1_chart.png", processor, model)
    """
    if processor is None or model is None:
        print("LayoutLMv3 not initialized")
        return ""
    
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Process with LayoutLMv3
        encoding = processor(image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**encoding)
        
        # Extract text predictions
        predictions = outputs.logits.argmax(-1)
        
        # Convert predictions to readable text
        tokens = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
        predicted_labels = [model.config.id2label.get(pred.item(), "O") for pred in predictions[0]]
        
        # Combine tokens with predicted labels (simplified)
        contextual_text = " ".join([token for token, label in zip(tokens, predicted_labels) if label != "O"])
        
        print(f"Extracted contextual text from {image_path}")
        return contextual_text
    
    except Exception as e:
        print(f"✗ Error extracting contextual text: {str(e)}")
        return ""

#Extract all chart/table data from document images and runs Chandra OCR and LayoutLMv3 on each image.
def extract_from_document(image_paths: List[Tuple[str, int]], output_dir: str = "extracted_data") -> List[Dict]:
    """ 
    Returns:
        List of extracted data dictionaries
        
    Example:
        >>> from utils import pdf_to_images
        >>> images = pdf_to_images("sample.pdf")
        >>> extracted = extract_from_document(images)
        >>> for data in extracted:
        ...     print(f"Page {data['page']}: {data['chart_type']}")
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize models
    ocr = initialize_chandra()
    processor, model = initialize_layoutlm()
    
    extracted_data = []
    
    for image_path, page_num in image_paths:
        print(f"\n--- Processing Page {page_num}: {image_path} ---")
        
        # Extract chart/table data
        chart_data = extract_chart_data(image_path, ocr)
        if not chart_data:
            continue
        
        chart_data["page"] = page_num
        
        # Extract contextual text
        context_text = extract_contextual_text(image_path, processor, model)
        chart_data["context"] = context_text
        
        extracted_data.append(chart_data)
        
        # Save extracted data for this page
        output_path = os.path.join(output_dir, f"page_{page_num}_extraction.json")
        with open(output_path, 'w') as f:
            json.dump(chart_data, f, indent=2)
        print(f"Saved extraction to {output_path}")
    
    print(f"\nCompleted extraction for {len(extracted_data)} pages")
    return extracted_data