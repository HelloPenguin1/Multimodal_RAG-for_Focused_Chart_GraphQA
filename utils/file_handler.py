"""
utils/file_handler.py
Simple file I/O utilities - JSON save/load only.
"""

import json
from pathlib import Path
from typing import Dict, List, Any


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save dictionary as JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save JSON file
        
    Example:
        >>> save_json({"type": "bar", "values": [1, 2, 3]}, "chart.json")
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Saved to {filepath}")


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load JSON file as dictionary.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded dictionary
        
    Example:
        >>> data = load_json("chart.json")
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"✓ Loaded from {filepath}")
    return data


def save_jsonl(records: List[Dict], filepath: str) -> None:
    """
    Save list of dictionaries as JSONL (one per line).
    
    Args:
        records: List of dictionaries
        filepath: Path to save JSONL file
        
    Example:
        >>> save_jsonl([{"id": 1}, {"id": 2}], "data.jsonl")
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    print(f"✓ Saved {len(records)} records to {filepath}")


def load_jsonl(filepath: str) -> List[Dict]:
    """
    Load JSONL file (one JSON per line).
    
    Args:
        filepath: Path to JSONL file
        
    Returns:
        List of dictionaries
        
    Example:
        >>> records = load_jsonl("data.jsonl")
    """
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"✓ Loaded {len(records)} records from {filepath}")
    return records