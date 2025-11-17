import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HF_TOKEN")

# Configuration
PDF_OUTPUT_DIR = "temp_images"
CHART_DATA_DIR = "chart_data"
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 0
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index.bin"

print("Configuration loaded from .env")