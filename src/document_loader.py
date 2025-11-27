"""Document loading and processing utilities."""
import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


def load_documents_from_data_folder(data_dir: str) -> List[Document]:
    """
    Load all documents from the data folder at the project root.
    Supports PDF and text files.
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        List of Document objects with metadata
    """
    documents = []
    data_path = Path(data_dir)
    
    # Resolve to absolute path to ensure we're using the correct directory
    data_path = data_path.resolve()
    
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_path}")
    
    print(f"Loading documents from: {data_path}")
    
    # Get all files in the data folder
    pdf_files = list(data_path.glob("*.pdf"))
    txt_files = list(data_path.glob("*.txt"))
    
    print(f"Found {len(pdf_files)} PDF file(s) and {len(txt_files)} text file(s)")
    
    # Load PDF files
    for pdf_file in pdf_files:
        print(f"\nLoading PDF: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            # Add source metadata to each document
            for doc in docs:
                doc.metadata['source'] = str(pdf_file)
                doc.metadata['file_name'] = pdf_file.name
            documents.extend(docs)
            print(f"  ✓ Loaded {len(docs)} pages from {pdf_file.name}")
        except Exception as e:
            print(f"  ✗ Error loading {pdf_file.name}: {e}")
    
    # Load text files
    for txt_file in txt_files:
        print(f"\nLoading text file: {txt_file.name}")
        try:
            loader = TextLoader(str(txt_file), encoding='utf-8')
            docs = loader.load()
            # Add source metadata to each document
            for doc in docs:
                doc.metadata['source'] = str(txt_file)
                doc.metadata['file_name'] = txt_file.name
            documents.extend(docs)
            print(f"  ✓ Loaded text from {txt_file.name}")
        except Exception as e:
            print(f"  ✗ Error loading {txt_file.name}: {e}")
    
    return documents

