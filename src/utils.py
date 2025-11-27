"""Utility functions for the RAG system."""
import re
import uuid
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_documents_with_metadata(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Document]:
    """
    Process documents and add section metadata.
    
    Args:
        documents: List of Document objects
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of Document objects with metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Combine all document content
    raw_text = "\n\n".join([doc.page_content for doc in documents])
    
    # Simple section detection
    lines = raw_text.split('\n')
    potential_sections = []
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        # Check if line looks like a section header
        is_potential_header = (
            len(line_stripped) < 100 and
            (line_stripped[0].isupper() or line_stripped[0].isdigit()) and
            (line_stripped.isupper() or any(c.isupper() for c in line_stripped if c.isalpha()))
        )
        
        if is_potential_header:
            # Check if next few lines have content
            next_content = ""
            for j in range(i+1, min(i+5, len(lines))):
                if lines[j].strip() and (not lines[j].strip()[0].isupper() or len(lines[j].strip()) > 50):
                    next_content = lines[j].strip()
                    break
            
            if next_content or i == 0:
                potential_sections.append((i, line_stripped))
    
    # Create sections
    sections_data = []
    if potential_sections:
        for idx, (line_num, header) in enumerate(potential_sections):
            next_line_num = potential_sections[idx + 1][0] if idx + 1 < len(potential_sections) else len(lines)
            content = '\n'.join(lines[line_num+1:next_line_num]).strip()
            if content or idx == 0:
                sections_data.append((header, content))
    else:
        # Fallback: Split by paragraphs
        paragraphs = raw_text.split('\n\n')
        current_section_title = "Document Content"
        current_content = []
        
        for para in paragraphs:
            para_stripped = para.strip()
            if not para_stripped:
                continue
            
            if len(para_stripped) < 100 and para_stripped[0].isupper():
                if current_content:
                    sections_data.append((current_section_title, '\n\n'.join(current_content)))
                current_section_title = para_stripped
                current_content = []
            else:
                current_content.append(para_stripped)
        
        if current_content:
            sections_data.append((current_section_title, '\n\n'.join(current_content)))
    
    if not sections_data:
        sections_data = [("Full Document", raw_text)]
    
    # Create chunks with metadata
    doc_chunks_with_metadata = []
    source_file = documents[0].metadata.get('file_name', 'unknown') if documents else 'unknown'
    
    for section_title, content in sections_data:
        if not content.strip():
            continue
        
        clean_section_title = section_title.replace('\n', ' ').strip()[:200]
        section_chunks = text_splitter.split_text(content)
        
        if not section_chunks:
            doc_chunks_with_metadata.append(
                Document(
                    page_content=content.strip() or clean_section_title,
                    metadata={
                        "section": clean_section_title,
                        "source_doc": source_file,
                        "file_name": source_file,
                        "id": str(uuid.uuid4())
                    }
                )
            )
        else:
            for chunk in section_chunks:
                chunk_id = str(uuid.uuid4())
                doc_chunks_with_metadata.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "section": clean_section_title,
                            "source_doc": source_file,
                            "file_name": source_file,
                            "id": chunk_id
                        }
                    )
                )
    
    return doc_chunks_with_metadata

