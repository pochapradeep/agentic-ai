"""Advanced retrieval strategies including hybrid search."""
import os
import subprocess
import sys
from typing import List, Optional

# Fix OpenMP conflict on macOS (must be before any imports that use OpenMP)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Ensure rank_bm25 is available
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("Installing rank-bm25...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rank-bm25"])
    from rank_bm25 import BM25Okapi
    print("✓ rank-bm25 installed successfully")

import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


class HybridRetriever:
    """Hybrid retriever combining BM25 keyword search and semantic vector search."""
    
    def __init__(
        self,
        documents: List[Document],
        vector_store: FAISS,
        embedding_function: Embeddings
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            documents: List of documents with metadata
            vector_store: FAISS vector store for semantic search
            embedding_function: Embedding function
        """
        self.documents = documents
        self.vector_store = vector_store
        self.embedding_function = embedding_function
        
        # Build BM25 index
        print("Building BM25 index for keyword search...")
        tokenized_corpus = [doc.page_content.split(" ") for doc in documents]
        self.doc_ids_list = [doc.metadata.get("id", str(i)) for i, doc in enumerate(documents)]
        self.doc_map = {doc.metadata.get("id", str(i)): doc for i, doc in enumerate(documents)}
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def vector_search(self, query: str, section_filter: Optional[str] = None, k: int = 10) -> List[Document]:
        """Semantic search using FAISS with optional section filtering."""
        if section_filter and "Unknown" not in section_filter:
            # Filter documents by section before searching
            filtered_docs = [
                doc for doc in self.documents 
                if doc.metadata.get("section", "") == section_filter
            ]
            if filtered_docs:
                # Create temporary vector store for filtered docs
                temp_store = FAISS.from_documents(filtered_docs, self.embedding_function)
                return temp_store.similarity_search(query, k=k)
        
        return self.vector_store.similarity_search(query, k=k)
    
    def bm25_search(self, query: str, k: int = 10) -> List[Document]:
        """Keyword-based search using BM25."""
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(bm25_scores)[::-1][:k]
        return [self.doc_map[self.doc_ids_list[i]] for i in top_k_indices]
    
    def hybrid_search(
        self,
        query: str,
        section_filter: Optional[str] = None,
        k: int = 10
    ) -> List[Document]:
        """
        Combines BM25 keyword search and semantic vector search using Reciprocal Rank Fusion.
        
        Args:
            query: Search query
            section_filter: Optional section to filter by
            k: Number of documents to return
            
        Returns:
            List of documents ranked by RRF
        """
        # 1. Keyword Search (BM25)
        bm25_docs = self.bm25_search(query, k=k)
        
        # 2. Semantic Search (with metadata filtering)
        semantic_docs = self.vector_search(query, section_filter=section_filter, k=k)
        
        # 3. Reciprocal Rank Fusion (RRF)
        all_docs = {doc.metadata.get("id", str(i)): doc for i, doc in enumerate(bm25_docs + semantic_docs)}
        ranked_lists = [
            [doc.metadata.get("id", str(i)) for i, doc in enumerate(bm25_docs)],
            [doc.metadata.get("id", str(i)) for i, doc in enumerate(semantic_docs)]
        ]
        
        rrf_scores = {}
        for doc_list in ranked_lists:
            for i, doc_id in enumerate(doc_list):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0
                rrf_scores[doc_id] += 1 / (i + 61)  # RRF rank constant k = 60
        
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        final_docs = [all_docs[doc_id] for doc_id in sorted_doc_ids[:k]]
        return final_docs

