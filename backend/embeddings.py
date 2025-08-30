"""
Embedding model loader and FAISS index management.
"""

import logging
import os
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from .utils import Config, save_pickle, load_pickle, save_json, load_json

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages embedding model and FAISS index operations."""
    
    def __init__(self, model_name: str = None, config: Config = None):
        self.config = config or Config()
        self.model_name = model_name or self.config.embedding_model
        self.embedding_model = None
        self.langchain_embeddings = None
        self.vector_store = None
        self.document_store = {}
        
        # Initialize embedding model
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load the sentence transformer embedding model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Load SentenceTransformer model
            self.embedding_model = SentenceTransformer(self.model_name)
            
            # Create LangChain compatible embeddings
            self.langchain_embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Get embedding dimension
            test_embedding = self.embedding_model.encode(["test"], show_progress_bar=False)
            self.embedding_dim = test_embedding.shape[1]
            
            logger.info(f"Successfully loaded embedding model with dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def create_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            
        Returns:
            NumPy array of embeddings
        """
        try:
            if not texts:
                return np.array([])
            
            logger.info(f"Creating embeddings for {len(texts)} texts")
            
            # Create embeddings using SentenceTransformer
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def build_faiss_index(self, documents: List[Document]) -> bool:
        """
        Build FAISS index from documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Success status
        """
        try:
            if not documents:
                logger.warning("No documents provided for indexing")
                return False
            
            logger.info(f"Building FAISS index from {len(documents)} documents")
            
            # Extract texts and metadata
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Create vector store using LangChain FAISS wrapper
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.langchain_embeddings,
                metadatas=metadatas
            )
            
            # Store documents for retrieval
            self.document_store = {i: doc for i, doc in enumerate(documents)}
            
            logger.info(f"Successfully built FAISS index with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            return False
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add new documents to existing index.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            Success status
        """
        try:
            if not documents:
                return True
            
            logger.info(f"Adding {len(documents)} documents to existing index")
            
            # Extract texts and metadata
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            if self.vector_store is None:
                # Create new index if none exists
                return self.build_faiss_index(documents)
            else:
                # Add to existing index
                self.vector_store.add_texts(texts=texts, metadatas=metadatas)
                
                # Update document store
                start_idx = len(self.document_store)
                for i, doc in enumerate(documents):
                    self.document_store[start_idx + i] = doc
            
            logger.info(f"Successfully added documents to index")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to index: {e}")
            return False
    
    def save_index(self, index_path: str = None, docstore_path: str = None, metadata_path: str = None) -> bool:
        """
        Save FAISS index and document store to disk.
        
        Args:
            index_path: Path to save FAISS index
            docstore_path: Path to save document store
            metadata_path: Path to save metadata
            
        Returns:
            Success status
        """
        try:
            index_path = index_path or self.config.index_path
            docstore_path = docstore_path or self.config.docstore_path
            metadata_path = metadata_path or self.config.metadata_path
            
            if self.vector_store is None:
                logger.warning("No vector store to save")
                return False
            
            # Ensure directories exist
            for path in [index_path, docstore_path, metadata_path]:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            self.vector_store.save_local(os.path.dirname(index_path))
            
            # Save document store
            save_pickle(self.document_store, docstore_path)
            
            # Save metadata
            metadata = {
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim,
                'num_documents': len(self.document_store),
                'index_path': index_path,
                'docstore_path': docstore_path,
            }
            save_json(metadata, metadata_path)
            
            logger.info(f"Successfully saved index with {len(self.document_store)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return False
    
    def load_index(self, index_path: str = None, docstore_path: str = None, metadata_path: str = None) -> bool:
        """
        Load FAISS index and document store from disk.
        
        Args:
            index_path: Path to FAISS index
            docstore_path: Path to document store
            metadata_path: Path to metadata
            
        Returns:
            Success status
        """
        try:
            index_path = index_path or self.config.index_path
            docstore_path = docstore_path or self.config.docstore_path
            metadata_path = metadata_path or self.config.metadata_path
            
            # Check if all required files exist
            index_dir = os.path.dirname(index_path)
            faiss_index_file = os.path.join(index_dir, "index.faiss")
            faiss_pkl_file = os.path.join(index_dir, "index.pkl")
            
            if not all(os.path.exists(f) for f in [faiss_index_file, faiss_pkl_file, docstore_path]):
                logger.info("Index files not found, will create new index")
                return False
            
            # Load metadata
            metadata = load_json(metadata_path)
            if not metadata:
                logger.warning("Could not load metadata, will create new index")
                return False
            
            # Check model compatibility
            if metadata.get('model_name') != self.model_name:
                logger.warning(f"Model mismatch: expected {self.model_name}, found {metadata.get('model_name')}")
                return False
            
            # Load FAISS index
            self.vector_store = FAISS.load_local(
                index_dir,
                embeddings=self.langchain_embeddings
            )
            
            # Load document store
            self.document_store = load_pickle(docstore_path) or {}
            
            logger.info(f"Successfully loaded index with {len(self.document_store)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        if self.vector_store is None or not self.document_store:
            return {
                'num_documents': 0,
                'model_name': self.model_name,
                'embedding_dim': getattr(self, 'embedding_dim', 0)
            }
        
        # Analyze documents by file
        files = {}
        for doc in self.document_store.values():
            filename = doc.metadata.get('filename', 'unknown')
            if filename not in files:
                files[filename] = {'chunks': 0, 'pages': doc.metadata.get('num_pages', 0)}
            files[filename]['chunks'] += 1
        
        return {
            'num_documents': len(self.document_store),
            'num_files': len(files),
            'model_name': self.model_name,
            'embedding_dim': getattr(self, 'embedding_dim', 0),
            'files': files
        }
    
    def clear_index(self) -> bool:
        """Clear the current index and document store."""
        try:
            self.vector_store = None
            self.document_store = {}
            
            # Remove saved files
            for path in [self.config.index_path, self.config.docstore_path, self.config.metadata_path]:
                if os.path.exists(path):
                    os.remove(path)
            
            # Remove FAISS index directory files
            index_dir = os.path.dirname(self.config.index_path)
            for file in ["index.faiss", "index.pkl"]:
                file_path = os.path.join(index_dir, file)
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            logger.info("Successfully cleared index")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            return False