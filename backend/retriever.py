"""
Similarity search and document retrieval functionality.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import re

from langchain.schema import Document

from .embeddings import EmbeddingManager
from .utils import Config

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """Handles document retrieval and similarity search."""
    
    def __init__(self, embedding_manager: EmbeddingManager, config: Config = None):
        self.embedding_manager = embedding_manager
        self.config = config or Config()
    
    def similarity_search(self, query: str, k: int = None, score_threshold: float = 0.0) -> List[Tuple[Document, float]]:
        """
        Perform similarity search on indexed documents.
        
        Args:
            query: Search query string
            k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of tuples (Document, similarity_score)
        """
        try:
            if not self.embedding_manager.vector_store:
                logger.warning("No vector store available for search")
                return []
            
            k = k or self.config.top_k
            
            logger.info(f"Performing similarity search for query: '{query[:100]}...' (k={k})")
            
            # Perform similarity search with scores
            results = self.embedding_manager.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            # Filter by score threshold
            filtered_results = [
                (doc, score) for doc, score in results
                if score >= score_threshold
            ]
            
            logger.info(f"Retrieved {len(filtered_results)} documents above threshold {score_threshold}")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            return []
    
    def hybrid_search(self, query: str, k: int = None, 
                     file_filter: Optional[List[str]] = None,
                     page_filter: Optional[List[int]] = None) -> List[Tuple[Document, float]]:
        """
        Perform hybrid search with additional filtering.
        
        Args:
            query: Search query string
            k: Number of top results to return
            file_filter: List of filenames to filter by
            page_filter: List of page numbers to filter by
            
        Returns:
            List of tuples (Document, similarity_score)
        """
        try:
            # First perform similarity search
            results = self.similarity_search(query, k=k*2)  # Get more results for filtering
            
            # Apply filters
            filtered_results = []
            
            for doc, score in results:
                # File filter
                if file_filter:
                    doc_filename = doc.metadata.get('filename', '')
                    if not any(filename in doc_filename for filename in file_filter):
                        continue
                
                # Page filter
                if page_filter:
                    doc_page = doc.metadata.get('chunk_page', 1)
                    if doc_page not in page_filter:
                        continue
                
                filtered_results.append((doc, score))
                
                # Stop when we have enough results
                if len(filtered_results) >= (k or self.config.top_k):
                    break
            
            logger.info(f"Hybrid search returned {len(filtered_results)} filtered results")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error performing hybrid search: {e}")
            return []
    
    def keyword_search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """
        Perform keyword-based search on document content.
        
        Args:
            query: Search query string
            k: Number of top results to return
            
        Returns:
            List of tuples (Document, relevance_score)
        """
        try:
            if not self.embedding_manager.document_store:
                logger.warning("No document store available for keyword search")
                return []
            
            k = k or self.config.top_k
            query_terms = query.lower().split()
            
            logger.info(f"Performing keyword search for: {query_terms}")
            
            # Score documents based on keyword matches
            scored_docs = []
            
            for doc in self.embedding_manager.document_store.values():
                content = doc.page_content.lower()
                
                # Calculate keyword match score
                score = 0.0
                for term in query_terms:
                    # Count exact matches
                    exact_matches = len(re.findall(r'\b' + re.escape(term) + r'\b', content))
                    score += exact_matches
                    
                    # Bonus for matches in title/metadata
                    filename = doc.metadata.get('filename', '').lower()
                    if term in filename:
                        score += 2
                
                if score > 0:
                    # Normalize by document length
                    normalized_score = score / (len(content) / 1000 + 1)
                    scored_docs.append((doc, normalized_score))
            
            # Sort by score and return top k
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            results = scored_docs[:k]
            
            logger.info(f"Keyword search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error performing keyword search: {e}")
            return []
    
    def get_context_around_match(self, doc: Document, query: str, context_chars: int = 200) -> str:
        """
        Get context around the best matching part of a document.
        
        Args:
            doc: Document to search in
            query: Query to find matches for
            context_chars: Number of characters of context to include
            
        Returns:
            Text snippet with context around the match
        """
        try:
            content = doc.page_content
            query_terms = query.lower().split()
            
            # Find the best matching position
            best_pos = 0
            max_matches = 0
            
            for i in range(0, len(content) - context_chars, 50):
                chunk = content[i:i + context_chars * 2].lower()
                matches = sum(1 for term in query_terms if term in chunk)
                
                if matches > max_matches:
                    max_matches = matches
                    best_pos = i
            
            # Extract context around the best position
            start = max(0, best_pos - context_chars // 2)
            end = min(len(content), best_pos + context_chars * 3 // 2)
            
            context = content[start:end]
            
            # Clean up the context
            if start > 0:
                context = "..." + context
            if end < len(content):
                context = context + "..."
            
            return context.strip()
            
        except Exception as e:
            logger.error(f"Error extracting context: {e}")
            return doc.page_content[:context_chars] + "..."
    
    def retrieve_for_question(self, question: str, k: int = None, 
                            use_hybrid: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a question with formatted results.
        
        Args:
            question: Question to search for
            k: Number of results to return
            use_hybrid: Whether to use hybrid search
            
        Returns:
            List of formatted document results
        """
        try:
            k = k or self.config.top_k
            
            # Perform search
            if use_hybrid:
                results = self.similarity_search(question, k=k)
            else:
                results = self.similarity_search(question, k=k)
            
            # Format results
            formatted_results = []
            
            for i, (doc, score) in enumerate(results):
                # Get context around the match
                context = self.get_context_around_match(doc, question)
                
                result = {
                    'rank': i + 1,
                    'content': context,
                    'full_content': doc.page_content,
                    'score': float(score),
                    'metadata': {
                        'filename': doc.metadata.get('filename', 'Unknown'),
                        'page': doc.metadata.get('chunk_page', 1),
                        'chunk_id': doc.metadata.get('chunk_id', 0),
                        'file_size': doc.metadata.get('file_size', 0),
                        'total_chunks': doc.metadata.get('total_chunks', 0)
                    }
                }
                
                formatted_results.append(result)
            
            logger.info(f"Retrieved and formatted {len(formatted_results)} results for question")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error retrieving documents for question: {e}")
            return []
    
    def get_document_summary(self) -> Dict[str, Any]:
        """Get summary of available documents for retrieval."""
        if not self.embedding_manager.document_store:
            return {'total_documents': 0, 'files': {}}
        
        files = {}
        for doc in self.embedding_manager.document_store.values():
            filename = doc.metadata.get('filename', 'unknown')
            if filename not in files:
                files[filename] = {
                    'chunks': 0,
                    'pages': doc.metadata.get('num_pages', 0),
                    'size': doc.metadata.get('file_size', 0)
                }
            files[filename]['chunks'] += 1
        
        return {
            'total_documents': len(self.embedding_manager.document_store),
            'total_files': len(files),
            'files': files
        }