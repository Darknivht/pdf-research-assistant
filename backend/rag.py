"""
RAG (Retrieval-Augmented Generation) pipeline implementation.
"""

import logging
from typing import List, Dict, Any, Optional
import json

from .retriever import DocumentRetriever
from .llm import LLMManager
from .utils import Config, log_qa_pair

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG pipeline for question answering with document retrieval."""
    
    def __init__(self, retriever: DocumentRetriever, llm_manager: LLMManager, config: Config = None):
        self.retriever = retriever
        self.llm_manager = llm_manager
        self.config = config or Config()
        
        # Define the RAG prompt template
        self.rag_prompt_template = """You are an AI assistant that answers questions based on the provided document context. 
Use ONLY the information from the provided documents to answer the question. Be precise and cite your sources.

Context Documents:
{context}

Question: {question}

Instructions:
1. Answer the question using ONLY information from the provided documents
2. If the documents don't contain enough information to answer the question, say so clearly
3. Cite your sources by referencing the document name and page number when possible
4. Be concise but thorough in your response
5. If there are conflicting information in the documents, mention this

Answer:"""
    
    def _format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context string."""
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            source_info = f"[Document: {doc['metadata']['filename']}, Page: {doc['metadata']['page']}]"
            content = doc['content']
            
            context_part = f"Source {i}: {source_info}\n{content}\n"
            context_parts.append(context_part)
        
        return "\n" + "-" * 50 + "\n".join(context_parts)
    
    def _create_rag_prompt(self, question: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Create the complete RAG prompt with context and question."""
        context = self._format_context(retrieved_docs)
        
        return self.rag_prompt_template.format(
            context=context,
            question=question
        )
    
    def answer_question(self, question: str, k: int = None, 
                       temperature: float = None, max_tokens: int = None,
                       enable_logging: bool = None) -> Dict[str, Any]:
        """
        Answer a question using RAG pipeline.
        
        Args:
            question: User question
            k: Number of documents to retrieve
            temperature: LLM temperature override
            max_tokens: Max tokens override
            enable_logging: Whether to log Q&A pairs
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        try:
            if not question.strip():
                return {
                    'success': False,
                    'error': 'Question cannot be empty',
                    'answer': '',
                    'sources': [],
                    'metadata': {}
                }
            
            # Use config defaults if not specified
            k = k or self.config.top_k
            temperature = temperature or self.config.temperature
            max_tokens = max_tokens or self.config.max_tokens
            enable_logging = enable_logging if enable_logging is not None else self.config.enable_logging
            
            logger.info(f"Processing question: {question[:100]}...")
            
            # Step 1: Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve_for_question(question, k=k)
            
            if not retrieved_docs:
                return {
                    'success': True,
                    'answer': "I couldn't find any relevant documents to answer your question. Please make sure you have uploaded PDFs and they have been processed successfully.",
                    'sources': [],
                    'metadata': {
                        'retrieved_docs': 0,
                        'llm_provider': self.llm_manager.current_provider
                    }
                }
            
            # Step 2: Create RAG prompt
            rag_prompt = self._create_rag_prompt(question, retrieved_docs)
            
            logger.info(f"Generated RAG prompt with {len(retrieved_docs)} documents")
            
            # Step 3: Generate answer using LLM
            answer = self.llm_manager.generate(
                rag_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if not answer.strip():
                return {
                    'success': False,
                    'error': 'LLM generated empty response',
                    'answer': '',
                    'sources': retrieved_docs,
                    'metadata': {
                        'retrieved_docs': len(retrieved_docs),
                        'llm_provider': self.llm_manager.current_provider
                    }
                }
            
            # Step 4: Format sources for display
            formatted_sources = []
            for doc in retrieved_docs:
                source = {
                    'filename': doc['metadata']['filename'],
                    'page': doc['metadata']['page'],
                    'excerpt': doc['content'][:300] + "..." if len(doc['content']) > 300 else doc['content'],
                    'full_content': doc['full_content'],
                    'score': doc['score'],
                    'rank': doc['rank']
                }
                formatted_sources.append(source)
            
            # Step 5: Prepare response metadata
            response_metadata = {
                'retrieved_docs': len(retrieved_docs),
                'llm_provider': self.llm_manager.current_provider,
                'llm_params': {
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'k': k
                },
                'prompt_length': len(rag_prompt)
            }
            
            # Step 6: Log Q&A pair if enabled
            if enable_logging:
                try:
                    log_qa_pair(
                        question=question,
                        answer=answer,
                        sources=formatted_sources,
                        metadata=response_metadata
                    )
                except Exception as e:
                    logger.warning(f"Failed to log Q&A pair: {e}")
            
            response = {
                'success': True,
                'answer': answer,
                'sources': formatted_sources,
                'metadata': response_metadata
            }
            
            logger.info(f"Successfully generated answer with {len(formatted_sources)} sources")
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return {
                'success': False,
                'error': str(e),
                'answer': '',
                'sources': [],
                'metadata': {}
            }
    
    def batch_answer_questions(self, questions: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Answer multiple questions in batch.
        
        Args:
            questions: List of questions
            **kwargs: Parameters to pass to answer_question
            
        Returns:
            List of answer responses
        """
        answers = []
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}")
            
            try:
                answer = self.answer_question(question, **kwargs)
                answers.append(answer)
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {e}")
                answers.append({
                    'success': False,
                    'error': str(e),
                    'answer': '',
                    'sources': [],
                    'metadata': {}
                })
        
        return answers
    
    def get_similar_questions(self, question: str, threshold: float = 0.8) -> List[str]:
        """
        Find similar questions from previous logs.
        
        Args:
            question: Current question
            threshold: Similarity threshold
            
        Returns:
            List of similar questions
        """
        # This would require loading previous Q&A logs and comparing
        # For now, return empty list as a placeholder
        return []
    
    def evaluate_answer_quality(self, question: str, answer: str, sources: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate the quality of a generated answer.
        
        Args:
            question: Original question
            answer: Generated answer
            sources: Source documents used
            
        Returns:
            Quality evaluation metrics
        """
        try:
            # Basic quality metrics
            metrics = {
                'answer_length': len(answer),
                'num_sources': len(sources),
                'avg_source_score': sum(s['score'] for s in sources) / len(sources) if sources else 0,
                'has_citations': any(word in answer.lower() for word in ['document', 'page', 'source']),
                'completeness_score': min(len(answer) / 100, 1.0),  # Normalize by expected length
            }
            
            # Check if answer acknowledges limitations
            limitation_phrases = [
                "don't have enough information",
                "documents don't contain",
                "not mentioned in",
                "cannot determine from"
            ]
            metrics['acknowledges_limitations'] = any(phrase in answer.lower() for phrase in limitation_phrases)
            
            # Overall quality score (simple heuristic)
            quality_score = (
                metrics['completeness_score'] * 0.3 +
                min(metrics['avg_source_score'], 1.0) * 0.3 +
                (1.0 if metrics['has_citations'] else 0.0) * 0.2 +
                min(metrics['num_sources'] / 4, 1.0) * 0.2
            )
            
            metrics['overall_quality'] = quality_score
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating answer quality: {e}")
            return {'error': str(e)}
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get status information about the RAG pipeline."""
        try:
            # Get retriever status
            doc_summary = self.retriever.get_document_summary()
            
            # Get LLM status
            llm_info = self.llm_manager.get_provider_info()
            
            # Get embedding status
            embedding_stats = self.retriever.embedding_manager.get_index_stats()
            
            return {
                'documents': doc_summary,
                'llm': llm_info,
                'embeddings': embedding_stats,
                'config': {
                    'top_k': self.config.top_k,
                    'max_tokens': self.config.max_tokens,
                    'temperature': self.config.temperature,
                    'enable_logging': self.config.enable_logging
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting pipeline status: {e}")
            return {'error': str(e)}