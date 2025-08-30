"""
PDF parsing, chunking, and metadata extraction functionality.
"""

import logging
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
import hashlib

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from .utils import get_file_hash

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF parsing and text extraction."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text content and metadata from PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                
                # Extract metadata
                metadata = {
                    'filename': Path(pdf_path).name,
                    'filepath': pdf_path,
                    'num_pages': len(pdf_reader.pages),
                    'file_hash': get_file_hash(pdf_path),
                    'file_size': Path(pdf_path).stat().st_size,
                }
                
                # Try to extract PDF metadata
                try:
                    pdf_metadata = pdf_reader.metadata
                    if pdf_metadata:
                        metadata.update({
                            'title': pdf_metadata.get('/Title', ''),
                            'author': pdf_metadata.get('/Author', ''),
                            'subject': pdf_metadata.get('/Subject', ''),
                            'creator': pdf_metadata.get('/Creator', ''),
                            'producer': pdf_metadata.get('/Producer', ''),
                            'creation_date': str(pdf_metadata.get('/CreationDate', '')),
                            'modification_date': str(pdf_metadata.get('/ModDate', '')),
                        })
                except Exception as e:
                    logger.warning(f"Could not extract PDF metadata from {pdf_path}: {e}")
                
                # Extract text from all pages
                full_text = ""
                page_texts = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            page_text = self._clean_text(page_text)
                            page_texts.append((page_num + 1, page_text))
                            full_text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1} of {pdf_path}: {e}")
                        continue
                
                metadata['page_texts'] = page_texts
                
                if not full_text.strip():
                    raise ValueError(f"No text content could be extracted from {pdf_path}")
                
                logger.info(f"Successfully extracted {len(full_text)} characters from {metadata['filename']}")
                return full_text, metadata
                
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
        
        # Fix common spacing issues
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)  # Ensure space after sentence
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenated words across lines
        
        # Remove excessive newlines but preserve paragraph breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()
    
    def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Chunk document text into smaller pieces with metadata.
        
        Args:
            text: Full document text
            metadata: Document metadata
            
        Returns:
            List of Document objects with chunks
        """
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            documents = []
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                
                # Determine which page(s) this chunk likely comes from
                chunk_page = self._estimate_chunk_page(chunk, metadata.get('page_texts', []))
                
                # Create document with metadata
                doc_metadata = {
                    **metadata,
                    'chunk_id': i,
                    'chunk_page': chunk_page,
                    'chunk_length': len(chunk),
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
                
                # Remove page_texts from individual chunk metadata to save space
                if 'page_texts' in doc_metadata:
                    del doc_metadata['page_texts']
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=doc_metadata
                ))
            
            logger.info(f"Created {len(documents)} chunks from {metadata.get('filename', 'document')}")
            return documents
            
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            raise
    
    def _estimate_chunk_page(self, chunk: str, page_texts: List[Tuple[int, str]]) -> int:
        """Estimate which page a chunk belongs to based on text matching."""
        if not page_texts:
            return 1
        
        best_match_page = 1
        max_overlap = 0
        
        chunk_words = set(chunk.lower().split())
        
        for page_num, page_text in page_texts:
            page_words = set(page_text.lower().split())
            overlap = len(chunk_words.intersection(page_words))
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_match_page = page_num
        
        return best_match_page
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """
        Complete PDF processing pipeline.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of Document objects with chunks
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text and metadata
        text, metadata = self.extract_text_from_pdf(pdf_path)
        
        # Chunk the document
        documents = self.chunk_document(text, metadata)
        
        logger.info(f"Successfully processed {pdf_path} into {len(documents)} chunks")
        return documents

def process_uploaded_files(uploaded_files) -> List[Document]:
    """
    Process multiple uploaded PDF files.
    
    Args:
        uploaded_files: List of uploaded file objects from Streamlit
        
    Returns:
        List of Document objects from all processed PDFs
    """
    processor = PDFProcessor()
    all_documents = []
    
    for uploaded_file in uploaded_files:
        try:
            # Save uploaded file temporarily
            temp_path = f"./temp_{uploaded_file.name}"
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Process the PDF
            documents = processor.process_pdf(temp_path)
            all_documents.extend(documents)
            
            # Clean up temporary file
            Path(temp_path).unlink(missing_ok=True)
            
        except Exception as e:
            logger.error(f"Error processing uploaded file {uploaded_file.name}: {e}")
            continue
    
    return all_documents

def get_document_stats(documents: List[Document]) -> Dict[str, Any]:
    """Get statistics about processed documents."""
    if not documents:
        return {}
    
    # Group by filename
    files = {}
    for doc in documents:
        filename = doc.metadata.get('filename', 'unknown')
        if filename not in files:
            files[filename] = {
                'chunks': 0,
                'total_length': 0,
                'pages': doc.metadata.get('num_pages', 0)
            }
        files[filename]['chunks'] += 1
        files[filename]['total_length'] += len(doc.page_content)
    
    return {
        'total_documents': len(files),
        'total_chunks': len(documents),
        'total_characters': sum(len(doc.page_content) for doc in documents),
        'files': files
    }