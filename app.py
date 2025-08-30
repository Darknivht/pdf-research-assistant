"""
Streamlit frontend for the AI-Powered PDF Research Assistant.
"""

import streamlit as st
import os
import logging
from pathlib import Path
from typing import List, Dict, Any

# Import backend modules
from backend.utils import Config, setup_logging, format_file_size, truncate_text
from backend.ingest import process_uploaded_files, get_document_stats
from backend.embeddings import EmbeddingManager
from backend.retriever import DocumentRetriever
from backend.llm import LLMManager
from backend.rag import RAGPipeline

# Configure page
st.set_page_config(
    page_title="AI-Powered PDF Research Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

@st.cache_resource
def initialize_system():
    """Initialize the RAG system components."""
    try:
        config = Config()
        
        # Initialize embedding manager
        embedding_manager = EmbeddingManager(config=config)
        
        # Try to load existing index
        index_loaded = embedding_manager.load_index()
        if index_loaded:
            logger.info("Loaded existing FAISS index")
        else:
            logger.info("No existing index found, will create new one")
        
        # Initialize retriever
        retriever = DocumentRetriever(embedding_manager, config=config)
        
        # Initialize LLM manager
        llm_manager = LLMManager(config=config)
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(retriever, llm_manager, config=config)
        
        return {
            'config': config,
            'embedding_manager': embedding_manager,
            'retriever': retriever,
            'llm_manager': llm_manager,
            'rag_pipeline': rag_pipeline,
            'initialized': True
        }
        
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        st.error(f"Failed to initialize system: {e}")
        return {'initialized': False, 'error': str(e)}

def display_sidebar(system: Dict[str, Any]):
    """Display sidebar with configuration and controls."""
    st.sidebar.title("⚙️ Configuration")
    
    if not system.get('initialized', False):
        st.sidebar.error("System not initialized")
        return {}
    
    config = system['config']
    rag_pipeline = system['rag_pipeline']
    
    # LLM Provider Selection
    st.sidebar.subheader("🤖 LLM Provider")
    
    current_provider = system['llm_manager'].current_provider
    
    if current_provider == "No Provider Available":
        st.sidebar.error("⚠️ No LLM Provider Available")
        st.sidebar.warning("Configure an API key:")
        st.sidebar.markdown("• **OpenRouter**: Sign up at openrouter.ai (free $1 credit)")
        st.sidebar.markdown("• **OpenAI**: Add your OpenAI API key")
    else:
        st.sidebar.info(f"Current: {current_provider}")
    
    # Provider switching
    provider_options = ["OpenAI", "OpenRouter", "Local"]
    selected_provider = st.sidebar.selectbox(
        "Switch Provider",
        provider_options,
        index=provider_options.index(current_provider) if current_provider in provider_options else 0
    )
    
    if st.sidebar.button("Switch Provider") and selected_provider != current_provider:
        try:
            success = system['llm_manager'].switch_provider(selected_provider.lower())
            if success:
                st.sidebar.success(f"Switched to {selected_provider}")
                st.rerun()
            else:
                st.sidebar.error(f"Failed to switch to {selected_provider}")
        except Exception as e:
            st.sidebar.error(f"Error switching provider: {e}")
    
    # RAG Parameters
    st.sidebar.subheader("🔧 RAG Parameters")
    
    top_k = st.sidebar.slider("Top-K Documents", 1, 10, config.top_k, 
                             help="Number of relevant documents to retrieve")
    
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, config.temperature, 0.1,
                                   help="Controls randomness in LLM responses")
    
    max_tokens = st.sidebar.slider("Max Tokens", 100, 1000, config.max_tokens, 50,
                                  help="Maximum length of LLM response")
    
    # Logging
    st.sidebar.subheader("📊 Logging")
    enable_logging = st.sidebar.checkbox("Enable Q&A Logging", config.enable_logging,
                                        help="Log questions and answers for evaluation")
    
    # Index Management
    st.sidebar.subheader("🗂️ Index Management")
    
    # Display index stats
    embedding_stats = system['embedding_manager'].get_index_stats()
    if embedding_stats['num_documents'] > 0:
        st.sidebar.metric("Documents Indexed", embedding_stats['num_documents'])
        st.sidebar.metric("Files", embedding_stats['num_files'])
    else:
        st.sidebar.info("No documents indexed yet")
    
    # Rebuild index button
    if st.sidebar.button("🔄 Rebuild Index", help="Clear and rebuild the entire index"):
        if st.sidebar.confirm("Are you sure? This will delete all indexed documents."):
            try:
                system['embedding_manager'].clear_index()
                st.sidebar.success("Index cleared successfully")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error clearing index: {e}")
    
    return {
        'top_k': top_k,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'enable_logging': enable_logging
    }

def display_upload_section(system: Dict[str, Any]):
    """Display file upload section."""
    st.header("📁 Upload PDFs")
    
    if not system.get('initialized', False):
        st.error("Please wait for system initialization to complete.")
        return
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF files to analyze"
    )
    
    if uploaded_files:
        st.subheader("📊 Upload Summary")
        
        # Display file info
        total_size = sum(len(f.getvalue()) for f in uploaded_files)
        st.info(f"Selected {len(uploaded_files)} files ({format_file_size(total_size)})")
        
        # Process files button
        if st.button("🚀 Process Files", type="primary"):
            try:
                with st.spinner("Processing PDFs and building embeddings..."):
                    # Process uploaded files
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Extracting text from PDFs...")
                    progress_bar.progress(0.2)
                    
                    # Process documents
                    documents = process_uploaded_files(uploaded_files)
                    
                    if not documents:
                        st.error("No documents could be processed. Please check your PDF files.")
                        return
                    
                    status_text.text("Creating embeddings...")
                    progress_bar.progress(0.5)
                    
                    # Add documents to index
                    success = system['embedding_manager'].add_documents(documents)
                    
                    if not success:
                        st.error("Failed to add documents to index")
                        return
                    
                    status_text.text("Saving index...")
                    progress_bar.progress(0.8)
                    
                    # Save index
                    system['embedding_manager'].save_index()
                    
                    progress_bar.progress(1.0)
                    status_text.text("Processing complete!")
                    
                    # Display results
                    doc_stats = get_document_stats(documents)
                    st.success(f"Successfully processed {doc_stats['total_documents']} files "
                              f"into {doc_stats['total_chunks']} chunks")
                    
                    # Display file breakdown
                    with st.expander("📋 Processing Details"):
                        for filename, stats in doc_stats['files'].items():
                            st.write(f"**{filename}**: {stats['chunks']} chunks, "
                                   f"{stats['pages']} pages, "
                                   f"{format_file_size(stats['total_length'])} text")
                
            except Exception as e:
                logger.error(f"Error processing files: {e}")
                st.error(f"Error processing files: {e}")

def display_qa_section(system: Dict[str, Any], sidebar_config: Dict[str, Any]):
    """Display question-answering section."""
    st.header("❓ Ask Questions")
    
    if not system.get('initialized', False):
        st.error("System not initialized")
        return
    
    # Check if documents are indexed
    embedding_stats = system['embedding_manager'].get_index_stats()
    if embedding_stats['num_documents'] == 0:
        st.warning("⚠️ No documents are indexed yet. Please upload and process PDFs first.")
        return
    
    # Display current index info
    st.info(f"📚 Ready to answer questions from {embedding_stats['num_documents']} "
           f"document chunks across {embedding_stats['num_files']} files")
    
    # Question input
    question = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="Ask anything about your uploaded documents...",
        help="Type your question about the uploaded PDFs"
    )
    
    # Search button
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        search_button = st.button("🔍 Search", type="primary", disabled=not question.strip())
    
    with col2:
        if st.button("🗑️ Clear"):
            st.rerun()
    
    # Process question
    if search_button and question.strip():
        try:
            with st.spinner("Searching documents and generating answer..."):
                # Get answer using RAG pipeline
                response = system['rag_pipeline'].answer_question(
                    question=question,
                    k=sidebar_config.get('top_k', 4),
                    temperature=sidebar_config.get('temperature', 0.2),
                    max_tokens=sidebar_config.get('max_tokens', 512),
                    enable_logging=sidebar_config.get('enable_logging', True)
                )
                
                if response['success']:
                    # Display answer
                    st.subheader("💡 Answer")
                    st.write(response['answer'])
                    
                    # Display sources
                    if response['sources']:
                        st.subheader("📖 Sources")
                        
                        for i, source in enumerate(response['sources'], 1):
                            with st.expander(f"Source {i}: {source['filename']} (Page {source['page']})"):
                                st.write(f"**Relevance Score:** {source['score']:.3f}")
                                st.write(f"**Excerpt:**")
                                st.write(source['excerpt'])
                                
                                # Show full content button
                                if st.button(f"Show Full Content", key=f"full_content_{i}"):
                                    st.text_area(
                                        "Full Content:",
                                        source['full_content'],
                                        height=200,
                                        key=f"full_text_{i}"
                                    )
                    
                    # Display metadata
                    with st.expander("🔧 Technical Details"):
                        metadata = response['metadata']
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Documents Retrieved", metadata['retrieved_docs'])
                        
                        with col2:
                            st.metric("LLM Provider", metadata['llm_provider'])
                        
                        with col3:
                            st.metric("Prompt Length", metadata['prompt_length'])
                        
                        # Show parameters
                        st.write("**Parameters:**")
                        st.json(metadata.get('llm_params', {}))
                
                else:
                    st.error(f"Error generating answer: {response.get('error', 'Unknown error')}")
                    
                    if response.get('sources'):
                        st.info("Retrieved documents are available below, but answer generation failed.")
                        for i, source in enumerate(response['sources'], 1):
                            with st.expander(f"Source {i}: {source['filename']}"):
                                st.write(source['content'])
                
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            st.error(f"Error processing question: {e}")

def display_system_info(system: Dict[str, Any]):
    """Display system information and status."""
    with st.expander("🖥️ System Information"):
        if not system.get('initialized', False):
            st.error("System not initialized")
            return
        
        try:
            status = system['rag_pipeline'].get_pipeline_status()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Document Statistics")
                docs = status.get('documents', {})
                if docs.get('total_documents', 0) > 0:
                    st.write(f"**Total Documents:** {docs['total_documents']}")
                    st.write(f"**Total Files:** {docs['total_files']}")
                    
                    if 'files' in docs:
                        st.write("**Files:**")
                        for filename, info in docs['files'].items():
                            st.write(f"- {truncate_text(filename, 30)}: {info['chunks']} chunks")
                else:
                    st.write("No documents indexed")
            
            with col2:
                st.subheader("🤖 LLM Information")
                llm = status.get('llm', {})
                st.write(f"**Provider:** {llm.get('provider', 'Unknown')}")
                st.write(f"**Model:** {llm.get('model_name', 'Unknown')}")
                st.write(f"**Max Tokens:** {llm.get('max_tokens', 'Unknown')}")
                st.write(f"**Temperature:** {llm.get('temperature', 'Unknown')}")
            
            st.subheader("⚙️ Configuration")
            config = status.get('config', {})
            st.json(config)
            
        except Exception as e:
            st.error(f"Error getting system status: {e}")

def main():
    """Main Streamlit application."""
    st.title("🤖 AI-Powered PDF Research Assistant")
    st.markdown("Upload PDFs and ask questions to get AI-powered answers with source citations.")
    
    # Initialize system
    with st.spinner("Initializing system..."):
        system = initialize_system()
    
    if not system.get('initialized', False):
        st.error("Failed to initialize the system. Please check the logs for details.")
        if 'error' in system:
            st.error(f"Error: {system['error']}")
        return
    
    # Display sidebar configuration
    sidebar_config = display_sidebar(system)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📁 Upload Documents", "❓ Ask Questions", "ℹ️ System Info"])
    
    with tab1:
        display_upload_section(system)
    
    with tab2:
        display_qa_section(system, sidebar_config)
    
    with tab3:
        display_system_info(system)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**💡 Pro Tips:**\n"
        "- Upload multiple PDFs for comprehensive knowledge coverage\n"
        "- Ask specific questions for better results\n"
        "- Use the sidebar to adjust search parameters\n"
        "- Check sources for accuracy and context"
    )

if __name__ == "__main__":
    main()