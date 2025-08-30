# 🤖 AI-Powered PDF Research Assistant

A comprehensive RAG (Retrieval-Augmented Generation) application that allows you to upload multiple PDFs, ask natural language questions, and receive AI-powered answers with precise source citations.

## ✨ Features

- **📁 Multi-PDF Upload**: Drag-and-drop interface for uploading multiple PDF documents
- **🧠 Intelligent Chunking**: Advanced text splitting with context preservation
- **🔍 Similarity Search**: FAISS-powered vector search for relevant content retrieval
- **🤖 Multiple LLM Providers**: Support for OpenAI, OpenRouter (free tier), and local models
- **💾 Persistent Storage**: Embeddings cached on disk to avoid recomputation
- **📊 Source Citations**: Detailed source tracking with page numbers and excerpts
- **⚙️ Configurable Parameters**: Adjustable search parameters and model settings
- **📈 Evaluation Logging**: Optional Q&A logging for model evaluation
- **🐳 Docker Support**: Containerized deployment ready

## 🚀 Quick Start

### Local Development

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd pdf-research-assistant
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (optional)
   ```

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

4. **Access the App**
   Open http://localhost:8501 in your browser

### Docker Deployment

```bash
# Build the image
docker build -t pdf-research-assistant .

# Run the container
docker run -p 8501:8501 \
  -v $(pwd)/storage:/app/storage \
  -v $(pwd)/logs:/app/logs \
  pdf-research-assistant
```

## 🔑 LLM Provider Configuration

The application supports multiple LLM providers with automatic fallback:

### 1. OpenAI (Paid)
```bash
export OPENAI_API_KEY="your-api-key"
export MODEL_NAME="gpt-3.5-turbo"
```

### 2. OpenRouter (Free Tier Available)
OpenRouter provides access to free models with API key (sign up at openrouter.ai):

```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
export OPENROUTER_MODEL="meta-llama/llama-3.2-3b-instruct:free"
```

**Free Models Available:**
- `meta-llama/llama-3.2-3b-instruct:free` - Meta Llama 3.2 3B (free tier)
- `microsoft/phi-3-mini-128k-instruct:free` - Microsoft Phi-3 Mini (free tier)
- `google/gemma-2-9b-it:free` - Google Gemma 2 9B (free tier)
- Various other free community models

**Getting OpenRouter API Key:**
1. Visit https://openrouter.ai/
2. Sign up for a free account
3. Go to API Keys section and generate a new key
4. Free tier includes $1 of credits to get started

### 3. Local Transformers (Offline)
No API key required - runs entirely offline:
- Automatically downloads `microsoft/DialoGPT-medium`
- Runs on CPU (configurable for GPU)
- Slower but completely free and private

### Provider Priority
The system automatically selects providers in this order:
1. **OpenAI** (if `OPENAI_API_KEY` is set)
2. **OpenRouter** (if `OPENROUTER_API_KEY` is set - free tier available)
3. **Local Transformers** (offline fallback)

## 📊 Streamlit Cloud Deployment

### 1. Fork the Repository
Fork this repository to your GitHub account.

### 2. Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Select your forked repository
4. Set the main file path: `app.py`

### 3. Configure Secrets
In your Streamlit Cloud dashboard, add these secrets:

```toml
# .streamlit/secrets.toml
[env]
OPENAI_API_KEY = "your-openai-key"  # Optional - for paid OpenAI
OPENROUTER_API_KEY = "your-openrouter-key"  # Free tier available
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "meta-llama/llama-3.2-3b-instruct:free"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4
MAX_TOKENS = 512
TEMPERATURE = 0.2
ENABLE_LOGGING = true
```

### 4. Free Mode Setup
For free operation on Streamlit Cloud:
- **Option 1**: Set `OPENROUTER_API_KEY` (sign up at openrouter.ai for free $1 credit)
- **Option 2**: Don't set any API keys - app will fall back to local transformers
- OpenRouter free tier provides access to high-quality models

## 🛠️ Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key (optional) |
| `OPENROUTER_API_KEY` | - | OpenRouter API key (free tier available) |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter API endpoint |
| `OPENROUTER_MODEL` | `meta-llama/llama-3.2-3b-instruct:free` | OpenRouter model name |
| `MODEL_NAME` | `gpt-3.5-turbo` | OpenAI model name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence transformer model |
| `TOP_K` | `4` | Number of documents to retrieve |
| `MAX_TOKENS` | `512` | Maximum response length |
| `TEMPERATURE` | `0.2` | LLM temperature (randomness) |
| `ENABLE_LOGGING` | `true` | Enable Q&A logging |

### Storage Configuration

| Path | Purpose |
|------|---------|
| `./storage/index.faiss` | FAISS vector index |
| `./storage/docstore.pkl` | Document metadata |
| `./storage/metadata.json` | Index metadata |
| `./logs/app.log` | Application logs |
| `./logs/qa_log.jsonl` | Q&A evaluation logs |

## 📖 Usage Guide

### 1. Upload Documents
- Use the "Upload PDFs" tab
- Drag-and-drop or browse for PDF files
- Click "Process Files" to extract and index content
- Wait for processing to complete

### 2. Ask Questions
- Switch to "Ask Questions" tab
- Type your question in natural language
- Adjust parameters in the sidebar if needed
- Click "Search" to get AI-powered answers

### 3. Review Sources
- Each answer includes source citations
- Click on source expandable sections to see excerpts
- Use "Show Full Content" to view complete chunks
- Check relevance scores to assess source quality

### 4. System Management
- Use sidebar controls to switch LLM providers
- Adjust Top-K, temperature, and token limits
- Rebuild index if needed (clears all documents)
- Monitor system status in "System Info" tab

## 📄 Test with Sample PDFs

For testing purposes, use these sample PDFs:

- **AI Research Paper**: [Attention Is All You Need (Transformer)](https://arxiv.org/pdf/1706.03762.pdf)
- **Technical Documentation**: [Python Official Tutorial PDF](https://docs.python.org/3/tutorial/)
- **Create your own**: Generate a simple PDF from any text document

## 📝 Sample Questions

Try these example questions with uploaded PDFs:

- "What are the main findings of this research?"
- "Summarize the methodology used in the study"
- "What are the key recommendations?"
- "Compare the results across different sections"
- "What limitations are mentioned?"

## 🔧 Troubleshooting

### Common Issues

**1. "No LLM providers available"**
- Ensure at least one provider is configured
- Check API keys and network connectivity
- Try switching to local mode in sidebar

**2. "No relevant documents found"**
- Verify PDFs were processed successfully
- Try different question phrasing
- Check if documents contain relevant content

**3. "Empty response from LLM"**
- Increase max_tokens parameter
- Try different temperature settings
- Switch to a different LLM provider

**4. Processing errors**
- Ensure PDFs are not password protected
- Check PDF file integrity
- Verify sufficient disk space in ./storage/

### Performance Optimization

**For better performance:**
- Use smaller PDFs (< 10MB each)
- Limit concurrent uploads
- Adjust chunk_size in ingest.py for your use case
- Consider using GPU for local models

**For memory constraints:**
- Reduce TOP_K parameter
- Use smaller embedding models
- Clear index regularly for large datasets

### Rebuilding Index

To completely reset the system:

1. **Via UI**: Use "Rebuild Index" in sidebar
2. **Manual cleanup**:
   ```bash
   rm -rf storage/*
   rm -rf logs/*
   ```
3. **Restart application**

## 📊 Evaluation and Logging

### Q&A Logging
When `ENABLE_LOGGING=true`, all questions and answers are logged to `./logs/qa_log.jsonl`:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "question": "What is the main conclusion?",
  "answer": "Based on the documents...",
  "sources": [...],
  "metadata": {...}
}
```

### Quality Metrics
The system tracks:
- Answer completeness
- Source relevance scores
- Citation quality
- Response time
- Provider reliability

## 🔬 Testing

### Test with Sample PDFs
For testing, you can use these sample documents:
- [Sample Research Paper](https://arxiv.org/pdf/1706.03762.pdf) (Attention Is All You Need)
- [Sample Report](https://www.ipcc.ch/site/assets/uploads/2018/02/SYR_AR5_FINAL_full.pdf) (IPCC Summary)

### Validation Steps
1. Upload test PDFs
2. Verify processing completes without errors
3. Ask sample questions and review answers
4. Check source citations for accuracy
5. Test different LLM providers
6. Verify logging functionality

## 🚀 Production Deployment

### Recommended Setup
- Use OpenAI for best results (paid)
- Configure persistent storage volumes
- Set up monitoring and alerting
- Implement rate limiting
- Use HTTPS/SSL
- Regular backup of storage directory

### Scaling Considerations
- Horizontal scaling not currently supported
- Single instance recommended
- Consider Redis for session management
- Implement proper error handling
- Monitor memory usage with large document sets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Update documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: GitHub Issues tracker
- **Discussions**: GitHub Discussions
- **Documentation**: This README and inline code comments

## 🔄 Version History

- **v1.0.0**: Initial release with full RAG pipeline
- **v1.0.1**: Added multi-provider LLM support
- **v1.0.2**: OpenRouter free tier integration
- **v1.0.3**: Improved error handling and UI

---

**Built with ❤️ using Streamlit, LangChain, and FAISS**