# 🧠 Smart PDF Chat Assistant

An intelligent conversational AI application that allows you to upload PDF documents and have natural, context-aware conversations about their content. Built with Streamlit, LangChain, and powered by Groq's lightning-fast LLM inference.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![LangChain](https://img.shields.io/badge/LangChain-Latest-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
(q&a 2.png)

## ✨ Key Features

### 📄 Advanced Document Processing
- **Multi-PDF Upload** - Process multiple PDF files simultaneously
- **Intelligent Chunking** - Smart text splitting with configurable chunk size and overlap
- **Persistent Storage** - ChromaDB vector store for efficient document retrieval
- **Real-time Progress** - Visual feedback during document processing

### 💬 Conversational AI
- **Context-Aware Chat** - AI remembers conversation history for follow-up questions
- **Multiple Chat Sessions** - Manage separate conversations for different topics
- **History Preservation** - Full chat history tracking and display
- **Source Citations** - See which document sections were used for each answer

### 🤖 Flexible AI Models
- **Gemma2-9b-It** - Fast and efficient responses
- **LLaMA 3 8B** - Balanced performance for most use cases
- **LLaMA 3 70B** - Maximum accuracy for complex queries

### 🎨 User Experience
- **Beautiful UI** - Modern gradient design with smooth animations
- **Sample Questions** - Pre-built question templates to get started
- **Session Management** - Create, switch between, and track multiple chat sessions
- **Customizable Settings** - Adjust temperature, response length, chunk size, and more

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Groq API Key ([Sign up here](https://console.groq.com))
- 2GB+ RAM recommended for embedding models

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/smart-pdf-chat.git
cd smart-pdf-chat
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables (Optional)**

Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

*Note: You can also enter your API key directly in the sidebar when running the app.*

5. **Run the application**
```bash
streamlit run app1.py
```

The app will open automatically in your browser at `http://localhost:8501`

## 📋 Requirements

Create a `requirements.txt` file with:

```txt
streamlit>=1.28.0
langchain>=0.1.0
langchain-groq>=0.0.1
langchain-chroma>=0.1.0
langchain-community>=0.0.13
langchain-core>=0.1.0
langchain-huggingface>=0.0.1
langchain-text-splitters>=0.0.1
chromadb>=0.4.22
pypdf>=3.17.0
python-dotenv>=1.0.0
sentence-transformers>=2.2.0
```

## 🎯 Usage Guide

### Getting Started

1. **Configure API Key**
   - Open the sidebar
   - Enter your Groq API key in the password field
   - See the ✅ confirmation message

2. **Select AI Model**
   - Choose from Gemma2, LLaMA 3 8B, or LLaMA 3 70B
   - Each model offers different speed/accuracy trade-offs

3. **Upload Documents**
   - Click "Browse files" or drag-and-drop PDFs
   - Multiple files can be uploaded at once
   - Watch the progress bar as documents are processed

4. **Start Chatting**
   - Type your question in the chat input
   - Or click sample questions to get started
   - View AI responses with source citations

### Advanced Features

#### Session Management
- **Create New Sessions**: Start fresh conversations for different topics
- **Switch Sessions**: Select from existing sessions in the sidebar
- **Track Activity**: See message count for each session

#### Customization Options
- **Chunk Size**: Control how documents are split (1000-10000 chars)
- **Chunk Overlap**: Set overlap between chunks (0-1000 chars)
- **Max Tokens**: Limit response length (50-500 tokens)
- **Temperature**: Adjust creativity level (0.0-1.0)

#### Chat History
- Enable "📜 Show Chat History" to view past messages
- Last 10 messages displayed for each session
- Full history preserved across sessions

## 🏗️ Architecture

```
┌─────────────────────┐
│   PDF Upload UI     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  PyPDFLoader        │
│  (Extract Text)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Text Splitter       │
│ (Chunking)          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ HuggingFace         │
│ Embeddings          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ ChromaDB            │
│ Vector Store        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ History-Aware       │
│ Retriever           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Groq LLM            │
│ (Response Gen)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Chat UI             │
│ (with History)      │
└─────────────────────┘
```

## ⚙️ Configuration

### Model Selection

```python
model_options = {
    "Gemma2-9b-It": "Fast and efficient",
    "llama3-8b-8192": "Balanced performance", 
    "llama3-70b-8192": "High accuracy (slower)"
}
```

### Text Splitting Configuration

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,      # Adjustable via slider
    chunk_overlap=500     # Adjustable via slider
)
```

### Retriever Settings

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}  # Returns top 3 relevant chunks
)
```

### LLM Parameters

```python
llm = ChatGroq(
    groq_api_key=api_key,
    model_name=selected_model,
    temperature=0.1,      # Adjustable via slider
    max_tokens=150        # Adjustable via slider
)
```

## 📁 Project Structure

```
smart-pdf-chat/
│
├── app1.py                # Main application file
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (optional)
├── .gitignore           # Git ignore file
├── README.md            # This file
│
├── temp_*.pdf           # Temporary files (auto-cleaned)
└── chroma_db/           # ChromaDB storage (auto-created)
```

## 🔒 Security & Privacy

- **API Keys**: Stored securely in session state, never logged
- **Temporary Files**: Automatically cleaned after processing
- **Local Processing**: Embeddings created locally on your machine
- **No Data Retention**: Documents not stored on external servers

## 🛠️ Customization

### Modifying the System Prompt

```python
system_prompt = (
    "You are a knowledgeable assistant helping users understand their documents. "
    "Use the retrieved context to provide accurate, helpful answers. "
    "If you don't know something, admit it honestly. "
    "Be conversational and engaging while staying factual. "
    "Use markdown formatting for better readability when appropriate.\n\n"
    "Context: {context}"
)
```

### Changing Embedding Model

```python
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # Change to any HuggingFace model
)
```

### Adjusting Retrieval

```python
retriever = vector_store.as_retriever(
    search_type="similarity",      # or "mmr" for diversity
    search_kwargs={"k": 3}         # Number of chunks to retrieve
)
```

## 💡 Sample Questions

Try these questions to explore your documents:
- "What is the main topic of the uploaded documents?"
- "Can you summarize the key points?"
- "What are the important dates mentioned?"
- "Who are the key people or organizations discussed?"
- "What conclusions can you draw from the content?"

## 🐛 Troubleshooting

### Common Issues

**Issue**: API Key Error
- **Solution**: Ensure your Groq API key is valid and has available credits

**Issue**: Memory Error during processing
- **Solution**: Reduce chunk size or process fewer documents at once

**Issue**: Slow responses
- **Solution**: Switch to a faster model (Gemma2-9b-It) or reduce max_tokens

**Issue**: Embeddings fail to load
- **Solution**: Ensure you have internet connection for first-time model download

## 🚀 Performance Tips

1. **Optimal Chunk Size**: 5000 characters works well for most documents
2. **Chunk Overlap**: 500 characters provides good context continuity
3. **Model Selection**: Use Gemma2 for speed, LLaMA 3 70B for accuracy
4. **Batch Processing**: Upload all PDFs at once rather than one at a time
5. **Session Management**: Create new sessions for unrelated topics

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Update README for new features
- Test with multiple PDF types

## 📊 Roadmap

- [ ] Support for DOCX, TXT, and other document formats
- [ ] Export chat history to PDF/JSON
- [ ] Multi-language support
- [ ] Document comparison feature
- [ ] Advanced filtering and search
- [ ] Cloud storage integration (S3, Google Drive)
- [ ] API endpoint for programmatic access
- [ ] Mobile-responsive improvements
- [ ] Voice input/output capabilities
- [ ] Collaborative chat sessions

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Streamlit](https://streamlit.io/)** - For the amazing web framework
- **[LangChain](https://langchain.com/)** - For RAG and chain implementations
- **[Groq](https://groq.com/)** - For ultra-fast LLM inference
- **[HuggingFace](https://huggingface.co/)** - For embedding models
- **[ChromaDB](https://www.trychroma.com/)** - For vector storage

## 📧 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/smart-pdf-chat/issues)
- **Email**: your.email@example.com
- **Twitter**: [@yourhandle](https://twitter.com/yourhandle)
- **Documentation**: [Wiki](https://github.com/yourusername/smart-pdf-chat/wiki)

## 🌟 Star History

If you find this project helpful, please consider giving it a star! ⭐

---

**Made with ❤️ using Streamlit, LangChain, and Groq AI**

*Last Updated: October 2025*
