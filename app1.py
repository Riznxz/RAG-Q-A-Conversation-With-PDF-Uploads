import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import time
import uuid
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🧠 Smart PDF Chat Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #667eea;
    }
    
    .assistant-message {
        background-color: #e8f4f8;
        border-left-color: #764ba2;
    }
    
    .stat-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'store' not in st.session_state:
        st.session_state.store = {}
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'conversation_count' not in st.session_state:
        st.session_state.conversation_count = 0
    if 'chat_sessions' not in st.session_state:
        st.session_state.chat_sessions = {}

initialize_session_state()

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 Smart PDF Chat Assistant</h1>
    <p>Upload PDFs and have intelligent conversations with their content using AI</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input("🔑 Enter your Groq API key:", type="password", help="Get your API key from Groq Console")
    
    if api_key:
        st.success("✅ API Key configured!")
        
        # Model selection
        model_options = {
            "Gemma2-9b-It": "Fast and efficient",
            "llama3-8b-8192": "Balanced performance", 
            "llama3-70b-8192": "High accuracy (slower)"
        }
        
        selected_model = st.selectbox(
            "🤖 Choose AI Model:",
            options=list(model_options.keys()),
            help="Different models offer various speed/accuracy trade-offs"
        )
        
        st.info(f"📊 {model_options[selected_model]}")
        
        # Session management
        st.subheader("💬 Chat Sessions")
        
        # Create new session
        if st.button("➕ New Chat Session"):
            new_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.chat_sessions[new_session_id] = {
                'created': datetime.now(),
                'message_count': 0
            }
            st.session_state.current_session = new_session_id
            st.success("New session created!")
        
        # Select existing session
        if st.session_state.chat_sessions:
            session_options = {
                session_id: f"{session_id} ({info['message_count']} messages)"
                for session_id, info in st.session_state.chat_sessions.items()
            }
            
            current_session = st.selectbox(
                "Select Chat Session:",
                options=list(session_options.keys()),
                format_func=lambda x: session_options[x]
            )
        else:
            current_session = "default_session"
            
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            chunk_size = st.slider("Chunk Size", 1000, 10000, 5000, help="Size of text chunks for processing")
            chunk_overlap = st.slider("Chunk Overlap", 0, 1000, 500, help="Overlap between chunks")
            max_tokens = st.slider("Max Response Length", 50, 500, 150, help="Maximum tokens in AI response")
            temperature = st.slider("Creativity", 0.0, 1.0, 0.1, help="Higher values = more creative responses")

# Main content area
if not api_key:
    st.markdown("""
    <div class="warning-message">
        <strong>⚠️ Getting Started:</strong><br>
        1. Get your API key from <a href="https://console.groq.com" target="_blank">Groq Console</a><br>
        2. Enter it in the sidebar<br>
        3. Upload PDF files to start chatting!
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Initialize LLM
try:
    llm = ChatGroq(
        groq_api_key=api_key, 
        model_name=selected_model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"❌ Error initializing AI model: {str(e)}")
    st.stop()

# File upload section
st.subheader("📁 Document Upload")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF files to chat with"
    )

with col2:
    if st.session_state.processed_files:
        st.metric("📄 Processed Files", len(st.session_state.processed_files))
    if st.session_state.vector_store:
        st.metric("🔍 Ready for Chat", "✅")

# Process uploaded files
if uploaded_files:
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    
    if new_files:
        with st.spinner("🔄 Processing documents... This may take a moment."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            documents = []
            
            for i, uploaded_file in enumerate(new_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                progress_bar.progress((i + 1) / len(new_files))
                
                # Save temp file
                temp_pdf = f"./temp_{uuid.uuid4().hex}.pdf"
                try:
                    with open(temp_pdf, "wb") as file:
                        file.write(uploaded_file.getvalue())
                    
                    # Load and process PDF
                    loader = PyPDFLoader(temp_pdf)
                    docs = loader.load()
                    documents.extend(docs)
                    
                    st.session_state.processed_files.append(uploaded_file.name)
                    
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_pdf):
                        os.remove(temp_pdf)
            
            if documents:
                # Split and create embeddings
                status_text.text("Creating embeddings...")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap
                )
                splits = text_splitter.split_documents(documents)
                
                if st.session_state.vector_store is None:
                    st.session_state.vector_store = Chroma.from_documents(
                        documents=splits, 
                        embedding=embeddings
                    )
                else:
                    st.session_state.vector_store.add_documents(splits)
                
                progress_bar.progress(1.0)
                status_text.empty()
                
                st.markdown("""
                <div class="success-message">
                    <strong>✅ Success!</strong> Documents processed and ready for chat.
                </div>
                """, unsafe_allow_html=True)
                
                # Show document stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📄 Total Files", len(st.session_state.processed_files))
                with col2:
                    st.metric("📝 Text Chunks", len(splits))
                with col3:
                    st.metric("📊 Total Pages", sum(len(doc.page_content.split('\n')) for doc in documents))

# Chat interface
if st.session_state.vector_store:
    st.subheader("💬 Chat with your Documents")
    
    # Initialize RAG chain
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Contextualization prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference "
        "context in the chat history, formulate a standalone question which can be "
        "understood without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # QA prompt
    system_prompt = (
        "You are a knowledgeable assistant helping users understand their documents. "
        "Use the retrieved context to provide accurate, helpful answers. "
        "If you don't know something, admit it honestly. "
        "Be conversational and engaging while staying factual. "
        "Use markdown formatting for better readability when appropriate.\n\n"
        "Context: {context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
            st.session_state.chat_sessions[session] = {
                'created': datetime.now(),
                'message_count': 0
            }
        return st.session_state.store[session]
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    # Chat input
    user_input = st.chat_input("Ask me anything about your documents...")
    
    # Sample questions
    with st.expander("💡 Try these sample questions"):
        sample_questions = [
            "What is the main topic of the uploaded documents?",
            "Can you summarize the key points?",
            "What are the important dates mentioned?",
            "Who are the key people or organizations discussed?",
            "What conclusions can you draw from the content?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(sample_questions):
            col = cols[i % 2]
            if col.button(f"💭 {question}", key=f"sample_{i}"):
                user_input = question
    
    # Process user input
    if user_input:
        session_history = get_session_history(current_session)
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = conversational_rag_chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": current_session}}
                    )
                    
                    # Display response
                    st.write(response['answer'])
                    
                    # Update session stats
                    if current_session in st.session_state.chat_sessions:
                        st.session_state.chat_sessions[current_session]['message_count'] += 1
                    
                    # Show source documents in expander
                    if 'context' in response:
                        with st.expander("📚 Source Documents"):
                            for i, doc in enumerate(response['context']):
                                st.text_area(
                                    f"Source {i+1}:",
                                    doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                                    height=100,
                                    key=f"source_{i}"
                                )
                                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Chat history display
    if current_session in st.session_state.store:
        messages = st.session_state.store[current_session].messages
        if messages and st.checkbox("📜 Show Chat History"):
            st.subheader("Chat History")
            for i, message in enumerate(messages[-10:]):  # Show last 10 messages
                message_type = "user" if message.type == "human" else "assistant"
                with st.chat_message(message_type):
                    st.write(message.content)

else:
    st.info("👆 Upload some PDF files above to start chatting!")
    
    # Show app features
    st.subheader("🌟 Features")
    features_col1, features_col2 = st.columns(2)
    
    with features_col1:
        st.markdown("""
        **📄 Document Processing**
        - Multi-PDF upload support
        - Intelligent text chunking
        - Advanced embedding generation
        
        **💬 Smart Chat**
        - Context-aware conversations
        - Chat history preservation
        - Multiple conversation sessions
        """)
    
    with features_col2:
        st.markdown("""
        **🤖 AI Models**
        - Multiple Groq model options
        - Customizable response length
        - Adjustable creativity settings
        
        **🎨 User Experience**
        - Beautiful, responsive interface
        - Real-time processing feedback
        - Source document references
        """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, LangChain, and Groq AI")