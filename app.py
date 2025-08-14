# QueryNest StudyMate - Enhanced PDF Analysis Application
import os, io, re, json, uuid, datetime
from pathlib import Path
from typing import List, Dict, Optional
import streamlit as st
from dotenv import load_dotenv
import requests
import time

# Core ML and NLP imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SK = True
except Exception:
    HAS_SK = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
    import torch
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

try:
    import speech_recognition as sr
    HAS_SPEECH = True
except Exception:
    HAS_SPEECH = False

try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="QueryNest StudyMate",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Setup directories
ROOT = Path.cwd()
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# HuggingFace API configuration
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HF_API_URL = "https://api-inference.huggingface.co/models/"

# Initialize session state
def init_session_state():
    if "kb" not in st.session_state:
        st.session_state.kb = []  # Knowledge base chunks
    if "vectorizer" not in st.session_state:
        st.session_state.vectorizer = None
    if "doc_matrix" not in st.session_state:
        st.session_state.doc_matrix = None
    if "notes" not in st.session_state:
        st.session_state.notes = []
    if "history" not in st.session_state:
        st.session_state.history = []
    if "current_pdf" not in st.session_state:
        st.session_state.current_pdf = None
    if "qa_pipeline" not in st.session_state:
        st.session_state.qa_pipeline = None
    if "summarizer" not in st.session_state:
        st.session_state.summarizer = None
    if "current_nav" not in st.session_state:
        st.session_state.current_nav = "Home"

init_session_state()

# Enhanced CSS styling for QueryNest
def load_css():
    accent = "#4f46e5"  # Indigo accent color
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {{
        --accent: {accent};
        --accent-light: #6366f1;
        --bg: #ffffff;
        --panel: #f8fafc;
        --text: #111827;
        --muted: #6b7280;
        --border: #e5e7eb;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
    }}

    .main {{
        font-family: 'Inter', sans-serif;
    }}

    .main-header {{
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-light) 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(79, 70, 229, 0.2);
    }}

    .main-header h1 {{
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}

    .main-header p {{
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }}

    .nav-container {{
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }}

    .content-card {{
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid var(--border);
        margin-bottom: 1.5rem;
    }}

    .feature-card {{
        background: var(--panel);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid var(--border);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }}

    .feature-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }}

    .success-message {{
        background: #d1fae5;
        color: #065f46;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--success);
        margin: 1rem 0;
    }}

    .warning-message {{
        background: #fef3c7;
        color: #92400e;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--warning);
        margin: 1rem 0;
    }}

    .error-message {{
        background: #fee2e2;
        color: #991b1b;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--error);
        margin: 1rem 0;
    }}

    .highlight {{
        background: #fef08a;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-weight: 500;
    }}

    .stats-container {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }}

    .stat-card {{
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid var(--border);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }}

    .stat-number {{
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent);
        margin-bottom: 0.5rem;
    }}

    .stat-label {{
        color: var(--muted);
        font-size: 0.9rem;
        font-weight: 500;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

load_css()

# Core utility functions
def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.replace('\xa0', ' ')).strip()
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-\'""]', ' ', text)
    return text

def extract_pages_from_pdf_bytes(file_bytes: bytes) -> List[Dict]:
    """Extract text content from PDF bytes."""
    try:
        import fitz  # pymupdf
    except ImportError:
        st.error("❌ PyMuPDF not installed. Run: pip install pymupdf")
        return []

    try:
        doc = fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf")
        pages = []

        for i, page in enumerate(doc):
            text = clean_text(page.get_text("text") or "")
            if text.strip():  # Only add pages with content
                pages.append({
                    "page": i + 1,
                    "text": text,
                    "word_count": len(text.split())
                })

        doc.close()
        return pages
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        return []

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    if not text.strip():
        return []

    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        if not chunk_words:
            break

        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)

        # Move forward with overlap
        i += max(1, chunk_size - overlap)

    return chunks

# HuggingFace API functions
def query_huggingface_api(model_name: str, payload: dict, max_retries: int = 3) -> dict:
    """Query HuggingFace API with retry logic."""
    if not HF_API_KEY:
        st.warning("⚠️ HuggingFace API key not found. Some features will be limited.")
        return {"error": "No API key"}

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    url = f"{HF_API_URL}{model_name}"

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                # Model is loading, wait and retry
                if attempt < max_retries - 1:
                    st.info(f"🔄 Model loading... Retrying in {2 ** attempt} seconds")
                    time.sleep(2 ** attempt)
                    continue
            else:
                return {"error": f"API Error: {response.status_code}"}

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                st.warning(f"⚠️ Request failed, retrying... ({attempt + 1}/{max_retries})")
                time.sleep(1)
            else:
                return {"error": f"Request failed: {str(e)}"}

    return {"error": "Max retries reached"}

def summarize_with_hf(text: str, max_lines: int = 5) -> str:
    """Summarize text using HuggingFace API with line control."""
    if not text.strip():
        return "No text to summarize"

    # If no API key, provide basic extractive summary
    if not HF_API_KEY:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        return '. '.join(sentences[:max_lines]) + '.' if sentences else "Unable to generate summary"

    # Calculate approximate max_length based on desired lines
    max_length = max_lines * 18
    min_length = max(10, max_lines * 8)

    payload = {
        "inputs": text[:4000],  # Limit input length
        "parameters": {
            "max_length": max_length,
            "min_length": min_length,
            "do_sample": False,
            "early_stopping": True
        }
    }

    result = query_huggingface_api("facebook/bart-large-cnn", payload)

    if result and isinstance(result, list) and len(result) > 0:
        summary = result[0].get("summary_text", "")
        # Ensure we don't exceed the requested line count
        lines = summary.split('. ')
        if len(lines) > max_lines:
            summary = '. '.join(lines[:max_lines]) + '.'
        return summary

    return "❌ Failed to generate summary"

def answer_question_with_hf(question: str, context: str) -> str:
    """Answer question using HuggingFace Q&A model."""
    if not question.strip() or not context.strip():
        return "Please provide both a question and context"

    # If no API key, provide basic search-based answer
    if not HF_API_KEY:
        # Simple keyword matching fallback
        question_words = question.lower().split()
        context_sentences = re.split(r'[.!?]+', context)
        best_sentence = ""
        max_matches = 0
        
        for sentence in context_sentences:
            matches = sum(1 for word in question_words if word in sentence.lower())
            if matches > max_matches:
                max_matches = matches
                best_sentence = sentence.strip()
        
        return best_sentence if best_sentence else "No relevant answer found in context"

    payload = {
        "inputs": {
            "question": question,
            "context": context[:4000]  # Limit context length
        }
    }

    result = query_huggingface_api("deepset/roberta-base-squad2", payload)

    if result and "answer" in result:
        confidence = result.get("score", 0)
        answer = result["answer"]

        if confidence > 0.1:  # Minimum confidence threshold
            return f"{answer}\n\n*Confidence: {confidence:.2%}*"
        else:
            return "❓ I'm not confident about this answer based on the provided context."

    return "❌ Failed to generate answer"

def build_tfidf_index():
    """Build TF-IDF search index for document chunks."""
    texts = [c["text"] for c in st.session_state.kb]
    if not texts:
        st.warning("⚠️ No document chunks to index.")
        return False

    if not HAS_SK:
        st.error("❌ scikit-learn not installed. Run: pip install scikit-learn")
        return False

    try:
        vectorizer = TfidfVectorizer(
            max_features=50000,
            stop_words="english",
            ngram_range=(1, 2),  # Include bigrams
            max_df=0.95,
            min_df=2
        )
        doc_matrix = vectorizer.fit_transform(texts)

        st.session_state.vectorizer = vectorizer
        st.session_state.doc_matrix = doc_matrix

        st.success(f"✅ Search index built successfully! ({len(texts)} chunks indexed)")
        return True

    except Exception as e:
        st.error(f"❌ Failed to build search index: {str(e)}")
        return False

def tfidf_search(query: str, top_k: int = 5) -> List[tuple]:
    """Search document chunks using TF-IDF similarity."""
    if not st.session_state.vectorizer or st.session_state.doc_matrix is None:
        return []

    try:
        query_vector = st.session_state.vectorizer.transform([query])
        similarities = cosine_similarity(st.session_state.doc_matrix, query_vector).ravel()

        # Get top-k most similar chunks
        top_indices = similarities.argsort()[-top_k:][::-1]

        return [(int(idx), float(similarities[idx])) for idx in top_indices if similarities[idx] > 0.01]

    except Exception as e:
        st.error(f"❌ Search failed: {str(e)}")
        return []

def extract_key_phrases(text: str, max_phrases: int = 5) -> List[str]:
    """Extract key phrases from text using simple TF-IDF."""
    if not HAS_SK or not text.strip():
        return []

    try:
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(sentences) < 2:
            return []

        # Use TF-IDF to find important phrases
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(2, 4),
            max_df=0.8,
            min_df=1
        )

        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()

        # Get average TF-IDF scores
        mean_scores = tfidf_matrix.mean(axis=0).A1
        phrase_scores = list(zip(feature_names, mean_scores))
        phrase_scores.sort(key=lambda x: x[1], reverse=True)

        return [phrase for phrase, score in phrase_scores[:max_phrases] if score > 0]

    except Exception:
        return []

# Voice recognition functions
def record_voice_input() -> Optional[str]:
    """Record voice input and convert to text."""
    if not HAS_SPEECH:
        st.error("❌ Speech recognition not available. Install: pip install speechrecognition pyaudio")
        return None

    try:
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()

        # Adjust for ambient noise
        with microphone as source:
            st.info("🎤 Adjusting for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source, duration=1)

        # Record audio
        with microphone as source:
            st.info("🎤 Listening... Speak your question now!")
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=30)

        # Convert speech to text
        st.info("🔄 Processing your speech...")
        text = recognizer.recognize_google(audio)

        return text

    except sr.WaitTimeoutError:
        st.warning("⏰ No speech detected. Please try again.")
        return None
    except sr.UnknownValueError:
        st.warning("❓ Could not understand the speech. Please try again.")
        return None
    except sr.RequestError as e:
        st.error(f"❌ Speech recognition service error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Voice recording failed: {str(e)}")
        return None

# Main application header
st.markdown("""
<div class="main-header">
    <h1>🧠 QueryNest</h1>
    <p>Your AI-Powered StudyMate for PDF Analysis</p>
</div>
""", unsafe_allow_html=True)

# Navigation - Fixed the navigation logic
nav_options = ["🏠 Home", "📄 Summarize", "❓ Q&A", "📝 Notes", "📚 History"]

# Use columns for better navigation layout
cols = st.columns(len(nav_options))
selected_nav = None

for i, option in enumerate(nav_options):
    with cols[i]:
        if st.button(option, key=f"nav_{i}", use_container_width=True):
            selected_nav = option
            st.session_state.current_nav = option.split(" ", 1)[1] if " " in option else option

# Use session state to maintain navigation
if selected_nav is None:
    nav = st.session_state.current_nav
else:
    nav = selected_nav.split(" ", 1)[1] if " " in selected_nav else selected_nav

# 🏠 HOME PAGE
if nav == "Home":
    st.markdown('<div class="content-card">', unsafe_allow_html=True)

    # Statistics Dashboard
    if st.session_state.kb:
        total_chunks = len(st.session_state.kb)
        total_files = len(set(chunk["filename"] for chunk in st.session_state.kb))
        total_pages = len(set((chunk["filename"], chunk["page"]) for chunk in st.session_state.kb))

        st.markdown(f"""
        <div class="stats-container">
            <div class="stat-card">
                <div class="stat-number">{total_files}</div>
                <div class="stat-label">PDF Files</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{total_pages}</div>
                <div class="stat-label">Pages Processed</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{total_chunks}</div>
                <div class="stat-label">Text Chunks</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # PDF Upload Section
    st.markdown("### 📄 Upload PDF Documents")
    st.markdown("Upload one or more PDF files to build your knowledge base for analysis and Q&A.")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDF files to upload and process"
    )

    if uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()

        new_chunks = 0
        processed_files = 0

        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")

            try:
                # Read file bytes
                file_bytes = uploaded_file.read()

                # Extract pages
                pages = extract_pages_from_pdf_bytes(file_bytes)

                if pages:
                    # Process each page
                    for page_data in pages:
                        chunks = chunk_text(page_data["text"])

                        for chunk in chunks:
                            if chunk.strip():  # Only add non-empty chunks
                                chunk_data = {
                                    "id": str(uuid.uuid4()),
                                    "filename": uploaded_file.name,
                                    "page": page_data["page"],
                                    "text": chunk,
                                    "word_count": len(chunk.split()),
                                    "upload_time": datetime.datetime.now().isoformat()
                                }
                                st.session_state.kb.append(chunk_data)
                                new_chunks += 1

                    processed_files += 1
                    st.success(f"✅ {uploaded_file.name}: {len(pages)} pages")
                else:
                    st.warning(f"⚠️ {uploaded_file.name}: No text content found")

            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")

            # Update progress
            progress_bar.progress((i + 1) / len(uploaded_files))

        status_text.text("Processing complete!")

        if new_chunks > 0:
            st.markdown(f"""
            <div class="success-message">
                <strong>Upload Complete!</strong><br>
                📁 Files processed: {processed_files}<br>
                📄 New chunks added: {new_chunks}<br>
                📚 Total chunks in knowledge base: {len(st.session_state.kb)}
            </div>
            """, unsafe_allow_html=True)

    # Search Index Section
    st.markdown("---")
    st.markdown("### 🔍 Search Index")

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.session_state.kb:
            st.markdown("Build a search index to enable fast document retrieval for Q&A.")
            index_status = "✅ Ready" if st.session_state.vectorizer else "❌ Not built"
            st.markdown(f"**Index Status:** {index_status}")
        else:
            st.markdown("Upload PDF files first to build the search index.")

    with col2:
        if st.session_state.kb:
            if st.button("🔨 Build Search Index", type="primary"):
                with st.spinner("Building search index..."):
                    success = build_tfidf_index()
                    if success:
                        st.rerun()
        else:
            st.button("🔨 Build Search Index", disabled=True)

    st.markdown('</div>', unsafe_allow_html=True)

# 📄 SUMMARIZE PAGE
elif nav == "Summarize":
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### 📄 Document Summarization")

    if not st.session_state.kb:
        st.markdown("""
        <div class="warning-message">
            <strong>No documents found!</strong><br>
            Please upload PDF files on the Home page first.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Summarization options
        col1, col2 = st.columns([2, 1])

        with col1:
            scope = st.radio(
                "📋 Summarization Scope",
                ["All documents", "Selected file"],
                help="Choose whether to summarize all documents together or a specific file"
            )

        with col2:
            max_lines = st.slider(
                "📏 Summary Length (lines)",
                min_value=2,
                max_value=15,
                value=5,
                help="Control the length of the generated summary"
            )

        # File selection for individual file summarization
        if scope == "Selected file":
            available_files = sorted(set(chunk["filename"] for chunk in st.session_state.kb))
            selected_file = st.selectbox("Choose a file to summarize:", available_files)

        # Generate summary button
        if st.button("🚀 Generate Summary", type="primary"):
            with st.spinner("Generating summary..."):

                if scope == "All documents":
                    # Combine all document text
                    all_text = " ".join([chunk["text"] for chunk in st.session_state.kb])

                    # Limit text length for API
                    if len(all_text) > 8000:
                        all_text = all_text[:8000] + "..."
                        st.info("📝 Text truncated to fit processing limits")

                    summary = summarize_with_hf(all_text, max_lines)

                    st.markdown("#### 📋 Summary of All Documents")
                    st.markdown(f"""
                    <div class="feature-card">
                        <strong>📊 Document Statistics:</strong><br>
                        • Total files: {len(set(chunk['filename'] for chunk in st.session_state.kb))}<br>
                        • Total chunks: {len(st.session_state.kb)}<br>
                        • Summary length: {max_lines} lines<br><br>
                        <strong>📄 Summary:</strong><br>
                        {summary}
                    </div>
                    """, unsafe_allow_html=True)

                else:  # Selected file
                    file_chunks = [chunk["text"] for chunk in st.session_state.kb if chunk["filename"] == selected_file]
                    file_text = " ".join(file_chunks)

                    # Limit text length
                    if len(file_text) > 6000:
                        file_text = file_text[:6000] + "..."

                    summary = summarize_with_hf(file_text, max_lines)

                    st.markdown(f"#### 📄 {selected_file}")
                    st.markdown(f"""
                    <div class="feature-card">
                        <strong>📊 File Statistics:</strong><br>
                        • Chunks: {len(file_chunks)}<br>
                        • Pages: {len(set(chunk['page'] for chunk in st.session_state.kb if chunk['filename'] == selected_file))}<br><br>
                        <strong>📄 Summary:</strong><br>
                        {summary}
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ❓ Q&A PAGE
elif nav == "Q&A":
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### ❓ Ask Questions About Your Documents")

    if not st.session_state.kb:
        st.markdown("""
        <div class="warning-message">
            <strong>No documents found!</strong><br>
            Please upload PDF files on the Home page first.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Question input
        st.markdown("#### 💬 Ask Your Question")
        
        question = st.text_area(
            "Type your question here:",
            height=100,
            placeholder="e.g., What are the main findings of this research?",
            help="Ask any question about your uploaded documents"
        )

        col1, col2 = st.columns([2, 1])
        
        with col2:
            top_k = st.slider("📊 Results to retrieve", 1, 10, 5, help="Number of relevant chunks to find")

        # Answer generation
        if st.button("🚀 Get Answer", type="primary", disabled=not question.strip()):
            if question.strip():
                with st.spinner("🔍 Searching documents and generating answer..."):

                    # If we have search index, use it
                    if st.session_state.vectorizer:
                        search_results = tfidf_search(question, top_k)
                        
                        if search_results:
                            # Prepare context from retrieved chunks
                            retrieved_chunks = []
                            context_parts = []

                            for idx, score in search_results:
                                if score > 0.01:  # Minimum relevance threshold
                                    chunk = st.session_state.kb[idx]
                                    retrieved_chunks.append({
                                        "score": score,
                                        "filename": chunk["filename"],
                                        "page": chunk["page"],
                                        "text": chunk["text"]
                                    })
                                    context_parts.append(chunk["text"])

                            if context_parts:
                                # Combine context
                                context = " ".join(context_parts)

                                # Generate answer
                                answer = answer_question_with_hf(question, context)

                                # Display answer
                                st.markdown("#### 🎯 Answer")
                                st.markdown(f"""
                                <div class="feature-card">
                                    <strong>❓ Question:</strong> {question}<br><br>
                                    <strong>🎯 Answer:</strong><br>
                                    {answer}
                                </div>
                                """, unsafe_allow_html=True)

                                # Display sources
                                st.markdown("#### 📚 Sources")
                                for i, chunk in enumerate(retrieved_chunks[:3], 1):
                                    relevance_bar = "🟢" * int(chunk["score"] * 10) + "⚪" * (10 - int(chunk["score"] * 10))
                                    st.markdown(f"""
                                    <div class="feature-card">
                                        <strong>📄 Source {i}:</strong> {chunk['filename']} (Page {chunk['page']})<br>
                                        <strong>🎯 Relevance:</strong> {relevance_bar} ({chunk['score']:.1%})<br><br>
                                        <em>{chunk['text'][:300]}{'...' if len(chunk['text']) > 300 else ''}</em>
                                    </div>
                                    """, unsafe_allow_html=True)

                                # Save to history
                                history_entry = {
                                    "id": str(uuid.uuid4()),
                                    "question": question,
                                    "answer": answer,
                                    "sources": retrieved_chunks,
                                    "timestamp": datetime.datetime.now().isoformat(),
                                    "method": "text"
                                }
                                st.session_state.history.insert(0, history_entry)

                            else:
                                st.warning("❓ No relevant information found for your question.")
                        else:
                            st.warning("❓ No relevant documents found. Try rephrasing your question.")
                    else:
                        # Fallback: search without index
                        st.warning("⚠️ Search index not built. Using basic search...")
                        all_text = " ".join([chunk["text"] for chunk in st.session_state.kb])
                        answer = answer_question_with_hf(question, all_text[:4000])
                        
                        st.markdown("#### 🎯 Answer")
                        st.markdown(f"""
                        <div class="feature-card">
                            <strong>❓ Question:</strong> {question}<br><br>
                            <strong>🎯 Answer:</strong><br>
                            {answer}
                        </div>
                        """, unsafe_allow_html=True)

        # Build index prompt
        if not st.session_state.vectorizer:
            st.markdown("---")
            st.info("💡 **Tip:** Build a search index on the Home page for better question answering!")

    st.markdown('</div>', unsafe_allow_html=True)

# 📝 NOTES PAGE
elif nav == "Notes":
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### 📝 Personal Notes")

    # Note creation section
    st.markdown("#### ✏️ Create New Note")

    with st.form("note_form", clear_on_submit=True):
        col1, col2 = st.columns([3, 1])

        with col1:
            note_title = st.text_input(
                "Note Title",
                placeholder="Enter a title for your note...",
                help="Give your note a descriptive title"
            )

        with col2:
            note_category = st.selectbox(
                "Category",
                ["General", "Summary", "Question", "Important", "Research", "Todo"],
                help="Categorize your note"
            )

        note_content = st.text_area(
            "Note Content",
            height=150,
            placeholder="Write your note here...",
            help="Add your note content, thoughts, or observations"
        )

        # Tags input
        note_tags = st.text_input(
            "Tags (comma-separated)",
            placeholder="e.g., research, important, follow-up",
            help="Add tags to organize your notes"
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            save_note = st.form_submit_button("💾 Save Note", type="primary")

        with col2:
            clear_form = st.form_submit_button("🗑️ Clear")

    if save_note and (note_title.strip() or note_content.strip()):
        # Process tags
        tags = [tag.strip() for tag in note_tags.split(",") if tag.strip()] if note_tags else []

        new_note = {
            "id": str(uuid.uuid4()),
            "title": note_title.strip() or "Untitled Note",
            "content": note_content.strip(),
            "category": note_category,
            "tags": tags,
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat()
        }

        st.session_state.notes.insert(0, new_note)
        st.success("✅ Note saved successfully!")
        st.rerun()

    elif save_note:
        st.warning("⚠️ Please enter a title or content for your note.")

    # Notes display section
    st.markdown("---")
    st.markdown("#### 📚 Your Notes")

    if not st.session_state.notes:
        st.markdown("""
        <div class="feature-card" style="text-align: center; padding: 3rem;">
            <h3>📝 No notes yet!</h3>
            <p>Create your first note using the form above.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Search and filter options
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            search_query = st.text_input("🔍 Search notes", placeholder="Search by title, content, or tags...")

        with col2:
            category_filter = st.selectbox("Filter by category", ["All"] + ["General", "Summary", "Question", "Important", "Research", "Todo"])

        with col3:
            sort_by = st.selectbox("Sort by", ["Newest first", "Oldest first", "Title A-Z"])

        # Filter and sort notes
        filtered_notes = st.session_state.notes.copy()

        # Apply search filter
        if search_query:
            filtered_notes = [
                note for note in filtered_notes
                if (search_query.lower() in note["title"].lower() or
                    search_query.lower() in note["content"].lower() or
                    any(search_query.lower() in tag.lower() for tag in note.get("tags", [])))
            ]

        # Apply category filter
        if category_filter != "All":
            filtered_notes = [note for note in filtered_notes if note.get("category", "General") == category_filter]

        # Apply sorting
        if sort_by == "Newest first":
            filtered_notes.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        elif sort_by == "Oldest first":
            filtered_notes.sort(key=lambda x: x.get("created_at", ""))
        elif sort_by == "Title A-Z":
            filtered_notes.sort(key=lambda x: x["title"].lower())

        # Display notes
        if filtered_notes:
            st.markdown(f"**Found {len(filtered_notes)} note(s)**")

            for i, note in enumerate(filtered_notes):
                # Note card
                created_date = datetime.datetime.fromisoformat(note.get("created_at", datetime.datetime.now().isoformat())).strftime("%Y-%m-%d %H:%M")

                # Category emoji mapping
                category_emojis = {
                    "General": "📄",
                    "Summary": "📋",
                    "Question": "❓",
                    "Important": "⭐",
                    "Research": "🔬",
                    "Todo": "✅"
                }

                category_emoji = category_emojis.get(note.get("category", "General"), "📄")

                # Tags display
                tags_html = ""
                if note.get("tags"):
                    tags_html = " ".join([f"<span style='background: #e5e7eb; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; margin-right: 4px;'>#{tag}</span>" for tag in note["tags"]])

                with st.expander(f"{category_emoji} {note['title']}", expanded=False):
                    st.markdown(f"**Created:** {created_date} • **Category:** {note.get('category', 'General')}")
                    
                    if tags_html:
                        st.markdown(tags_html, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.markdown(note['content'])
                    
                    # Action buttons
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("✏️ Edit", key=f"edit_{note['id']}"):
                            st.info("Edit functionality would be implemented here")
                    with col2:
                        if st.button("🗑️ Delete", key=f"delete_{note['id']}"):
                            st.session_state.notes = [n for n in st.session_state.notes if n["id"] != note["id"]]
                            st.success("🗑️ Note deleted!")
                            st.rerun()

        else:
            st.info("🔍 No notes match your search criteria.")

    st.markdown('</div>', unsafe_allow_html=True)

# 📚 HISTORY PAGE
elif nav == "History":
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### 📚 Q&A History")

    if not st.session_state.history:
        st.markdown("""
        <div class="feature-card" style="text-align: center; padding: 3rem;">
            <h3>📚 No Q&A history yet!</h3>
            <p>Your question and answer sessions will appear here.</p>
            <p>Go to the Q&A page to start asking questions about your documents.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # History statistics
        total_sessions = len(st.session_state.history)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.metric("📊 Total Q&A Sessions", total_sessions)
        
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.success("🗑️ History cleared!")
                st.rerun()

        # Display history
        st.markdown("---")
        
        for i, session in enumerate(st.session_state.history[:20]):  # Show latest 20
            # Format timestamp
            try:
                timestamp = datetime.datetime.fromisoformat(session.get("timestamp", "")).strftime("%Y-%m-%d %H:%M")
            except:
                timestamp = "Unknown time"

            # Question preview
            question = session.get("question", "")
            question_preview = question[:100] + "..." if len(question) > 100 else question

            with st.expander(f"❓ {i+1}. {question_preview}", expanded=False):
                # Full question
                st.markdown(f"**❓ Question:**")
                st.markdown(f'<div class="feature-card">{question}</div>', unsafe_allow_html=True)

                # Answer
                answer = session.get("answer", "")
                st.markdown(f"**🎯 Answer:**")
                st.markdown(f'<div class="feature-card">{answer}</div>', unsafe_allow_html=True)

                # Sources/References
                sources = session.get("sources", [])
                if sources:
                    st.markdown("**📚 Sources:**")
                    for j, source in enumerate(sources[:3], 1):  # Show top 3 sources
                        filename = source.get("filename", "Unknown file")
                        page = source.get("page", "Unknown page")
                        score = source.get("score", 0)
                        text_preview = source.get("text", "")[:150] + "..." if len(source.get("text", "")) > 150 else source.get("text", "")

                        st.markdown(f"""
                        <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 3px solid var(--accent);">
                            <strong>📄 Source {j}:</strong> {filename} (Page {page})<br>
                            <strong>🎯 Relevance:</strong> {score:.1%}<br><br>
                            <em style="color: var(--muted);">{text_preview}</em>
                        </div>
                        """, unsafe_allow_html=True)

                # Session metadata
                st.markdown("---")
                st.markdown(f"**🕒 Time:** {timestamp} • **📚 Sources:** {len(sources)}")

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--muted); padding: 2rem 0;">
    <p>🧠 <strong>QueryNest StudyMate</strong> - Your AI-Powered PDF Analysis Companion</p>
    <p>Built with ❤️ using Streamlit and HuggingFace Transformers</p>
</div>
""", unsafe_allow_html=True)
