import streamlit as st
import pytesseract
import numpy as np
import faiss
import pickle
import os
from PIL import Image
from PyPDF2 import PdfReader
from io import BytesIO
from langchain_ollama.llms import OllamaLLM
from sentence_transformers import SentenceTransformer

# Initialize session state
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None  # Will hold FAISS index
if 'chunks' not in st.session_state:
    st.session_state.chunks = []  # Store text chunks
if 'embedder' not in st.session_state:
    st.session_state.embedder = SentenceTransformer('all-MiniLM-L6-v2')
if 'processed_text' not in st.session_state:
    st.session_state.processed_text = ""

# Initialize components
text_llm = OllamaLLM(model="gemma3:12b")
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def process_file(uploaded_file):
    """Process PDF or image file"""
    content = ""
    
    if uploaded_file.type == "application/pdf":
        # Extract text from PDF
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            content += page.extract_text() + "\n"
    else:
        # OCR for images
        img = Image.open(uploaded_file)
        content = pytesseract.image_to_string(img)
    
    return content

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def store_in_vector_db(text):
    """Store text chunks with embeddings using FAISS"""
    chunks = chunk_text(text)
    st.session_state.chunks = chunks
    
    # Generate embeddings
    embeddings = st.session_state.embedder.encode(chunks)
    
    # Convert to numpy array
    embeddings = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    st.session_state.vector_db = index
    
    # Save index and chunks
    faiss.write_index(index, "faiss_index.bin")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

def load_vector_db():
    """Load FAISS index and chunks from disk"""
    if os.path.exists("faiss_index.bin") and os.path.exists("chunks.pkl"):
        index = faiss.read_index("faiss_index.bin")
        with open("chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        return index, chunks
    return None, []

def answer_question(question):
    """Retrieve relevant context and generate answer using FAISS"""
    if st.session_state.vector_db is None:
        return "No documents processed yet"
    
    # Generate query embedding
    query_embedding = st.session_state.embedder.encode([question])
    query_embedding = np.array(query_embedding).astype('float32')
    
    # Search FAISS index
    distances, indices = st.session_state.vector_db.search(query_embedding, 3)
    
    # Get relevant chunks
    relevant_chunks = [st.session_state.chunks[i] for i in indices[0]]
    
    # Build context
    context = "\n\n".join(relevant_chunks)
    
    # Generate answer with Gemma
    prompt = f"""Context:
    {context}
    
    Question: {question}
    
    As Medical Assistant . Answer the question using the context above. Be concise and factual.
    If any deficiency are observed .Suggest some Ayurvedic Medicine and thier procedures how to use in the tabular format.
    Tabular format should have Deficiency/Imbabalance , Current Reading ,Expected Reading , Suggested Ayurvedic Intervention,
    Procedure/Dosage, Notes , Online Link.
    If unsure, state what information is missing."""
    
    return text_llm.invoke(prompt)

# Load existing data at startup
if st.session_state.vector_db is None:
    index, chunks = load_vector_db()
    if index:
        st.session_state.vector_db = index
        st.session_state.chunks = chunks

# Streamlit UI
st.title("Document Intelligence System")
st.sidebar.header("File Processing")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Document",
    type=['pdf', 'png', 'jpg', 'jpeg']
)

if uploaded_file:
    if st.sidebar.button("Process Document"):
        with st.spinner("Analyzing document..."):
            # Process file
            text_content = process_file(uploaded_file)
            st.session_state.processed_text = text_content
            
            # Store in vector DB
            store_in_vector_db(text_content)
            
            st.sidebar.success("Document processed and stored!")

# Display processed text
if st.session_state.processed_text:
    st.subheader("Processed Content")
    with st.expander("View raw text"):
        st.text(st.session_state.processed_text)

# Q&A Section
st.header("Document Query")
question = st.text_input("Ask about your document:")

if question and st.session_state.processed_text:
    with st.spinner("Generating answer..."):
        answer = answer_question(question)
        st.markdown(f"**Answer:** {answer}")

# Image preview
if uploaded_file and uploaded_file.type in ['image/png', 'image/jpeg']:
    st.subheader("Uploaded Image Preview")
    img = Image.open(uploaded_file)
    st.image(img, caption=uploaded_file.name)