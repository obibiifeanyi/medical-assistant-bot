"""
Simplified Medical Assistant Streamlit App
This version has better error handling and clearer setup instructions
"""

import streamlit as st
import os
import sys
import pandas as pd
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page config
st.set_page_config(
    page_title="Medical Assistant Bot",
    page_icon="🏥",
    layout="wide"
)

# Title
st.title("🏥 Medical Assistant Chatbot")

# Check if all required files exist
def check_files():
    """Check if all required files are present."""
    required_files = {
        "Python modules": ["medical_tools.py", "medical_agent_langchain.py"],
        "Data files": [
            "data/disease_symptoms.csv",
            "data/disease_symptom_severity.csv", 
            "data/disease_precautions.csv",
            "data/disease_symptom_description.csv"
        ],
        "FAISS indices": [
            "indices/faiss_symptom_index",
            "indices/faiss_severity_index"
        ]
    }
    
    missing_files = {}
    for category, files in required_files.items():
        missing = [f for f in files if not os.path.exists(f)]
        if missing:
            missing_files[category] = missing
    
    return missing_files

# Check files
missing = check_files()
if missing:
    st.error("⚠️ Missing required files!")
    for category, files in missing.items():
        st.write(f"**{category}:**")
        for f in files:
            st.write(f"  - {f}")
    
    st.info("""
    📋 **Setup Instructions:**
    1. Make sure all Python files are uploaded
    2. Create 'data/' folder and upload CSV files
    3. Create 'indices/' folder and upload FAISS index files
    4. Check the deployment guide for detailed instructions
    """)
    st.stop()

# Now try to import modules
try:
    from medical_tools import set_global_resources, check_resources_available
    from medical_agent_langchain import create_medical_assistant
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError as e:
    st.error(f"❌ Error importing modules: {str(e)}")
    st.info("Please check that all dependencies are installed correctly.")
    st.stop()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'assistant' not in st.session_state:
    st.session_state.assistant = None

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    
    # API Key input
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key to use the medical assistant"
    )
    
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
    
    # About section
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.markdown("""
    This AI assistant can help analyze symptoms and provide health information.
    
    **⚠️ Disclaimer:** Not a replacement for professional medical advice.
    """)
    
    # Clear button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        if st.session_state.assistant:
            st.session_state.assistant.reset_conversation()

# Main content
@st.cache_resource
def load_resources():
    """Load all medical resources."""
    try:
        # Load embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Load FAISS indices - try both with and without _medibot suffix
        symptom_index_path = "indices/faiss_symptom_index"
        if os.path.exists("indices/faiss_symptom_index_medibot"):
            symptom_index_path = "indices/faiss_symptom_index_medibot"
            
        severity_index_path = "indices/faiss_severity_index"
        if os.path.exists("indices/faiss_severity_index_medibot"):
            severity_index_path = "indices/faiss_severity_index_medibot"
        
        faiss_symptom = FAISS.load_local(
            symptom_index_path, embeddings,
            allow_dangerous_deserialization=True
        )
        faiss_severity = FAISS.load_local(
            severity_index_path, embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Load CSVs
        df_precautions = pd.read_csv("data/disease_precautions.csv")
        df_descriptions = pd.read_csv("data/disease_symptom_description.csv")
        df_severity = pd.read_csv("data/disease_symptom_severity.csv")
        
        return {
            'symptom_index': faiss_symptom,
            'severity_index': faiss_severity,
            'precautions': df_precautions,
            'descriptions': df_descriptions,
            'severity': df_severity
        }
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        return None

# Initialize assistant
def init_assistant():
    """Initialize the medical assistant."""
    if not os.getenv('OPENAI_API_KEY'):
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar.")
        return False
    
    if st.session_state.assistant is None:
        with st.spinner("Loading medical knowledge base..."):
            resources = load_resources()
            
        if resources:
            try:
                st.session_state.assistant = create_medical_assistant(
                    faiss_symptom_index=resources['symptom_index'],
                    faiss_severity_index=resources['severity_index'],
                    df_disease_precautions=resources['precautions'],
                    df_disease_symptom_description=resources['descriptions'],
                    df_disease_symptom_severity=resources['severity'],
                    model_name="gpt-3.5-turbo",
                    use_memory=True
                )
                return True
            except Exception as e:
                st.error(f"Failed to initialize assistant: {str(e)}")
                return False
    return True

# Chat interface
if init_assistant():
    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Input
    if prompt := st.chat_input("Describe your symptoms..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    response = st.session_state.assistant.chat(prompt)
                    st.write(response)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response
                    })
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
else:
    # Show setup help
    st.info("""
    👋 **Welcome to the Medical Assistant!**
    
    To get started:
    1. Enter your OpenAI API key in the sidebar
    2. Start describing your symptoms
    3. Ask follow-up questions about conditions
    
    **Example questions:**
    - "I have a headache and fever"
    - "What are the symptoms of malaria?"
    - "I have itchy skin with rashes"
    """)

# Footer
st.markdown("---")
st.caption("⚠️ This tool provides general information only. Always consult healthcare professionals for medical advice.")