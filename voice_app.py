"""Voice-Based Medical Assistant with Animated Avatar Interface"""

import streamlit as st
import time
import base64
from datetime import datetime, timedelta
import sys
import os
from PIL import Image
import io
import json
from typing import Optional
import pandas as pd

# Add current directory and src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import medical modules
try:
    from medical_agent_langchain import create_medical_assistant
    from medical_tools import get_enhanced_medical_tools
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    
    MEDICAL_MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import medical modules: {e}")
    MEDICAL_MODULES_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Voice Medical Assistant",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize session state variables first
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'call_active' not in st.session_state:
    st.session_state.call_active = False
if 'speaking' not in st.session_state:
    st.session_state.speaking = False
if 'call_start_time' not in st.session_state:
    st.session_state.call_start_time = None
if 'voice_messages' not in st.session_state:
    st.session_state.voice_messages = []
if 'uploaded_audio' not in st.session_state:
    st.session_state.uploaded_audio = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'image_analysis' not in st.session_state:
    st.session_state.image_analysis = None
if 'assistant' not in st.session_state:
    st.session_state.assistant = None
if 'resources_loaded' not in st.session_state:
    st.session_state.resources_loaded = False
if 'is_speaking' not in st.session_state:
    st.session_state.is_speaking = False
if 'in_call' not in st.session_state:
    st.session_state.in_call = False
if 'call_duration' not in st.session_state:
    st.session_state.call_duration = 0

# iPhone Caller UI Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');
    
    /* Hide Streamlit default elements */
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stDecoration {display:none;}
    header[data-testid="stHeader"] {display: none;}
    
    /* iOS-style variables for iPhone caller UI */
    :root {
        --bg-primary: #000000;
        --bg-secondary: #1c1c1e;
        --bg-tertiary: #2c2c2e;
        --accent-green: #30d158;
        --accent-red: #ff453a;
        --accent-blue: #007aff;
        --text-primary: #ffffff;
        --text-secondary: #8e8e93;
        --text-muted: #636366;
        --border-color: rgba(255, 255, 255, 0.08);
        --shadow-light: rgba(255, 255, 255, 0.05);
        --shadow-dark: rgba(0, 0, 0, 0.3);
        --glass-bg: rgba(255, 255, 255, 0.03);
        --glass-border: rgba(255, 255, 255, 0.06);
        --ios-blur: blur(20px);
        --shadow-glow: 0 0 30px rgba(48, 209, 88, 0.3);
        --shadow-red-glow: 0 0 30px rgba(255, 69, 58, 0.3);
    }
    
    /* Main app styling with iOS design language */
    .stApp {
        background: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        line-height: 1.47;
    }
    
    /* Global iOS-style body styling */
    body {
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
        color: var(--text-primary);
        line-height: 1.47;
        overflow-x: hidden;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    .main .block-container {
        padding: 2rem 1.5rem 4rem 1.5rem;
        max-width: 100%;
        background: transparent;
    }
    
    /* iPhone-style Main Container */
    .main-container {
        background: linear-gradient(135deg, var(--bg-primary), var(--bg-secondary));
        backdrop-filter: blur(30px);
        -webkit-backdrop-filter: blur(30px);
        border: 1px solid var(--glass-border);
        border-radius: 30px;
        padding: 2.5rem 2rem;
        margin: 1.5rem auto;
        max-width: 420px;
        box-shadow: 
            0 10px 40px rgba(0, 0, 0, 0.25),
            0 4px 12px rgba(0, 0, 0, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            inset 0 -1px 0 rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        animation: ios-fade-in 0.6s ease-out;
    }
    
    .main-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 0.5px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
    }
    
    /* iPhone-style Typography */
    h1, h2, h3 {
        color: var(--text-primary);
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 600;
        margin-bottom: 1.5rem;
        letter-spacing: -0.02em;
        text-align: center;
    }
    
    h1 {
        font-size: 1.75rem;
        font-weight: 700;
    }
    
    h2 {
        font-size: 1.25rem;
        font-weight: 600;
    }
    
    p, .stMarkdown {
        color: var(--text-secondary);
        font-family: 'SF Pro Text', -apple-system, BlinkMacSystemFont, sans-serif;
        line-height: 1.5;
        margin-bottom: 1rem;
        font-size: 0.95rem;
        text-align: center;
    }
    
    /* Main container */
    .voice-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 80vh;
        background: var(--bg-primary);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
    }
    
    /* iPhone-style Avatar Circle Container */
    .avatar-container {
        position: relative;
        width: 240px;
        height: 240px;
        margin: 3rem 0;
        display: flex;
        justify-content: center;
        align-items: center;
        animation: ios-scale-in 0.8s ease-out 0.2s both;
    }
    
    /* iPhone Caller UI Avatar Circle */
     .avatar-circle {
         width: 100%;
         height: 100%;
         border-radius: 50%;
         background: linear-gradient(135deg, var(--bg-tertiary) 0%, var(--bg-secondary) 100%);
         display: flex;
         align-items: center;
         justify-content: center;
         font-size: 3.5rem;
         color: var(--text-primary);
         border: 2px solid var(--glass-border);
         transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
         cursor: pointer;
         position: relative;
         overflow: hidden;
         box-shadow: 
             0 8px 32px rgba(0, 0, 0, 0.4),
             0 2px 8px rgba(0, 0, 0, 0.2),
             inset 0 1px 0 rgba(255, 255, 255, 0.1);
     }
     
     /* Rotating border animation */
     .avatar-circle::before {
         content: '';
         position: absolute;
         top: -50%;
         left: -50%;
         width: 200%;
         height: 200%;
         background: conic-gradient(
             from 0deg,
             transparent 0deg,
             var(--accent-green) 60deg,
             transparent 120deg,
             var(--accent-green) 180deg,
             transparent 240deg,
             var(--accent-green) 300deg,
             transparent 360deg
         );
         animation: rotate 4s linear infinite;
         opacity: 0;
         transition: opacity 0.5s ease;
     }
     
     /* Inner circle mask */
     .avatar-circle::after {
         content: '';
         position: absolute;
         inset: 4px;
         background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
         border-radius: 50%;
         z-index: 1;
         box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.3);
     }
     
     /* Avatar content container */
     .avatar-content {
         position: relative;
         z-index: 2;
         display: flex;
         align-items: center;
         justify-content: center;
         width: 100%;
         height: 100%;
         border-radius: 50%;
         transition: all 0.3s ease;
     }
    
    /* Animated ring for speaking state */
    .speaking-ring {
        position: absolute;
        top: -10px;
        left: -10px;
        width: calc(100% + 20px);
        height: calc(100% + 20px);
        border-radius: 50%;
        border: 3px solid transparent;
        background: conic-gradient(from 0deg, transparent, var(--accent-red), transparent);
        animation: rotate 2s linear infinite;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .speaking .speaking-ring {
        opacity: 1;
    }
    
    /* iPhone-style Speaking state animations */
     .speaking .avatar-circle {
         background: linear-gradient(135deg, var(--accent-green), #28cd41);
         box-shadow: 
             0 0 0 4px rgba(48, 209, 88, 0.2),
             0 0 0 8px rgba(48, 209, 88, 0.1),
             0 8px 32px rgba(48, 209, 88, 0.3),
             0 2px 8px rgba(0, 0, 0, 0.2),
             inset 0 1px 0 rgba(255, 255, 255, 0.2);
         animation: ios-speaking-pulse 1.2s ease-in-out infinite;
         border-color: var(--accent-green);
     }
     
     .speaking .avatar-circle::before {
         background: conic-gradient(
             from 0deg,
             transparent 0deg,
             var(--accent-green) 45deg,
             transparent 90deg,
             var(--accent-green) 135deg,
             transparent 180deg,
             var(--accent-green) 225deg,
             transparent 270deg,
             var(--accent-green) 315deg,
             transparent 360deg
         );
         opacity: 1;
         animation: rotate 2s linear infinite;
     }
     
     .speaking .avatar-circle::after {
         background: linear-gradient(135deg, var(--bg-primary) 0%, #001a00 100%);
         box-shadow: inset 0 2px 10px rgba(48, 209, 88, 0.2);
     }
     
     /* iPhone-style Listening state animations */
     .listening .avatar-circle {
         background: linear-gradient(135deg, var(--accent-red), #ff6b47);
         box-shadow: 
             0 0 0 4px rgba(255, 69, 58, 0.2),
             0 0 0 8px rgba(255, 69, 58, 0.1),
             0 8px 32px rgba(255, 69, 58, 0.3),
             0 2px 8px rgba(0, 0, 0, 0.2),
             inset 0 1px 0 rgba(255, 255, 255, 0.2);
         animation: ios-listening-pulse 1.8s ease-in-out infinite;
         border-color: var(--accent-red);
     }
     
     .listening .avatar-circle::before {
         background: conic-gradient(
             from 0deg,
             transparent 0deg,
             var(--accent-red) 30deg,
             transparent 60deg,
             var(--accent-red) 120deg,
             transparent 150deg,
             var(--accent-red) 210deg,
             transparent 240deg,
             var(--accent-red) 300deg,
             transparent 330deg
         );
         opacity: 1;
         animation: rotate 3s linear infinite;
     }
     
     .listening .avatar-circle::after {
         background: linear-gradient(135deg, var(--bg-primary) 0%, #1a0000 100%);
         box-shadow: inset 0 2px 10px rgba(255, 69, 58, 0.2);
     }
     
     /* Enhanced keyframe animations */
     @keyframes rotate {
         from { transform: rotate(0deg); }
         to { transform: rotate(360deg); }
     }
     
     @keyframes ios-speaking-pulse {
         0%, 100% { 
             transform: scale(1);
             box-shadow: 
                 0 0 0 4px rgba(48, 209, 88, 0.2),
                 0 0 0 8px rgba(48, 209, 88, 0.1),
                 0 8px 32px rgba(48, 209, 88, 0.3),
                 0 2px 8px rgba(0, 0, 0, 0.2);
         }
         50% { 
             transform: scale(1.04);
             box-shadow: 
                 0 0 0 6px rgba(48, 209, 88, 0.3),
                 0 0 0 12px rgba(48, 209, 88, 0.15),
                 0 12px 40px rgba(48, 209, 88, 0.4),
                 0 4px 12px rgba(0, 0, 0, 0.3);
         }
     }
     
     @keyframes ios-listening-pulse {
         0%, 100% { 
             transform: scale(1);
             box-shadow: 
                 0 0 0 4px rgba(255, 69, 58, 0.2),
                 0 0 0 8px rgba(255, 69, 58, 0.1),
                 0 8px 32px rgba(255, 69, 58, 0.3),
                 0 2px 8px rgba(0, 0, 0, 0.2);
         }
         50% { 
             transform: scale(1.02);
             box-shadow: 
                 0 0 0 6px rgba(255, 69, 58, 0.3),
                 0 0 0 12px rgba(255, 69, 58, 0.15),
                 0 10px 36px rgba(255, 69, 58, 0.4),
                 0 4px 12px rgba(0, 0, 0, 0.3);
         }
     }
     
     /* iPhone-style Entrance Animations */
     @keyframes ios-fade-in {
         from { 
             opacity: 0; 
             transform: translateY(30px) scale(0.95); 
         }
         to { 
             opacity: 1; 
             transform: translateY(0) scale(1); 
         }
     }
     
     @keyframes ios-slide-up {
         from {
             opacity: 0;
             transform: translateY(50px);
         }
         to {
             opacity: 1;
             transform: translateY(0);
         }
     }
     
     @keyframes ios-scale-in {
         from {
             opacity: 0;
             transform: scale(0.8);
         }
         to {
             opacity: 1;
             transform: scale(1);
         }
     }
    
    /* Pulse animation */
    @keyframes pulse {
        0% { transform: scale(1); opacity: 0.7; }
        50% { transform: scale(1.1); opacity: 1; }
        100% { transform: scale(1); opacity: 0.7; }
    }
    
    /* Phone controls container */
    .phone-controls {
        display: flex;
        justify-content: space-around;
        align-items: center;
        width: 100%;
        max-width: 400px;
        margin: 2rem 0;
    }
    
    /* iPhone-style control buttons */
    .control-btn {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        border: none;
        color: white;
        font-size: 1.5rem;
        cursor: pointer;
        transition: all 0.2s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        overflow: hidden;
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    
    /* iPhone-style Button Styling for Streamlit buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary));
        border: 1px solid var(--glass-border);
        border-radius: 25px;
        color: var(--text-primary);
        font-weight: 600;
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        padding: 0.875rem 2rem;
        transition: all 0.2s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        backdrop-filter: blur(20px);
        box-shadow: 
            0 2px 10px rgba(0, 0, 0, 0.15),
            0 1px 3px rgba(0, 0, 0, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        font-size: 1rem;
        min-height: 50px;
        letter-spacing: -0.01em;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--bg-tertiary), var(--bg-secondary));
        transform: translateY(-1px);
        box-shadow: 
            0 4px 20px rgba(0, 0, 0, 0.25),
            0 2px 6px rgba(0, 0, 0, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
        box-shadow: 
            0 1px 5px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }
    
    .control-btn::before {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 50%;
        padding: 2px;
        background: linear-gradient(135deg, var(--accent-green), #00e676);
        mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        mask-composite: xor;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .dial-btn {
        background: linear-gradient(135deg, var(--accent-green), #00e676);
        color: var(--bg-primary);
        box-shadow: 0 10px 30px rgba(0, 255, 136, 0.3);
    }
    
    .end-call-btn {
        background: linear-gradient(135deg, var(--accent-red), #ff3742);
        color: white;
        box-shadow: 0 10px 30px rgba(255, 71, 87, 0.3);
    }
    
    .end-call-btn::before {
        background: linear-gradient(135deg, var(--accent-red), #ff3742);
    }
    
    .upload-btn {
        background: linear-gradient(45deg, #2196F3, #1976D2);
    }
    
    .control-btn:hover {
        transform: translateY(-2px) scale(1.02);
    }
    
    .control-btn:hover::before {
        opacity: 1;
    }
    
    .control-btn:active {
        transform: translateY(0px) scale(0.98);
    }
    
    /* Enhanced call timer */
    .call-timer {
        font-size: 1.5rem;
        color: var(--text-primary);
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 1rem 0;
        font-weight: 300;
        letter-spacing: 0.1em;
        font-variant-numeric: tabular-nums;
    }
    
    /* Status display */
    .status-display {
        color: var(--text-primary);
        font-size: 1.1rem;
        text-align: center;
        margin: 1rem 0;
        min-height: 2rem;
    }
    
    /* Enhanced conversation history */
    .conversation-history {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 2rem 0;
        max-height: 400px;
        overflow-y: auto;
        width: 100%;
        max-width: 500px;
        animation: ios-slide-up 0.5s ease-out 0.4s both;
    }
    
    .conversation-history::-webkit-scrollbar {
        width: 6px;
    }
    
    .conversation-history::-webkit-scrollbar-track {
        background: var(--bg-secondary);
        border-radius: 3px;
    }
    
    .conversation-history::-webkit-scrollbar-thumb {
        background: var(--accent-green);
        border-radius: 3px;
    }
    
    .conversation-item {
        margin: 1rem 0;
        padding: 1rem 1.5rem;
        border-radius: 16px;
        max-width: 85%;
        position: relative;
        backdrop-filter: blur(10px);
        animation: ios-fade-in 0.4s ease-out;
        transform-origin: center;
    }
    
    .user-message {
        background: linear-gradient(135deg, var(--accent-green), #00e676);
        color: var(--bg-primary);
        margin-left: auto;
        text-align: right;
        box-shadow: 0 4px 20px rgba(0, 255, 136, 0.2);
    }
    
    .assistant-message {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        color: var(--text-primary);
        margin-right: auto;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    /* iPhone-style File Upload Styling */
    .stFileUploader {
        background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary));
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.2s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        backdrop-filter: blur(20px);
        box-shadow: 
            0 4px 20px rgba(0, 0, 0, 0.15),
            0 1px 3px rgba(0, 0, 0, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stFileUploader:hover {
        background: linear-gradient(135deg, var(--bg-tertiary), var(--bg-secondary));
        transform: translateY(-1px);
        box-shadow: 
            0 6px 25px rgba(0, 0, 0, 0.2),
            0 2px 6px rgba(0, 0, 0, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
    }
    
    .stFileUploader label {
        color: var(--text-primary) !important;
        font-weight: 500;
        font-size: 1rem;
    }
    
    .stFileUploader [data-testid="stFileUploaderDropzone"] {
        background: transparent;
        border: 2px dashed var(--glass-border);
        border-radius: 16px;
        padding: 2rem;
        transition: all 0.2s ease;
    }
    
    .stFileUploader [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--accent-blue);
        background: rgba(0, 122, 255, 0.05);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online {
        background: var(--accent-green);
        box-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    }
    
    .status-speaking {
        background: var(--accent-red);
        box-shadow: 0 0 10px rgba(255, 71, 87, 0.5);
        animation: pulse-indicator 1s infinite;
    }
    
    @keyframes pulse-indicator {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
</style>
""", unsafe_allow_html=True)

# Session state already initialized above

@st.cache_resource
def load_resources():
    """Load medical resources with caching"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

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

        df_precautions = pd.read_csv("data/disease_precautions.csv")
        df_descriptions = pd.read_csv("data/disease_symptom_description.csv")
        df_severity = pd.read_csv("data/disease_symptom_severity.csv")

        vision_model = None
        if os.getenv('OPENAI_API_KEY'):
            try:
                vision_model = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0.1,
                    max_tokens=4000
                )
            except Exception:
                try:
                    vision_model = ChatOpenAI(
                        model="gpt-4-turbo",
                        temperature=0.1,
                        max_tokens=4000
                    )
                except Exception:
                    pass

        return {
            'symptom_index': faiss_symptom,
            'severity_index': faiss_severity,
            'precautions': df_precautions,
            'descriptions': df_descriptions,
            'severity': df_severity,
            'vision_model': vision_model
        }
    except Exception as e:
        st.error(f"Failed to load medical resources: {e}")
        return None

def init_assistant():
    """Initialize the medical assistant"""
    if not st.session_state.resources_loaded:
        with st.spinner("Loading medical knowledge base..."):
            resources = load_resources()
            if resources:
                st.session_state.assistant = create_medical_assistant(
                    faiss_symptom_index=resources['symptom_index'],
                    faiss_severity_index=resources['severity_index'],
                    df_disease_precautions=resources['precautions'],
                    df_disease_symptom_description=resources['descriptions'],
                    df_disease_symptom_severity=resources['severity'],
                    vision_model=resources['vision_model'],
                    model_name="gpt-4o",
                    use_memory=True
                )
                st.session_state.resources_loaded = True
            else:
                st.error("Failed to load medical resources")
                return False
    return True

def get_medical_response(user_input):
    """Get response from medical assistant"""
    if not st.session_state.assistant:
        if not init_assistant():
            return "I'm sorry, I'm having trouble accessing my medical knowledge base. Please try again later."
    
    st.session_state.speaking = True
    
    try:
        # Add image context if available
        if st.session_state.uploaded_audio:
            user_input += "\n\n[Note: User has uploaded a file for analysis]"
        
        response = st.session_state.assistant.chat(user_input)
        
    except Exception as e:
        response = f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."
    
    st.session_state.speaking = False
    return response

def format_call_duration(start_time):
    """Format call duration as MM:SS"""
    if start_time is None:
        return "00:00"
    duration = int(time.time() - start_time)
    minutes = duration // 60
    seconds = duration % 60
    return f"{minutes:02d}:{seconds:02d}"

def start_call():
    """Start a new call session"""
    st.session_state.call_active = True
    st.session_state.call_start_time = time.time()
    st.session_state.voice_messages = []
    st.rerun()

def end_call():
    """End the current call session"""
    st.session_state.call_active = False
    st.session_state.speaking = False
    st.session_state.call_start_time = None
    st.rerun()

def toggle_speaking():
    """Toggle speaking animation"""
    st.session_state.speaking = not st.session_state.speaking
    st.rerun()

# Main voice interface
st.markdown('<div class="voice-container">', unsafe_allow_html=True)

# Title
st.markdown('<h1 style="color: white; text-align: center; margin-bottom: 1rem;">🏥 Medora Consultant</h1>', unsafe_allow_html=True)

# Main UI Layout with Modern Design
st.markdown('''
<div class="main-container">
    <div class="voice-container">
''', unsafe_allow_html=True)

# Enhanced Avatar with Advanced Animations
avatar_state = ""
if st.session_state.is_speaking:
    avatar_state = "speaking"
elif st.session_state.in_call:
    avatar_state = "listening"

st.markdown(f'''
         <div class="avatar-container {avatar_state}">
             <div class="avatar-circle">
                 <div class="avatar-content">
                     🏥
                 </div>
             </div>
         </div>
''', unsafe_allow_html=True)

# Call timer
if st.session_state.call_active and st.session_state.call_start_time:
    duration = format_call_duration(st.session_state.call_start_time)
    st.markdown(f'        <div class="call-timer">⏱️ {duration}</div>', unsafe_allow_html=True)

# Status display
if st.session_state.call_active:
    status = "🟢 Call Active - Listening..." if not st.session_state.speaking else "🔴 AI Speaking..."
else:
    status = "⚪ Medora Consultant ready to start call"

status_indicator_class = "status-speaking" if st.session_state.speaking else "status-online" if st.session_state.call_active else ""
st.markdown(f'        <div class="status-display"><span class="status-indicator {status_indicator_class}"></span>{status}</div>', unsafe_allow_html=True)

# iPhone-style Call Controls with Timer
if st.session_state.call_active and st.session_state.call_start_time:
    elapsed_time = int(time.time() - st.session_state.call_start_time)
    minutes = elapsed_time // 60
    seconds = elapsed_time % 60
    timer_display = f"{minutes:02d}:{seconds:02d}"
    
    st.markdown(f"""
    <div style="text-align: center; margin: 1rem 0;">
        <div style="color: var(--text-primary); font-size: 1.2rem; font-weight: 600; font-family: 'SF Pro Display', monospace;">
            {timer_display}
        </div>
    </div>
    """, unsafe_allow_html=True)

# iPhone-style Call Control Buttons
st.markdown('<div class="phone-controls">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if not st.session_state.call_active:
        if st.button("📞", key="dial", help="Start Call"):
            start_call()
        
        st.markdown("""
        <style>
        div[data-testid="stButton"] > button[key="dial"] {
            background: linear-gradient(135deg, #30d158, #28cd41) !important;
            border: none !important;
            border-radius: 50% !important;
            width: 80px !important;
            height: 80px !important;
            font-size: 2rem !important;
            color: white !important;
            box-shadow: 0 4px 20px rgba(48, 209, 88, 0.4) !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stButton"] > button[key="dial"]:hover {
            transform: scale(1.05) !important;
            box-shadow: 0 6px 25px rgba(48, 209, 88, 0.6) !important;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        if st.button("🎤", key="toggle_mic", help="Toggle Speaking"):
            toggle_speaking()

with col2:
    # Enhanced Image Upload Section
    st.markdown('''
    <div class="upload-area" style="text-align: center; padding: 1rem; background: var(--glass-bg); border: 1px solid var(--glass-border); border-radius: 16px; margin: 0.5rem 0;">
        <div style="color: var(--text-secondary);">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📁</div>
            <p style="margin: 0; font-size: 0.9rem;">Upload Medical File folder</p>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload",
        type=['mp3', 'wav', 'png', 'jpg', 'jpeg'],
        key="voice_upload",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        st.session_state.uploaded_audio = uploaded_file
        st.markdown('''
        <div style="text-align: center; margin-top: 0.5rem;">
            <span style="background: var(--accent-green); color: var(--bg-primary); padding: 0.3rem 0.8rem; border-radius: 12px; font-size: 0.8rem; font-weight: 500;">
                ✅ File uploaded!
            </span>
        </div>
        ''', unsafe_allow_html=True)

with col3:
    if st.session_state.call_active:
        if st.button("📵", key="end_call", help="End Call"):
            end_call()
        
        st.markdown("""
        <style>
        div[data-testid="stButton"] > button[key="end_call"] {
            background: linear-gradient(135deg, #ff453a, #ff6b47) !important;
            border: none !important;
            border-radius: 50% !important;
            width: 80px !important;
            height: 80px !important;
            font-size: 2rem !important;
            color: white !important;
            box-shadow: 0 4px 20px rgba(255, 69, 58, 0.4) !important;
            transition: all 0.2s ease !important;
            transform: rotate(135deg) !important;
        }
        div[data-testid="stButton"] > button[key="end_call"]:hover {
            transform: rotate(135deg) scale(1.05) !important;
            box-shadow: 0 6px 25px rgba(255, 69, 58, 0.6) !important;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.button("📵", key="end_call_disabled", help="End Call", disabled=True)

st.markdown('''
    </div>
</div>
''', unsafe_allow_html=True)

# Voice input simulation (since Streamlit doesn't support real-time audio)
if st.session_state.call_active:
    st.markdown('<div style="color: white; text-align: center; margin: 1rem 0;">💡 Type your message or upload an audio file</div>', unsafe_allow_html=True)
    
    voice_input = st.text_input(
        "Voice Input Simulation",
        placeholder="Speak your symptoms or medical question...",
        key="voice_sim",
        label_visibility="collapsed"
    )
    
    if voice_input:
        # Add user message
        st.session_state.voice_messages.append({
            "role": "user",
            "content": voice_input,
            "timestamp": time.time()
        })
        
        # Get actual medical assistant response
        ai_response = get_medical_response(voice_input)
        
        st.session_state.voice_messages.append({
            "role": "assistant",
            "content": ai_response,
            "timestamp": time.time()
        })
        
        # Clear input and rerun
        st.rerun()

# Enhanced Conversation History
if st.session_state.voice_messages:
    st.markdown('''
    <div class="conversation-history">
        <h3 style="color: var(--text-primary); margin-bottom: 1.5rem; font-weight: 600; text-align: center;">
            💬 Conversation History
        </h3>
    ''', unsafe_allow_html=True)
    
    for msg in st.session_state.voice_messages[-6:]:  # Show last 6 messages
        role_class = "user-message" if msg["role"] == "user" else "assistant-message"
        icon = "👤" if msg["role"] == "user" else "🏥"
        role_name = "You" if msg["role"] == "user" else "Medical Assistant"
        st.markdown(f'''
        <div class="conversation-item {role_class}">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <span>{icon}</span>
                <strong>{role_name}</strong>
            </div>
            <div>{msg["content"]}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# JavaScript for real-time updates (if needed)
st.markdown("""
<script>
function toggleSpeaking() {
    // This would trigger the speaking animation
    console.log('Avatar clicked');
}

// Auto-refresh for call timer
if (window.callActive) {
    setTimeout(() => {
        window.location.reload();
    }, 1000);
}
</script>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.7); margin-top: 2rem; font-size: 0.9rem;">
⚠️ This is a voice-enabled medical assistant. Always consult healthcare professionals for proper diagnosis.
</div>
""", unsafe_allow_html=True)