"""
Kickstart HealthIQ Streamlit App - Clean Layout
=====================================================

This version has proper layout with:
- Chat input pinned to bottom above disclaimer
- Scrollable chat history at top
- Disclaimer at absolute bottom
"""

import os
import sys
import base64
from typing import Optional

import streamlit as st
import pandas as pd
from PIL import Image

# Add current directory and src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Page config
st.set_page_config(
    page_title="Kickstart HealthIQ Bot",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for better layout - FIXED VERSION
st.markdown("""
<style>
    /* Hide default Streamlit elements that cause white bars */
    .css-1d391kg {display: none;}
    .css-18e3th9 {padding-top: 0 !important;}
    .css-1y4p8pa {padding-top: 0 !important;}
    
    /* Remove default Streamlit padding/margins */
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 100px !important;
        max-width: 100% !important;
    }
    
    /* Chat History Section - Fixed styling */
    .chat-history-container {
        background: transparent !important;
        border: none !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Individual chat messages container */
    .chat-messages {
        max-height: 500px;
        overflow-y: auto;
        border: 2px solid #333;
        border-radius: 10px;
        padding: 1rem;
        background: #1a1a1a;
        margin-bottom: 2rem;
    }
    
    /* Input section - Remove white bar */
    .input-section {
        background: transparent !important;
        padding: 1rem 0 0 0 !important;
        margin: 2rem 0 0 0 !important;
        border: none !important;
    }
    
    /* Chat input styling */
    .stChatInput {
        background: transparent !important;
    }
    
    /* Fixed disclaimer */
    .disclaimer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: #2d2d2d;
        color: white;
        padding: 0.75rem;
        text-align: center;
        border-top: 2px solid #555;
        z-index: 1000;
        font-size: 0.85rem;
    }
    
    /* Custom scrollbar for chat */
    .chat-messages::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-messages::-webkit-scrollbar-track {
        background: #333;
        border-radius: 4px;
    }
    
    .chat-messages::-webkit-scrollbar-thumb {
        background: #666;
        border-radius: 4px;
    }
    
    .chat-messages::-webkit-scrollbar-thumb:hover {
        background: #888;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🏥 Kickstart HealthIQ Chatbot")
st.markdown("*Now with proper agentic image analysis capabilities*")


def check_files():
    """Check if all required files are present."""
    required_files = {
        "Python modules": [
            "src/medical_tools.py", 
            "src/medical_agent_langchain.py",
            "src/vision_tools.py"
        ],
        "Data files": [
            "data/disease_symptoms.csv",
            "data/disease_symptom_severity.csv",
            "data/disease_precautions.csv",
            "data/disease_symptom_description.csv"
        ],
        "FAISS indices": [
            "indices/faiss_symptom_index_medibot",
            "indices/faiss_severity_index_medibot",
        ]
    }

    missing_files = {}
    for file_category, files in required_files.items():
        missing = [f for f in files if not os.path.exists(f)]
        if missing:
            missing_files[file_category] = missing

    return missing_files


def check_api_key_available() -> bool:
    """Check if OpenAI API key is already available from environment or secrets."""
    # Check environment variable
    env_key = os.getenv('OPENAI_API_KEY')
    
    # Check Streamlit secrets
    secrets_key = None
    try:
        secrets_key = st.secrets.get("OPENAI_API_KEY", None)
    except (AttributeError, FileNotFoundError):
        pass
    
    return bool(env_key or secrets_key)


# Check files
missing = check_files()
if missing:
    st.error("⚠️ Missing required files!")
    for category, files in missing.items():
        st.write(f"**{category}:**")
        for f in files:
            st.write(f"  - {f}")

    st.info("""
    📋 **Setup Instructions for Organized Structure:**
    
    **Expected Project Structure:**
    ```
    medical-assistant-bot/
    ├── app.py                          # This file
    ├── src/
    │   ├── __init__.py                # Create empty file
    │   ├── medical_tools.py           # Move here
    │   ├── medical_agent_langchain.py # Move here
    │   └── vision_tools.py            # Move here
    ├── data/
    │   ├── disease_symptoms.csv
    │   └── ... (other CSV files)
    └── indices/
        ├── faiss_symptom_index_medibot/
        └── faiss_severity_index_medibot/
    ```
    
    **Quick Setup:**
    1. Create 'src/' folder
    2. Create empty 'src/__init__.py' file
    3. Move Python modules to src/ folder
    4. Ensure data/ and indices/ folders exist
    """)
    st.stop()

# Try to import modules from src folder
try:
    # Import from src folder
    from medical_agent_langchain import create_medical_assistant
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    
    st.success("✅ All modules imported successfully from src/ folder")
    
except ImportError as e:
    st.error(f"❌ Error importing modules from src/ folder: {str(e)}")
    st.info("""
    **Import Error Help:**
    
    If you see this error, you have two options:
    
    **Option A: Use Root Structure (Simpler)**
    - Put medical_tools.py, medical_agent_langchain.py, vision_tools.py in the main folder (same level as app.py)
    
    **Option B: Use src/ Structure (Recommended)**
    - Create src/ folder
    - Move all .py files (except app.py) to src/
    - Create empty src/__init__.py file
    
    The app will work with either structure!
    """)
    st.stop()


def encode_image_to_base64(image_file) -> Optional[str]:
    """Convert uploaded image to base64 string."""
    try:
        if image_file is not None:
            # Read image bytes
            image_bytes = image_file.read()

            # Encode to base64
            base64_string = base64.b64encode(image_bytes).decode('utf-8')
            return base64_string
        return None
    except Exception:
        st.error("Error processing image")
        return None


def display_image_preview(image_file, max_width=300):
    """Display image preview in sidebar."""
    try:
        image = Image.open(image_file)

        # Resize for preview if too large
        if image.width > max_width:
            ratio = max_width / image.width
            new_height = int(image.height * ratio)
            image = image.resize((max_width, new_height))

        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Show image info
        original_image = Image.open(image_file)
        st.caption(f"Size: {original_image.width}x{original_image.height} px")

    except Exception:
        st.error("Error displaying image")


def clear_all_data():
    """Clear all chat and image data."""
    # Clear session state
    st.session_state.messages = []
    st.session_state.uploaded_image = None
    st.session_state.image_base64 = None
    
    # Reset assistant
    if st.session_state.assistant:
        st.session_state.assistant = None
    
    # Force rerun to update UI
    st.rerun()


# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'assistant' not in st.session_state:
    st.session_state.assistant = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'image_base64' not in st.session_state:
    st.session_state.image_base64 = None

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")

    # Dynamic API Key input - only show if not already available
    api_key_available = check_api_key_available()
    
    if not api_key_available:
        # Show API key input if not available from environment/secrets
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to use the medical assistant"
        )

        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
    else:
        # API key already available - show status
        st.success("✅ OpenAI API Key configured")

    st.markdown("---")

    # Image Upload Section
    st.subheader("📸 Image Analysis")

    uploaded_file = st.file_uploader(
        "Upload medical image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image of visible symptoms (rash, burn, etc.)",
        key="image_uploader"
    )

    # Additional context for image
    image_context = st.text_area(
        "Additional Context",
        placeholder="Describe when symptoms appeared, any pain, itching, etc.",
        help="Provide additional context about the image or symptoms"
    )

    # Handle image upload and display
    if uploaded_file:
        st.session_state.uploaded_image = uploaded_file
        # Convert to base64 and store
        uploaded_file.seek(0)  # Reset file pointer
        st.session_state.image_base64 = encode_image_to_base64(uploaded_file)
        
        display_image_preview(uploaded_file)

        # Show image analysis options
        st.info("💡 Image uploaded! You can now:")
        st.markdown("""
        - Ask "Analyze the uploaded image"
        - Describe symptoms + mention the image
        - Let the AI agent decide when to use image analysis
        """)
    else:
        # Clear image data if no file uploaded
        if st.session_state.uploaded_image is not None:
            st.session_state.uploaded_image = None
            st.session_state.image_base64 = None
        
        st.info("Upload an image for AI-powered visual symptom analysis")

    st.markdown("---")

    # Clear button
    if st.button("🗑️ Clear Chat & Image", use_container_width=True):
        clear_all_data()

    st.markdown("---")

    # About section
    st.subheader("ℹ️ About")
    st.markdown("""
    This enhanced AI assistant uses agentic tools to analyze both text
    descriptions and medical images.

    **Agentic Image Analysis:**
    - AI agent automatically chooses when to analyze images
    - Uses same vision technology as ChatGPT
    - Integrates visual findings with medical knowledge base
    - Provides comprehensive multimodal diagnosis

    **⚠️ Disclaimer:** Not a replacement for professional medical advice.
    """)


@st.cache_resource
def load_resources():
    """Load all medical resources including vision model."""
    try:
        # Load embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load FAISS indices
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

        # Initialize vision model if API key is available
        vision_model = None
        if os.getenv('OPENAI_API_KEY') or check_api_key_available():
            try:
                vision_model = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0.1
                )
            except Exception:
                st.warning("⚠️ GPT-4o not available, trying GPT-4 Vision")
                try:
                    vision_model = ChatOpenAI(
                        model="gpt-4-vision-preview",
                        temperature=0.1
                    )
                except Exception:
                    st.warning("⚠️ No vision-capable model available")

        return {
            'symptom_index': faiss_symptom,
            'severity_index': faiss_severity,
            'precautions': df_precautions,
            'descriptions': df_descriptions,
            'severity': df_severity,
            'vision_model': vision_model
        }
    except Exception:
        st.error("Error loading resources")
        return None


def init_assistant():
    """Initialize the Kickstart HealthIQ."""
    if not (os.getenv('OPENAI_API_KEY') or check_api_key_available()):
        if not api_key_available:
            st.warning("⚠️ Please enter your OpenAI API key in the sidebar.")
        return False

    if st.session_state.assistant is None:
        with st.spinner("Loading enhanced medical knowledge base..."):
            resources = load_resources()

        if resources:
            try:
                st.session_state.assistant = create_medical_assistant(
                    faiss_symptom_index=resources['symptom_index'],
                    faiss_severity_index=resources['severity_index'],
                    df_disease_precautions=resources['precautions'],
                    df_disease_symptom_description=resources['descriptions'],
                    df_disease_symptom_severity=resources['severity'],
                    vision_model=resources['vision_model'],
                    model_name="gpt-3.5-turbo",
                    use_memory=True
                )
                return True
            except Exception:
                st.error("Failed to initialize assistant")
                return False
    return True


# Main layout container
main_container = st.container()

with main_container:
    # Check if assistant is initialized
    if init_assistant():
        
        # Chat History Section - FIXED
        st.markdown("### 💬 Chat History")
        
        if st.session_state.messages:
            # Create properly scrollable chat container
            st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
            
            for i, msg in enumerate(st.session_state.messages):
                with st.chat_message(msg["role"]):
                    if msg.get("has_image", False):
                        st.caption("📸 *Message includes uploaded image*")
                    st.write(msg["content"])
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Welcome message - properly formatted HTML
            st.markdown("""
            <div style="background: #1a1a1a; border: 2px solid #333; border-radius: 10px; padding: 2rem; margin-bottom: 2rem; color: white;">
                <h3 style="color: white; margin-top: 0;">👋 Welcome to Kickstart HealthIQ!</h3>
                
                <p><strong>🤖 Agentic AI Features:</strong></p>
                <ul style="margin-left: 20px;">
                    <li>Intelligent tool selection</li>
                    <li>Real image analysis</li>
                    <li>Multimodal integration</li>
                    <li>Medical knowledge base</li>
                </ul>
                
                <p><strong>🎯 To get started:</strong></p>
                <ol style="margin-left: 20px;">
                    <li>Upload an image of visible symptoms (optional)</li>
                    <li>Describe your symptoms in the chat box below</li>
                    <li>Get comprehensive analysis</li>
                </ol>
                
                <p><strong>💬 Try saying:</strong></p>
                <ul style="margin-left: 20px;">
                    <li><em>"Analyze this skin condition"</em></li>
                    <li><em>"I have fever and headache"</em></li>
                    <li><em>"What do you see in this image?"</em></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Input Section - NO WHITE BARS
        col1, col2 = st.columns([4, 1])

        with col1:
            USER_PROMPT = st.chat_input(
                "Describe your symptoms or ask me to analyze the uploaded image..."
            )

        with col2:
            # Quick action buttons
            if st.session_state.uploaded_image:
                if st.button("🔍 Analyze Image", use_container_width=True):
                    USER_PROMPT = "Please analyze the uploaded medical image using your image analysis tools and tell me what medical conditions you can identify."

        # Process user input
        if USER_PROMPT:
            # Determine if we have an image available
            HAS_IMAGE = bool(st.session_state.image_base64)

            # Add user message
            user_msg = {
                "role": "user",
                "content": USER_PROMPT,
                "has_image": HAS_IMAGE
            }
            st.session_state.messages.append(user_msg)

            # Get response
            with st.spinner("Analyzing..."):
                try:
                    # Enhanced prompt that provides image data to the agent
                    if HAS_IMAGE:
                        enhanced_prompt = f"""
                        USER REQUEST: {USER_PROMPT}

                        AVAILABLE TOOLS: You have access to medical analysis tools including:
                        - analyze_medical_image(image_base64, additional_context): Analyzes uploaded medical images
                        - analyze_symptoms_direct(user_input): Analyzes text symptoms
                        - analyze_combined_symptoms(text_symptoms, visual_symptoms): Combines both analyses
                        - get_disease_description(disease_name): Gets disease details
                        - get_disease_precautions(disease_name): Gets treatment recommendations

                        UPLOADED IMAGE DATA: {st.session_state.image_base64}
                        ADDITIONAL CONTEXT: {image_context}

                        INSTRUCTIONS:
                        1. If the user is asking about image analysis or mentions visual symptoms, use analyze_medical_image() with the provided image data
                        2. If the user describes text symptoms AND there's an image, consider using both analyze_symptoms_direct() and analyze_medical_image(), then analyze_combined_symptoms()
                        3. Always get disease descriptions for any conditions you identify
                        4. Provide comprehensive medical analysis
                        5. Include visual findings when image analysis is performed

                        Proceed with the appropriate analysis based on the user's request and available data.
                        """
                    else:
                        enhanced_prompt = USER_PROMPT

                    # Call the agent with enhanced prompt
                    if hasattr(st.session_state.assistant, 'chat'):
                        response = st.session_state.assistant.chat(enhanced_prompt)
                    else:
                        response = "Assistant not properly initialized"

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "has_image": HAS_IMAGE
                    })
                    
                    # Rerun to show the new messages
                    st.rerun()

                except Exception as e:
                    ERROR_MSG = f"An error occurred during analysis: {str(e)}"
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ERROR_MSG,
                        "has_image": False
                    })
                    st.rerun()

    else:
        # Show setup help when assistant is not initialized
        st.error("❌ Assistant not properly initialized. Please check your API key and try again.")

# Fixed Disclaimer at bottom
st.markdown("""
<div class="disclaimer">
⚠️ <strong>Kickstart HealthIQ Disclaimer:</strong> This tool provides general information only and includes AI-powered agentic image analysis. 
Visual analysis is for informational purposes and should not replace professional medical examination. 
Always consult healthcare professionals for proper diagnosis and treatment.
</div>
""", unsafe_allow_html=True)