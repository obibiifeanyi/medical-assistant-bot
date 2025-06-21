"""
Kickstart HealthIQ Streamlit App with Image Analysis
===========================================================

This version includes image upload and analysis capabilities for visual diagnosis.
Supports skin conditions, burns, rashes, jaundice, and other visible symptoms.
"""

import os
import sys
import base64
from typing import Optional

import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page config
st.set_page_config(
    page_title="Kickstart HealthIQ Bot",
    page_icon="🏥",
    layout="wide"
)

# Title
st.title("🏥 Kickstart HealthIQ Chatbot")
st.markdown("*Now with image analysis capabilities for visual symptoms*")

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
    1. Make sure all Python files are uploaded (including enhanced versions)
    2. Create 'data/' folder and upload CSV files
    3. Create 'indices/' folder and upload FAISS index files
    4. Check the deployment guide for detailed instructions
    """)
    st.stop()

# Try to import modules
try:
    from medical_agent_langchain import create_medical_assistant  # CORRECTED: Back to original function name
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
except ImportError as e:
    st.error(f"❌ Error importing modules: {str(e)}")
    st.info("Please check that all dependencies are installed correctly.")
    st.stop()

# Helper functions
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
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
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
        st.caption(f"Size: {original_image.width}x{original_image.height} pixels")
        
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'assistant' not in st.session_state:
    st.session_state.assistant = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

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

    st.markdown("---")
    
    # Image Upload Section
    st.subheader("📸 Image Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload medical image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image of visible symptoms (rash, burn, discoloration, etc.)"
    )
    
    # Additional context for image
    image_context = st.text_area(
        "Additional Context",
        placeholder="Describe when symptoms appeared, any pain, itching, etc.",
        help="Provide additional context about the image or symptoms"
    )
    
    if uploaded_file:
        st.session_state.uploaded_image = uploaded_file
        display_image_preview(uploaded_file)
        
        # Show image analysis options
        st.info("💡 Image uploaded! You can now:")
        st.markdown("""
        - Describe symptoms in text + image analysis
        - Ask for image-only analysis
        - Combine both for comprehensive diagnosis
        """)
    else:
        st.session_state.uploaded_image = None
        st.info("Upload an image for visual symptom analysis")

    st.markdown("---")

    # About section
    st.subheader("ℹ️ About")
    st.markdown("""
    This enhanced AI assistant can analyze both text descriptions and medical images.

    **Image Analysis Can Help With:**
    - Skin conditions (rashes, lesions, discoloration)
    - Burns and wounds
    - Swelling and inflammation
    - Eye conditions
    - Jaundice and color changes
    - Visible symptoms

    **⚠️ Disclaimer:** Not a replacement for professional medical advice.
    """)

    # Clear button
    if st.button("🗑️ Clear Chat & Image"):
        st.session_state.messages = []
        st.session_state.uploaded_image = None
        if st.session_state.assistant:
            st.session_state.assistant = None
        st.rerun()

# Main content
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
        if os.getenv('OPENAI_API_KEY'):
            try:
                vision_model = ChatOpenAI(
                    model="gpt-4o",  # Try GPT-4o first (has integrated vision)
                    temperature=0.1,
                    max_tokens=1000
                )
            except Exception as e:
                st.warning(f"⚠️ GPT-4o not available, trying GPT-4 Vision: {str(e)}")
                # Fall back to GPT-4 Vision Preview
                try:
                    vision_model = ChatOpenAI(
                        model="gpt-4-vision-preview",
                        temperature=0.1,
                        max_tokens=1000
                    )
                except Exception as e2:
                    st.warning(f"⚠️ No vision-capable model available: {str(e2)}")

        return {
            'symptom_index': faiss_symptom,
            'severity_index': faiss_severity,
            'precautions': df_precautions,
            'descriptions': df_descriptions,
            'severity': df_severity,
            'vision_model': vision_model
        }
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        return None

# Initialize assistant
def init_assistant():
    """Initialize the Kickstart HealthIQ."""
    if not os.getenv('OPENAI_API_KEY'):
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar.")
        return False

    if st.session_state.assistant is None:
        with st.spinner("Loading enhanced medical knowledge base..."):
            resources = load_resources()

        if resources:
            try:
                st.session_state.assistant = create_medical_assistant(  # CORRECTED: Back to original function name
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
            except Exception as e:
                st.error(f"Failed to initialize assistant: {str(e)}")
                return False
    return True

# Chat interface
if init_assistant():
    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("has_image", False):
                st.caption("📸 *Analysis included uploaded image*")
            st.write(msg["content"])

    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        prompt = st.chat_input("Describe your symptoms or ask about the uploaded image...")
    
    with col2:
        # Quick action buttons
        if st.session_state.uploaded_image:
            if st.button("🔍 Analyze Image Only"):
                prompt = "Please analyze the uploaded medical image and tell me what you see."
    
    if prompt:
        # Prepare image data if available
        image_data_b64 = ""
        has_image = False
        
        if st.session_state.uploaded_image:
            # Reset file pointer
            st.session_state.uploaded_image.seek(0)
            image_data_b64 = encode_image_to_base64(st.session_state.uploaded_image)
            has_image = bool(image_data_b64)
        
        # Add user message
        user_msg = {
            "role": "user", 
            "content": prompt,
            "has_image": has_image
        }
        st.session_state.messages.append(user_msg)
        
        with st.chat_message("user"):
            if has_image:
                st.caption("📸 *Message includes uploaded image*")
            st.write(prompt)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    # Use enhanced chat method with image support
                    if has_image:
                        response = st.session_state.assistant.chat_with_image(
                            user_input=prompt,
                            image_data_b64=image_data_b64,
                            additional_context=image_context
                        )
                    else:
                        response = st.session_state.assistant.chat(prompt)
                    
                    st.write(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "has_image": has_image
                    })
                    
                except Exception as e:
                    ERROR_MSG = f"Error: {str(e)}"
                    st.error(ERROR_MSG)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ERROR_MSG,
                        "has_image": False
                    })

else:
    # Show setup help
    st.info("""
    👋 **Welcome to Kickstart HealthIQ!**

    **New Features:**
    - 📸 **Image Analysis**: Upload photos of visible symptoms
    - 🔍 **Visual Diagnosis**: AI analysis of skin conditions, rashes, burns, etc.
    - 🧠 **Multimodal Analysis**: Combines text + image for better diagnosis

    **To get started:**
    1. Enter your OpenAI API key in the sidebar
    2. (Optional) Upload an image of visible symptoms
    3. Describe your symptoms or ask about the image
    4. Get comprehensive analysis combining text and visual information

    **Example interactions:**
    - "I have a rash on my arm" + upload image
    - "Analyze this skin condition" (image only)
    - "I have fever and this rash appeared yesterday"
    - "What could cause this discoloration?"

    **Image Analysis Works Best For:**
    - Skin rashes, lesions, discoloration
    - Burns, wounds, bruises
    - Swelling, inflammation
    - Eye conditions (redness, discharge)
    - Jaundice, pallor, cyanosis
    - Visible infections or abnormalities
    """)

# Footer
st.markdown("---")
st.caption("""
⚠️ **Kickstart HealthIQ Disclaimer**: This tool provides general information only and includes AI-powered image analysis. 
Visual analysis is for informational purposes and should not replace professional medical examination. 
Always consult healthcare professionals for proper diagnosis and treatment.
""")

# Additional info in expander
with st.expander("📋 Image Upload Guidelines"):
    st.markdown("""
    **For Best Results:**
    - Use good lighting and clear focus
    - Capture the affected area directly
    - Include relevant surrounding skin for context
    - Take photos from appropriate distance (not too close/far)
    
    **Supported Conditions:**
    - Skin rashes, eczema, dermatitis
    - Burns (thermal, chemical, sun)
    - Bruises, cuts, wounds
    - Insect bites, hives
    - Discoloration (jaundice, pallor)
    - Swelling, inflammation
    - Eye conditions (redness, discharge)
    
    **Privacy & Security:**
    - Images are processed in real-time only
    - No images are permanently stored
    - All analysis happens during your session
    """)

if st.session_state.uploaded_image and os.getenv('OPENAI_API_KEY'):
    st.success("✅ Ready for enhanced analysis with image + text!")
elif st.session_state.uploaded_image:
    st.warning("⚠️ Image uploaded but API key needed for analysis")
elif os.getenv('OPENAI_API_KEY'):
    st.info("💬 Ready for text-based analysis (upload image for visual analysis)")