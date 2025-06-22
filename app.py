"""
Kickstart HealthIQ Streamlit App - PRODUCTION VERSION
===================================================

Clean production version with all debugging removed.
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

# CSS Styling
st.markdown("""
<style>
    /* Remove default Streamlit padding/margins */
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 5rem !important;
        max-width: 100% !important;
    }
    
    /* Chat messages container with better scrolling */
    .chat-messages {
        max-height: 400px;
        overflow-y: auto;
        border: 2px solid #333;
        border-radius: 10px;
        padding: 1rem;
        background: #1a1a1a;
        margin-bottom: 1rem;
    }
    
    /* Container styling for welcome message */
    .stContainer {
        background: transparent;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Input section styling */
    .input-section {
        background: transparent;
        padding: 1rem 0;
        margin: 1rem 0;
    }
    
    /* Disclaimer styling */
    .disclaimer {
        position: sticky;
        bottom: 0;
        left: 0;
        right: 0;
        background: #2d2d2d;
        color: white;
        padding: 1rem;
        text-align: center;
        border-top: 2px solid #555;
        border-radius: 8px 8px 0 0;
        margin-top: 2rem;
        z-index: 1000;
        font-size: 0.85rem;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.3);
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
    
    /* Ensure sidebar content doesn't overflow */
    .css-1d391kg {
        padding-bottom: 2rem;
    }
    
    /* Better spacing for buttons */
    .stButton > button {
        width: 100%;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🏥 Kickstart HealthIQ Chatbot")
st.markdown("*Now with proper agentic image analysis capabilities*")


# Helper functions
def clear_all_data():
    """Clear all chat and image data."""
    st.session_state.messages = []
    st.session_state.uploaded_image = None
    st.session_state.image_base64 = None
    
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    st.session_state.uploader_key += 1
    
    if hasattr(st.session_state, 'assistant') and st.session_state.assistant:
        st.session_state.assistant = None
    
    st.rerun()


def manage_conversation_memory():
    """Automatically manage conversation memory to prevent context length issues."""
    if len(st.session_state.messages) > 10:
        total_chars = sum(len(msg["content"]) for msg in st.session_state.messages)
        estimated_tokens = total_chars // 4
        
        if estimated_tokens > 12000:
            st.info("🔄 **Auto-trimming old messages** to keep conversation flowing smoothly...")
            
            if len(st.session_state.messages) > 8:
                first_msg = st.session_state.messages[0]
                recent_msgs = st.session_state.messages[-6:]
                
                transition_msg = {
                    "role": "assistant",
                    "content": "📝 *[Earlier conversation trimmed to manage memory - conversation continues below]*",
                    "has_image": False
                }
                
                st.session_state.messages = [first_msg, transition_msg] + recent_msgs
                
                if hasattr(st.session_state.assistant, 'reset_conversation'):
                    st.session_state.assistant.reset_conversation()


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
    env_key = os.getenv('OPENAI_API_KEY')
    
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
    """Convert uploaded image to base64 string with enhanced validation and preprocessing."""
    try:
        if image_file is not None:
            image_file.seek(0)
            image_bytes = image_file.read()
            
            if not image_bytes:
                st.error("Image file appears to be empty")
                return None
            
            if len(image_bytes) < 1000:
                st.error("Image file is too small (less than 1KB)")
                return None
                
            if len(image_bytes) > 20 * 1024 * 1024:  # 20MB
                st.error("Image file is too large (over 20MB)")
                return None
            
            # Validate image format by checking magic bytes
            format_detected = False
            if image_bytes.startswith(b'\xFF\xD8\xFF'):
                if b'\xFF\xD9' not in image_bytes[-10:]:
                    st.warning("⚠️ JPEG file may be corrupted - missing end marker")
                format_detected = True
            elif image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
                format_detected = True
            elif image_bytes.startswith(b'GIF87a') or image_bytes.startswith(b'GIF89a'):
                format_detected = True
            elif image_bytes.startswith(b'BM'):
                st.error("BMP format is not supported. Please convert to JPEG or PNG.")
                return None
            
            if not format_detected:
                st.warning("⚠️ Image format may not be supported. Please use JPEG or PNG for best results.")
            
            # Try to process with PIL for additional validation
            try:
                import io
                img = Image.open(io.BytesIO(image_bytes))
                
                if img.format == 'RGBA' or img.mode == 'RGBA':
                    st.info("Converting RGBA image to RGB for better compatibility...")
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    
                    output = io.BytesIO()
                    rgb_img.save(output, format='JPEG', quality=90)
                    image_bytes = output.getvalue()
                    
                elif img.format not in ['JPEG', 'PNG']:
                    st.info(f"Converting {img.format} to JPEG for better compatibility...")
                    output = io.BytesIO()
                    img.convert('RGB').save(output, format='JPEG', quality=90)
                    image_bytes = output.getvalue()
                
            except Exception as pil_error:
                st.warning(f"Could not validate image with PIL: {str(pil_error)}")
            
            # Encode to base64
            base64_string = base64.b64encode(image_bytes).decode('utf-8')
            
            if len(base64_string) < 100:
                st.error("Image encoding failed - result too short")
                return None
                
            st.success(f"✅ Image encoded successfully ({len(image_bytes):,} bytes)")
            return base64_string
            
        return None
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None


def display_image_preview(image_file, max_width=300):
    """Display image preview in sidebar."""
    try:
        image = Image.open(image_file)

        if image.width > max_width:
            ratio = max_width / image.width
            new_height = int(image.height * ratio)
            image = image.resize((max_width, new_height))

        st.image(image, caption="Uploaded Image", use_container_width=True)

        original_image = Image.open(image_file)
        st.caption(f"Size: {original_image.width}x{original_image.height} px")

    except Exception:
        st.error("Error displaying image")


# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'assistant' not in st.session_state:
    st.session_state.assistant = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'image_base64' not in st.session_state:
    st.session_state.image_base64 = None
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")

    api_key_available = check_api_key_available()
    
    if not api_key_available:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to use the medical assistant"
        )

        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
    else:
        st.success("✅ OpenAI API Key configured")

    st.markdown("---")

    # Image Upload Section
    st.subheader("📸 Image Analysis")

    uploaded_file = st.file_uploader(
        "Upload medical image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image of visible symptoms (rash, burn, etc.)",
        key=f"image_uploader_{st.session_state.uploader_key}"
    )

    image_context = st.text_area(
        "Additional Context",
        placeholder="Describe when symptoms appeared, any pain, itching, etc.",
        help="Provide additional context about the image or symptoms"
    )

    # Handle image upload and display
    if uploaded_file is not None:
        st.session_state.uploaded_image = uploaded_file
        uploaded_file.seek(0)
        st.session_state.image_base64 = encode_image_to_base64(uploaded_file)
        
        if st.session_state.image_base64:
            display_image_preview(uploaded_file)

            st.info("💡 Image uploaded! You can now:")
            st.markdown("""
            - Ask "Analyze the uploaded image"
            - Describe symptoms + mention the image
            - Let the AI agent decide when to use image analysis
            """)
    else:
        st.session_state.uploaded_image = None
        st.session_state.image_base64 = None
        
        st.info("Upload an image for AI-powered visual symptom analysis")

    st.markdown("---")

    # Clear button with conversation status
    message_count = len(st.session_state.messages)
    if message_count > 0:
        total_chars = sum(len(msg["content"]) for msg in st.session_state.messages)
        estimated_tokens = total_chars // 4
        
        if estimated_tokens > 12000:
            st.warning(f"💬 Conversation getting long ({message_count} messages)")
            st.caption("Consider clearing chat soon to avoid memory limits")
        elif estimated_tokens > 8000:
            st.info(f"💬 {message_count} messages in conversation")
        else:
            st.caption(f"💬 {message_count} messages")
    
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


# Main content function
def main_content():
    """Render main content with fixed layout."""
    
    if not init_assistant():
        st.error("❌ Assistant not properly initialized. Please check your API key and try again.")
        return
    
    # Chat History Section
    st.markdown("### 💬 Chat History")
    
    if st.session_state.messages:
        st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
        
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                if msg.get("has_image", False):
                    st.caption("📸 *Message includes uploaded image*")
                st.write(msg["content"])
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        with st.container():
            st.markdown("### 👋 Welcome to Kickstart HealthIQ!")
            
            st.markdown("**🤖 Agentic AI Features:**")
            st.markdown("""
            - Intelligent tool selection
            - Real image analysis  
            - Multimodal integration
            - Medical knowledge base
            """)
            
            st.markdown("**🎯 To get started:**")
            st.markdown("""
            1. Upload an image of visible symptoms (optional)
            2. Describe your symptoms in the chat box below
            3. Get comprehensive analysis
            """)
            
            st.markdown("**💬 Try saying:**")
            st.markdown("""
            - *"Analyze this skin condition"*
            - *"I have fever and headache"*  
            - *"What do you see in this image?"*
            """)
            
            st.markdown("---")
    
    # Input Section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])

    with col1:
        USER_PROMPT = st.chat_input(
            "Describe your symptoms or ask me to analyze the uploaded image..."
        )

    with col2:
        if st.session_state.uploaded_image and st.session_state.image_base64:
            if st.button("🔍 Analyze Image", use_container_width=True):
                USER_PROMPT = "Please analyze the uploaded medical image using your image analysis tools and tell me what medical conditions you can identify."

    st.markdown('</div>', unsafe_allow_html=True)

    # Process user input
    if USER_PROMPT:
        manage_conversation_memory()
        
        HAS_IMAGE = bool(st.session_state.image_base64)

        user_msg = {
            "role": "user",
            "content": USER_PROMPT,
            "has_image": HAS_IMAGE
        }
        st.session_state.messages.append(user_msg)

        with st.spinner("Analyzing..."):
            try:
                if HAS_IMAGE:
                    previous_visual_symptoms = []
                    
                    # Look through conversation history for previous image analysis
                    messages_to_check = st.session_state.messages[:-1]  # Exclude current user message
                    
                    print(f"🔍 SEARCHING FOR VISUAL SYMPTOMS in {len(messages_to_check)} previous messages...")
                    
                    for i, msg in enumerate(messages_to_check):
                        if msg.get("role") == "assistant":
                            content = msg.get("content", "")
                            print(f"  Checking message {i}: {content[:100]}...")
                            
                            # Method 1: Look for the exact JSON structure from analyze_medical_image tool output
                            if '"visible_symptoms":' in content:
                                try:
                                    import re
                                    # Find the JSON array in the content
                                    json_match = re.search(r'"visible_symptoms":\s*\[(.*?)\]', content, re.DOTALL)
                                    if json_match:
                                        symptoms_str = json_match.group(1)
                                        # Extract quoted strings from the array
                                        symptoms_clean = re.findall(r'"([^"]+)"', symptoms_str)
                                        if symptoms_clean:
                                            previous_visual_symptoms = symptoms_clean
                                            print(f"    ✅ Found visual symptoms via JSON: {previous_visual_symptoms}")
                                            break
                                except Exception as e:
                                    print(f"    ❌ JSON extraction failed: {e}")
                            
                            # Method 2: Look for symptoms listed in text after "Based on" or similar patterns
                            elif any(phrase in content.lower() for phrase in ["based on the symptoms", "visible symptoms", "image analysis"]):
                                try:
                                    # Look for symptom keywords in the response
                                    import re
                                    symptom_patterns = [
                                        r'\b(?:small|large|red|round|oval)?\s*(?:red\s+)?(?:bumps|spots|blisters|rash|lesions|swelling|patches)\b',
                                        r'\b(?:bumps|spots|blisters|rash|lesions|swelling|patches)\b'
                                    ]
                                    
                                    found_symptoms = []
                                    for pattern in symptom_patterns:
                                        matches = re.findall(pattern, content.lower())
                                        found_symptoms.extend([m.strip() for m in matches if m.strip()])
                                    
                                    # Remove duplicates while preserving order
                                    unique_symptoms = []
                                    for symptom in found_symptoms:
                                        if symptom not in unique_symptoms:
                                            unique_symptoms.append(symptom)
                                    
                                    if unique_symptoms:
                                        previous_visual_symptoms = unique_symptoms
                                        print(f"    ✅ Found visual symptoms via text pattern: {previous_visual_symptoms}")
                                        break
                                except Exception as e:
                                    print(f"    ❌ Pattern extraction failed: {e}")
                            
                            # Method 3: Look for any message that mentions specific visual symptoms
                            elif any(visual_term in content.lower() for visual_term in ["red bumps", "small red", "blisters", "round spots", "rash", "skin"]):
                                try:
                                    # Extract common visual symptom phrases
                                    import re
                                    visual_phrases = re.findall(r'(?:small|large|red|round|oval)?\s*(?:red\s+)?(?:bumps|spots|blisters|rash|patches)', content.lower())
                                    if visual_phrases:
                                        # Clean up the phrases
                                        cleaned_phrases = []
                                        for phrase in visual_phrases:
                                            cleaned = phrase.strip()
                                            if cleaned and len(cleaned) > 3:
                                                cleaned_phrases.append(cleaned)
                                        
                                        if cleaned_phrases:
                                            previous_visual_symptoms = cleaned_phrases
                                            print(f"    ✅ Found visual symptoms via phrase matching: {previous_visual_symptoms}")
                                            break
                                except Exception as e:
                                    print(f"    ❌ Phrase extraction failed: {e}")
                    
                    print(f"🎯 FINAL VISUAL SYMPTOMS FOUND: {previous_visual_symptoms}")
                    
                    user_symptoms_lower = USER_PROMPT.lower()
                    visual_symptoms_lower = ', '.join(previous_visual_symptoms).lower() if previous_visual_symptoms else ''
                    
                    is_adding_new_symptoms = (
                        previous_visual_symptoms and 
                        USER_PROMPT.strip() and
                        user_symptoms_lower != visual_symptoms_lower and
                        not all(symptom in user_symptoms_lower for symptom in visual_symptoms_lower.split(', ') if symptom)
                    )
                    
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
                    PREVIOUS VISUAL SYMPTOMS DETECTED: {', '.join(previous_visual_symptoms) if previous_visual_symptoms else 'None found in conversation history'}

                    CRITICAL INSTRUCTIONS:
                    
                    1. IF USER IS ADDING NEW TEXT SYMPTOMS to existing visual analysis ({is_adding_new_symptoms}):
                       - Use analyze_combined_symptoms(text_symptoms="{USER_PROMPT}", visual_symptoms="{', '.join(previous_visual_symptoms)}")
                       - This combines NEW user symptoms with PREVIOUS visual symptoms
                    
                    2. IF USER IS REQUESTING INITIAL IMAGE ANALYSIS (no previous visual symptoms):
                       - Use analyze_medical_image() with the provided image data first
                       - Then use analyze_symptoms_direct() with the extracted symptoms
                    
                    3. IF USER IS ONLY PROVIDING TEXT SYMPTOMS (no image involved):
                       - Use analyze_symptoms_direct() with their text input
                    
                    4. ALWAYS call get_disease_description() for each disease you identify
                    5. Use the EXACT text returned by get_disease_description() - do not write your own descriptions

                    Remember: Only combine symptoms when user is adding NEW information to existing visual analysis.
                    """
                else:
                    enhanced_prompt = USER_PROMPT

                if hasattr(st.session_state.assistant, 'chat'):
                    response = st.session_state.assistant.chat(enhanced_prompt)
                else:
                    response = "Assistant not properly initialized"

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "has_image": HAS_IMAGE
                })
                
                st.rerun()

            except Exception as e:
                error_str = str(e).lower()
                
                if ("context_length_exceeded" in error_str or 
                    "maximum context length" in error_str or 
                    "16385 tokens" in error_str or
                    "too many tokens" in error_str):
                    
                    st.error("💬 **Conversation Getting Too Long!**")
                    st.info("""
                    You've reached the conversation memory limit. Here are your options:
                    
                    **🔄 Quick Fix:**
                    - Click "🗑️ Clear Chat & Image" in the sidebar to start fresh
                    
                    **💡 What happened?**
                    - The AI has a memory limit of about 16,000 tokens
                    - Long conversations with images use up this memory quickly
                    - Starting fresh will restore full functionality
                    
                    **✨ Your current image is still uploaded** - you can ask about it again after clearing!
                    """)
                    
                    ERROR_MSG = """🔄 **Conversation Memory Full**

I've reached my conversation memory limit! Please click "🗑️ Clear Chat & Image" in the sidebar to start a fresh conversation.

Don't worry - if you have an image uploaded, it will stay there and you can ask me to analyze it again after clearing the chat."""

                elif "rate limit" in error_str or "quota" in error_str:
                    st.error("⏱️ **API Rate Limit Reached**")
                    st.info("Please wait a moment before trying again. The OpenAI API has usage limits.")
                    ERROR_MSG = "⏱️ I've hit a rate limit. Please wait a moment and try again."

                elif "image" in error_str and ("unsupported" in error_str or "invalid" in error_str):
                    st.error("📸 **Image Processing Issue**")
                    st.info("""
                    There was an issue processing your image. Try:
                    - Uploading a different image format (JPEG or PNG work best)
                    - Taking a new photo if the current one seems corrupted
                    - Describing your symptoms in text instead
                    """)
                    ERROR_MSG = "📸 I had trouble processing your image. Please try uploading a different image or describe your symptoms in text."

                else:
                    st.error("⚠️ **Something Went Wrong**")
                    st.info("There was an unexpected error. Try rephrasing your question or starting a new conversation.")
                    ERROR_MSG = f"⚠️ I encountered an error: {str(e)[:200]}{'...' if len(str(e)) > 200 else ''}"

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": ERROR_MSG,
                    "has_image": False
                })
                st.rerun()


# Render main content
main_content()

# Disclaimer
st.markdown("""
<div class="disclaimer">
⚠️ <strong>Kickstart HealthIQ Disclaimer:</strong> This tool provides general information only and includes AI-powered agentic image analysis. 
Visual analysis is for informational purposes and should not replace professional medical examination. 
Always consult healthcare professionals for proper diagnosis and treatment.
</div>
""", unsafe_allow_html=True)