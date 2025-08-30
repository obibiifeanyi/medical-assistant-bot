import streamlit as st
import time
import base64
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="MEDORA - Medical AI Consultant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark futuristic design with glassmorphism
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

.stApp {
    background: hsl(240, 10%, 3.9%);
    font-family: 'Inter', sans-serif;
    overflow-x: hidden;
    padding: 0 !important;
    margin: 0 !important;
}

/* Hide Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}

/* Override Streamlit default styles */
.block-container {
    padding: 0 !important;
    margin: 0 !important;
    max-width: none !important;
}

.main .block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}

.stApp > div:first-child {
    padding: 0 !important;
}

.main {
    padding: 0 !important;
}

/* Background gradients and atmosphere */
.main-container {
    position: relative;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2rem;
    background: 
        radial-gradient(circle at center, hsla(271, 91%, 40%, 0.4) 0%, transparent 50%),
        linear-gradient(135deg, hsla(271, 91%, 50%, 0.2) 0%, hsla(217, 91%, 60%, 0.2) 100%),
        hsl(240, 10%, 3.9%);
    animation: backgroundPulse 8s ease-in-out infinite;
}

@keyframes backgroundPulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.95; }
}

/* Content container */
.content-wrapper {
    max-width: 1200px;
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 3rem;
    z-index: 10;
    position: relative;
}

/* Header section */
.hero-header {
    text-align: center;
    margin-bottom: 2rem;
    animation: fadeInUp 1s ease-out;
}

.main-title {
    font-size: clamp(3rem, 8vw, 5rem);
    font-weight: 800;
    background: linear-gradient(135deg, hsl(217, 91%, 60%) 0%, hsl(271, 91%, 60%) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
    letter-spacing: -0.02em;
    line-height: 1.1;
}

.subtitle-container {
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.animated-subtitle {
    font-size: clamp(1.2rem, 3vw, 1.5rem);
    color: hsl(0, 0%, 85%);
    font-weight: 400;
    text-align: center;
    min-height: 2em;
    display: flex;
    align-items: center;
}

/* AI Avatar */
.avatar-container {
    position: relative;
    margin: 2rem 0;
    animation: float 6s ease-in-out infinite;
}

.avatar-circle {
    width: 256px;
    height: 256px;
    border-radius: 50%;
    background: 
        radial-gradient(circle at 30% 30%, hsla(271, 91%, 60%, 0.8), hsla(217, 91%, 60%, 0.6)),
        linear-gradient(135deg, hsla(271, 91%, 40%, 0.9), hsla(217, 91%, 50%, 0.7));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 4rem;
    color: white;
    box-shadow: 
        0 0 60px hsla(271, 91%, 60%, 0.4),
        0 0 120px hsla(217, 91%, 60%, 0.3),
        inset 0 0 60px hsla(0, 0%, 100%, 0.1);
    transition: all 0.3s ease;
    cursor: pointer;
    position: relative;
    overflow: hidden;
}

.avatar-circle::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: conic-gradient(from 0deg, transparent, hsla(271, 91%, 60%, 0.3), transparent);
    animation: rotate 4s linear infinite;
    z-index: -1;
}

.avatar-circle:hover {
    transform: scale(1.05);
    filter: brightness(1.1);
    box-shadow: 
        0 0 80px hsla(271, 91%, 60%, 0.6),
        0 0 160px hsla(217, 91%, 60%, 0.4),
        inset 0 0 60px hsla(0, 0%, 100%, 0.2);
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-20px); }
}

@keyframes rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

/* Glassmorphism panel */
.glass-panel {
    background: rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    padding: 3rem;
    max-width: 500px;
    width: 100%;
    box-shadow: 
        0 8px 32px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
    animation: fadeInUp 1s ease-out 0.3s both;
}

/* Form elements */
.step-container {
    text-align: center;
}

.step-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: hsl(0, 0%, 98%);
    margin-bottom: 1.5rem;
}

.step-description {
    color: hsl(0, 0%, 75%);
    margin-bottom: 2rem;
    line-height: 1.6;
}

/* Input styling */
.stTextInput > div > div > input {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 12px !important;
    color: white !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
    transition: all 0.3s ease !important;
}

.stTextInput > div > div > input:focus {
    border-color: hsl(217, 91%, 60%) !important;
    box-shadow: 0 0 0 2px hsla(217, 91%, 60%, 0.2) !important;
    outline: none !important;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, hsl(217, 91%, 60%) 0%, hsl(271, 91%, 60%) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    margin: 0.5rem 0 !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px hsla(217, 91%, 60%, 0.4) !important;
    filter: brightness(1.1) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

/* Secondary button */
.secondary-btn {
    background: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}

.secondary-btn:hover {
    background: rgba(255, 255, 255, 0.15) !important;
    box-shadow: 0 8px 25px rgba(255, 255, 255, 0.1) !important;
}

/* Terms and pricing */
.terms-container {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1.5rem 0;
    text-align: left;
}

.pricing-highlight {
    font-size: 1.25rem;
    font-weight: 600;
    color: hsl(217, 91%, 60%);
    margin-bottom: 1rem;
}

.disclaimer {
    font-size: 0.9rem;
    color: hsl(0, 0%, 70%);
    line-height: 1.5;
    margin-bottom: 1rem;
}

/* Success state */
.success-container {
    text-align: center;
    padding: 2rem;
}

.success-icon {
    font-size: 4rem;
    color: hsl(142, 76%, 36%);
    margin-bottom: 1rem;
}

/* Loading spinner */
.loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-radius: 50%;
    border-top-color: hsl(217, 91%, 60%);
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Responsive design */
@media (max-width: 768px) {
    .main-container {
        padding: 1rem;
    }
    
    .glass-panel {
        padding: 2rem;
        margin: 1rem;
    }
    
    .avatar-circle {
        width: 200px;
        height: 200px;
        font-size: 3rem;
    }
}

/* Brand footer */
.brand-footer {
    text-align: center;
    margin-top: 2rem;
    color: hsl(0, 0%, 60%);
    font-size: 0.9rem;
}

.brand-link {
    color: hsl(217, 91%, 60%);
    text-decoration: none;
    font-weight: 500;
}

.brand-link:hover {
    color: hsl(271, 91%, 60%);
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'phone' not in st.session_state:
    st.session_state.phone = ''
if 'terms_accepted' not in st.session_state:
    st.session_state.terms_accepted = False
if 'payment_processed' not in st.session_state:
    st.session_state.payment_processed = False
if 'typing_index' not in st.session_state:
    st.session_state.typing_index = 0
if 'current_text' not in st.session_state:
    st.session_state.current_text = ''
if 'typing_complete' not in st.session_state:
    st.session_state.typing_complete = False

# Typing animation texts
typing_texts = [
    "Get expert medical advice instantly",
    "Available 24/7 for consultation", 
    "Specialized in medical diagnosis",
    "Built by Lumora Tech Company"
]

def typing_animation():
    """Handle typing animation for subtitle"""
    if st.session_state.typing_index < len(typing_texts):
        current_full_text = typing_texts[st.session_state.typing_index]
        
        if len(st.session_state.current_text) < len(current_full_text):
            st.session_state.current_text += current_full_text[len(st.session_state.current_text)]
        else:
            time.sleep(2)  # Pause at end of text
            st.session_state.typing_index = (st.session_state.typing_index + 1) % len(typing_texts)
            st.session_state.current_text = ''
    
    return st.session_state.current_text

def render_step_1():
    """Phone input step"""
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="step-title">📱 Enter Your Phone Number</h2>', unsafe_allow_html=True)
    st.markdown('<p class="step-description">We\'ll call you to start your medical consultation</p>', unsafe_allow_html=True)
    
    phone = st.text_input(
        "Phone Number",
        placeholder="+1 (555) 123-4567",
        key="phone_input",
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Continue", key="continue_btn"):
            if phone:
                st.session_state.phone = phone
                st.session_state.step = 2
                st.rerun()
            else:
                st.error("Please enter a valid phone number")
    
    with col2:
        if st.button("Cancel", key="cancel_btn"):
            st.session_state.step = 1
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_step_2():
    """Terms and agreement step"""
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="step-title">📋 Terms & Consultation Fee</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="terms-container">
        <div class="pricing-highlight">💰 Consultation Fee: $0.99/minute</div>
        
        <div class="disclaimer">
            <strong>Important Medical Disclaimer:</strong><br>
            • This service provides medical information and guidance, not formal medical diagnosis<br>
            • Always consult with a licensed healthcare provider for serious medical concerns<br>
            • In case of emergency, call 911 or go to your nearest emergency room<br>
            • This AI consultation supplements but does not replace professional medical care<br>
            • Minimum consultation time: 5 minutes ($4.95)<br>
            • You will be charged only for the actual consultation time
        </div>
        
        <div class="disclaimer">
            <strong>Privacy & Data:</strong><br>
            • Your medical information is encrypted and secure<br>
            • We comply with HIPAA privacy regulations<br>
            • Consultation records are stored securely for your reference
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Accept & Continue", key="accept_btn"):
            st.session_state.terms_accepted = True
            st.session_state.step = 3
            st.rerun()
    
    with col2:
        if st.button("Decline", key="decline_btn"):
            st.session_state.step = 1
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_step_3():
    """Payment step"""
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="step-title">💳 Secure Payment</h2>', unsafe_allow_html=True)
    st.markdown('<p class="step-description">Secure payment processing for your medical consultation</p>', unsafe_allow_html=True)
    
    # Simulated payment form
    st.text_input("Card Number", placeholder="1234 5678 9012 3456", key="card_num")
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Expiry", placeholder="MM/YY", key="expiry")
    with col2:
        st.text_input("CVV", placeholder="123", key="cvv")
    
    st.text_input("Cardholder Name", placeholder="John Doe", key="cardholder")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Process Payment", key="pay_btn"):
            # Simulate payment processing
            with st.spinner("Processing payment..."):
                time.sleep(2)
                st.session_state.payment_processed = True
                st.session_state.step = 4
                st.rerun()
    
    with col2:
        if st.button("Back", key="back_btn"):
            st.session_state.step = 2
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_step_4():
    """Success step"""
    st.markdown('<div class="success-container">', unsafe_allow_html=True)
    st.markdown('<div class="success-icon">✅</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="step-title">Consultation Confirmed!</h2>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <p class="step-description">
        Thank you! We'll call you at <strong>{st.session_state.phone}</strong> within the next 2 minutes to begin your medical consultation.
    </p>
    
    <div class="terms-container">
        <strong>What to expect:</strong><br>
        • Our AI medical consultant will call you shortly<br>
        • Have your medical history and current symptoms ready<br>
        • The consultation will be recorded for your records<br>
        • You'll receive a summary via email after the call<br>
        • Billing starts when the consultation begins
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Start New Consultation", key="new_consultation"):
        # Reset session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main app layout
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<div class="content-wrapper">', unsafe_allow_html=True)

# Header section
st.markdown('<div class="hero-header">', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">MEDORA</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-title" style="font-size: clamp(1.5rem, 4vw, 2.5rem); margin-top: -1rem;">Medical AI Consultant</p>', unsafe_allow_html=True)

# Animated subtitle
st.markdown('<div class="subtitle-container">', unsafe_allow_html=True)
current_text = typing_animation()
st.markdown(f'<div class="animated-subtitle">{current_text}<span style="animation: blink 1s infinite;">|</span></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# AI Avatar
st.markdown('<div class="avatar-container">', unsafe_allow_html=True)
st.markdown('<div class="avatar-circle">🏥</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Glass panel with steps
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)

if st.session_state.step == 1:
    render_step_1()
elif st.session_state.step == 2:
    render_step_2()
elif st.session_state.step == 3:
    render_step_3()
elif st.session_state.step == 4:
    render_step_4()

st.markdown('</div>', unsafe_allow_html=True)

# Brand footer
st.markdown("""
<div class="brand-footer">
    Built with ❤️ by <a href="#" class="brand-link">Lumora Tech Company</a>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Auto-refresh for typing animation
time.sleep(0.1)
st.rerun()