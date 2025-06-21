"""
Streamlined Vision Tools for Medical Image Analysis
==================================================

Focused on extracting visible symptoms that can be processed by the main medical tools.
"""

import json
import re
import warnings
from typing import Optional

from langchain.tools import tool
from langchain_openai import ChatOpenAI

warnings.filterwarnings('ignore')

# Global vision model
_VISION_MODEL = None

def set_vision_model(vision_model: Optional[ChatOpenAI]):
    """Set the global vision model for image analysis tools."""
    global _VISION_MODEL
    _VISION_MODEL = vision_model

def get_vision_model():
    """Get the current vision model."""
    return _VISION_MODEL

@tool
def analyze_medical_image(image_base64: str, additional_context: str = "") -> str:
    """
    Analyze a medical image to extract visible symptoms that can be used for medical diagnosis.
    Returns focused symptom list for further medical analysis.
    
    Args:
        image_base64: Base64 encoded image data
        additional_context: Additional context about the image or symptoms
        
    Returns:
        JSON string containing extracted visible symptoms for medical analysis
    """
    print(f"🔍 VISION TOOL CALLED: analyze_medical_image")
    print(f"📷 Image data length: {len(image_base64) if image_base64 else 0}")
    print(f"📝 Context: {additional_context}")
    
    if not _VISION_MODEL:
        result = {
            "visible_symptoms": [],
            "analysis_summary": "Vision analysis not available - OpenAI vision model not configured",
            "success": False,
            "message": "Vision model not available"
        }
        print(f"❌ VISION TOOL RESULT: {result}")
        return json.dumps(result)
    
    if not image_base64 or len(image_base64) < 100:
        result = {
            "visible_symptoms": [],
            "analysis_summary": "No valid image data provided for analysis",
            "success": False,
            "message": "No image data provided"
        }
        print(f"❌ VISION TOOL RESULT: {result}")
        return json.dumps(result)
    
    try:
        print("🚀 Calling OpenAI Vision API...")
        
        # Focused prompt to extract just visible symptoms
        focused_prompt = """You are a medical AI analyzing an image to extract VISIBLE SYMPTOMS ONLY.

Your task: Identify visible medical symptoms that can be seen in this image.

Focus ONLY on:
- Skin conditions (rashes, lesions, discoloration, swelling)
- Visible inflammation or irritation  
- Color changes (redness, pallor, yellowing)
- Visible wounds, marks, or abnormalities
- Texture changes in skin or visible areas

Respond with ONLY a simple list of visible symptoms using medical terminology.

Example good responses:
- "erythematous papules, vesicular lesions, localized swelling"
- "macular rash, erythema, skin discoloration"  
- "inflammatory papules, central umbilication"

Do NOT include:
- Detailed descriptions
- Possible diagnoses
- Treatment recommendations
- Severity assessments

Just provide the visible symptoms as a comma-separated list using precise medical terms."""

        context_addition = f"\nPatient context: {additional_context}" if additional_context else ""
        full_prompt = focused_prompt + context_addition
        
        # Create message for vision model
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": full_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
        
        # Call OpenAI Vision API
        response = _VISION_MODEL.invoke(messages)
        response_text = response.content.strip()
        
        print(f"✅ OpenAI Vision Response: {response_text}")
        
        # Extract symptoms from response
        # Clean up the response to get just the symptoms
        cleaned_response = response_text.lower()
        
        # Remove common prefixes and clean up
        prefixes_to_remove = [
            "visible symptoms:",
            "symptoms:",
            "i can see:",
            "the image shows:",
            "visible:",
            "observed:"
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned_response.startswith(prefix):
                cleaned_response = cleaned_response[len(prefix):].strip()
        
        # Split into individual symptoms
        symptoms = []
        if cleaned_response:
            # Split by common delimiters
            raw_symptoms = re.split(r'[,;]|and(?=\s)', cleaned_response)
            
            for symptom in raw_symptoms:
                symptom = symptom.strip()
                # Remove bullet points, numbers, and extra formatting
                symptom = re.sub(r'^[-•*\d.\s]+', '', symptom)
                symptom = symptom.strip()
                
                if symptom and len(symptom) > 2:  # Filter out very short items
                    symptoms.append(symptom)
        
        # Limit to most relevant symptoms
        symptoms = symptoms[:8]  # Keep top 8 symptoms max
        
        result = {
            "visible_symptoms": symptoms,
            "analysis_summary": f"Extracted {len(symptoms)} visible symptoms from image analysis",
            "raw_response": response_text,
            "success": True,
            "message": f"Successfully extracted {len(symptoms)} visible symptoms"
        }
        
        print(f"✅ VISION TOOL SUCCESS: Found symptoms: {symptoms}")
        
        return json.dumps(result)
        
    except Exception as e:
        result = {
            "visible_symptoms": [],
            "analysis_summary": f"Error during image analysis: {str(e)}",
            "success": False,
            "message": f"Image analysis failed: {str(e)}"
        }
        print(f"❌ VISION TOOL ERROR: {result}")
        return json.dumps(result)

def get_vision_tools():
    """
    Get streamlined vision tools for the medical assistant.
    
    Returns:
        List of vision tools
    """
    return [
        analyze_medical_image
    ]

def test_vision_tools():
    """Test vision tools with debug output."""
    print("=== TESTING STREAMLINED VISION TOOLS ===")
    
    # Test with mock data
    print("\n1. Testing analyze_medical_image with mock data...")
    result1 = analyze_medical_image("mock_base64_data", "test context")
    print(f"Result 1: {result1}")
    
    print("\n2. Checking vision model availability...")
    print(f"Vision model available: {_VISION_MODEL is not None}")
    
    print("\n=== STREAMLINED VISION TOOLS TEST COMPLETE ===")

if __name__ == "__main__":
    print("Streamlined Vision Tools for Medical Image Analysis loaded!")
    print("Available tools:")
    for tool in get_vision_tools():
        print(f"- {tool.name}: {tool.description}")
    print("\nTest function: test_vision_tools()")