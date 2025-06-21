"""
Kickstart HealthIQ Tools with Image Analysis
==========================================

Adds image analysis capabilities to the existing medical assistant.
Integrates with OpenAI's Vision API for visual symptom analysis.
"""

import json
import re
import warnings
import base64
from typing import Dict, Optional
from io import BytesIO

import numpy as np
import pandas as pd
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from PIL import Image

def safe_json_dumps(obj):
    """Safely serialize objects to JSON, converting numpy types to Python types."""
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj

    converted_obj = convert_types(obj)
    return json.dumps(converted_obj)

warnings.filterwarnings('ignore')

# ===============================
# GLOBAL VARIABLES FOR TOOLS
# ===============================
_FAISS_SYMPTOM_INDEX = None
_FAISS_SEVERITY_INDEX = None
_DF_DISEASE_PRECAUTIONS = None
_DF_DISEASE_SYMPTOM_DESCRIPTION = None
_DF_DISEASE_SYMPTOM_SEVERITY = None
_VISION_MODEL = None

def set_global_resources(faiss_symptom_index: FAISS,
                         faiss_severity_index: FAISS,
                         df_disease_precautions: pd.DataFrame,
                         df_disease_symptom_description: pd.DataFrame,
                         df_disease_symptom_severity: pd.DataFrame,
                         vision_model: Optional[ChatOpenAI] = None):
    """
    Set global resources for tools to access.
    
    Args:
        faiss_symptom_index: FAISS index for symptom-disease matching
        faiss_severity_index: FAISS index for severity analysis
        df_disease_precautions: DataFrame with disease precautions
        df_disease_symptom_description: DataFrame with disease descriptions
        df_disease_symptom_severity: DataFrame with symptom severity scores
        vision_model: OpenAI vision model for image analysis
    """
    global _FAISS_SYMPTOM_INDEX, _FAISS_SEVERITY_INDEX
    global _DF_DISEASE_PRECAUTIONS
    global _DF_DISEASE_SYMPTOM_DESCRIPTION, _DF_DISEASE_SYMPTOM_SEVERITY
    global _VISION_MODEL

    _FAISS_SYMPTOM_INDEX = faiss_symptom_index
    _FAISS_SEVERITY_INDEX = faiss_severity_index
    _DF_DISEASE_PRECAUTIONS = df_disease_precautions
    _DF_DISEASE_SYMPTOM_DESCRIPTION = df_disease_symptom_description
    _DF_DISEASE_SYMPTOM_SEVERITY = df_disease_symptom_severity
    _VISION_MODEL = vision_model

def encode_image_to_base64(image_data: bytes) -> str:
    """
    Convert image bytes to base64 string for OpenAI Vision API.
    
    Args:
        image_data: Raw image bytes
        
    Returns:
        Base64 encoded string
    """
    return base64.b64encode(image_data).decode('utf-8')

def preprocess_image(image_data: bytes, max_size: tuple = (1024, 1024)) -> bytes:
    """
    Preprocess image for better analysis - resize if too large.
    
    Args:
        image_data: Raw image bytes
        max_size: Maximum dimensions (width, height)
        
    Returns:
        Processed image bytes
    """
    try:
        image = Image.open(BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
            
        # Resize if too large
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
        # Save processed image to bytes
        output = BytesIO()
        image.save(output, format='JPEG', quality=85)
        return output.getvalue()
        
    except Exception as e:
        # If preprocessing fails, return original data
        print(f"Image preprocessing failed: {e}")
        return image_data

@tool
def analyze_medical_image(image_data_b64: str, additional_context: str = "") -> str:
    """
    Analyze a medical image to extract visual symptoms using OpenAI's Vision API.
    
    Args:
        image_data_b64: Base64 encoded image data
        additional_context: Additional context about the image or symptoms
        
    Returns:
        JSON string containing visual analysis results and extracted symptoms
    """
    if not _VISION_MODEL:
        return safe_json_dumps({
            "visual_symptoms": [],
            "analysis": "Vision analysis not available - please configure OpenAI vision model",
            "confidence": 0.0,
            "success": False,
            "message": "Vision model not configured"
        })
    
    if not image_data_b64:
        return safe_json_dumps({
            "visual_symptoms": [],
            "analysis": "No image provided",
            "confidence": 0.0,
            "success": False,
            "message": "No image data provided"
        })
    
    try:
        # Construct the prompt for medical image analysis
        system_prompt = """You are a medical AI assistant specializing in visual diagnosis. 
        Analyze the provided medical image and identify visible symptoms, conditions, or abnormalities.
        
        Focus on:
        1. Skin conditions (rashes, lesions, discoloration, texture changes)
        2. Visible swelling or inflammation
        3. Color changes (jaundice, pallor, cyanosis, erythema)
        4. Visible wounds, burns, or injuries
        5. Eye conditions (redness, discharge, swelling)
        6. Oral/throat conditions if visible
        7. Any other medically relevant visual findings
        
        Provide your response in this JSON format:
        {
            "primary_findings": ["finding1", "finding2"],
            "visual_symptoms": ["symptom1", "symptom2"],
            "possible_conditions": ["condition1", "condition2"],
            "severity_indicators": ["mild/moderate/severe finding"],
            "analysis": "detailed description of what you observe",
            "confidence": 0.8,
            "recommendations": ["seek medical attention if...", "monitor for..."]
        }
        
        Be precise and medical in your terminology. If you're uncertain, indicate that.
        Remember this is for informational purposes only."""
        
        context_addition = f"\nAdditional context provided by user: {additional_context}" if additional_context else ""
        
        # Create message for vision model
        messages = [
            {
                "role": "system", 
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Please analyze this medical image for visible symptoms and conditions.{context_addition}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data_b64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
        
        # Get response from vision model
        response = _VISION_MODEL.invoke(messages)
        response_text = response.content
        
        # Try to parse JSON response
        try:
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                analysis_result = json.loads(json_match.group())
            else:
                # Fallback: create structured response from text
                analysis_result = {
                    "primary_findings": [],
                    "visual_symptoms": [],
                    "possible_conditions": [],
                    "severity_indicators": [],
                    "analysis": response_text,
                    "confidence": 0.7,
                    "recommendations": []
                }
        except json.JSONDecodeError:
            # Fallback response structure
            analysis_result = {
                "primary_findings": [],
                "visual_symptoms": [],
                "possible_conditions": [],
                "severity_indicators": [],
                "analysis": response_text,
                "confidence": 0.6,
                "recommendations": []
            }
        
        # Ensure all required fields exist
        required_fields = ["primary_findings", "visual_symptoms", "possible_conditions", 
                         "severity_indicators", "analysis", "confidence", "recommendations"]
        for field in required_fields:
            if field not in analysis_result:
                analysis_result[field] = [] if field != "analysis" and field != "confidence" else ("" if field == "analysis" else 0.5)
        
        # Add success flag and metadata
        analysis_result.update({
            "success": True,
            "message": "Visual analysis completed successfully",
            "image_analyzed": True,
            "additional_context": additional_context
        })
        
        return safe_json_dumps(analysis_result)
        
    except Exception as e:
        return safe_json_dumps({
            "visual_symptoms": [],
            "analysis": f"Error analyzing image: {str(e)}",
            "confidence": 0.0,
            "success": False,
            "message": f"Vision analysis failed: {str(e)}",
            "image_analyzed": False
        })

@tool
def analyze_symptoms_with_image(user_input: str, visual_analysis: str = "") -> str:
    """
    Enhanced symptom analysis that combines text input with visual analysis results.
    
    Args:
        user_input: User's text description of symptoms
        visual_analysis: JSON string from analyze_medical_image
        
    Returns:
        JSON string containing combined analysis results
    """
    if not _FAISS_SYMPTOM_INDEX:
        return safe_json_dumps({
            "diseases": [],
            "text_symptoms": [],
            "visual_symptoms": [],
            "combined_symptoms": [],
            "success": False,
            "message": "Disease database not available"
        })
    
    # Parse visual analysis if provided
    visual_symptoms = []
    visual_conditions = []
    visual_confidence = 0.0
    
    if visual_analysis:
        try:
            visual_data = json.loads(visual_analysis)
            if visual_data.get("success", False):
                visual_symptoms = visual_data.get("visual_symptoms", [])
                visual_conditions = visual_data.get("possible_conditions", [])
                visual_confidence = visual_data.get("confidence", 0.0)
        except json.JSONDecodeError:
            pass
    
    # Get text-based analysis
    text_analysis = analyze_symptoms_direct(user_input)
    text_data = json.loads(text_analysis)
    
    # Combine symptoms from both sources
    text_symptoms = text_data.get("extracted_symptoms", [])
    combined_symptoms = list(set(text_symptoms + visual_symptoms))
    
    # If we have visual conditions, search for them too
    all_search_terms = []
    if user_input.strip():
        all_search_terms.append(user_input.strip())
    if visual_symptoms:
        all_search_terms.append(", ".join(visual_symptoms))
    if visual_conditions:
        all_search_terms.append(", ".join(visual_conditions))
    
    # Perform enhanced FAISS search with combined terms
    combined_results = []
    seen_diseases = set()
    
    for search_term in all_search_terms:
        if not search_term:
            continue
            
        try:
            results = _FAISS_SYMPTOM_INDEX.similarity_search_with_score(
                search_term, k=10
            )
            
            for doc, sim_score in results:
                sim_score = float(sim_score)
                
                if sim_score <= 1.8:  # Slightly more lenient for combined search
                    disease_name = doc.metadata.get('disease', 'unknown')
                    
                    if disease_name not in seen_diseases:
                        seen_diseases.add(disease_name)
                        
                        # Calculate confidence with visual boost
                        base_confidence = max(0.0, 1.0 - (sim_score / 2.0))
                        
                        # Boost confidence if visual analysis supports this
                        if visual_conditions and any(cond.lower() in disease_name.lower() 
                                                   for cond in visual_conditions):
                            base_confidence = min(1.0, base_confidence + (visual_confidence * 0.3))
                        
                        confidence_percentage = round(base_confidence * 100, 1)
                        
                        result_dict = {
                            'disease': str(disease_name),
                            'similarity_score': sim_score,
                            'confidence_score': base_confidence,
                            'confidence_percentage': confidence_percentage,
                            'matched_content': str(doc.page_content),
                            'search_source': search_term,
                            'visual_supported': visual_conditions and any(
                                cond.lower() in disease_name.lower() for cond in visual_conditions
                            )
                        }
                        
                        combined_results.append(result_dict)
        
        except Exception as e:
            continue
    
    # Sort by confidence
    combined_results.sort(key=lambda x: -x['confidence_score'])
    
    return safe_json_dumps({
        "diseases": combined_results[:6],  # Top 6 diseases
        "text_symptoms": text_symptoms,
        "visual_symptoms": visual_symptoms,
        "combined_symptoms": combined_symptoms,
        "visual_conditions": visual_conditions,
        "visual_confidence": visual_confidence,
        "has_visual_analysis": bool(visual_analysis),
        "total_matches": len(combined_results),
        "search_terms": all_search_terms,
        "success": True,
        "message": f"Combined analysis found {len(combined_results)} potential matches"
    })

@tool  
def complete_multimodal_analysis(user_input: str, image_data_b64: str = "", additional_context: str = "") -> str:
    """
    Complete multimodal medical analysis combining text, image, and severity analysis.
    
    Args:
        user_input: User's text description
        image_data_b64: Base64 encoded image (optional)
        additional_context: Additional context about symptoms or image
        
    Returns:
        JSON string with comprehensive multimodal analysis
    """
    try:
        visual_analysis = ""
        
        # Step 1: Analyze image if provided
        if image_data_b64:
            visual_analysis = analyze_medical_image(image_data_b64, additional_context)
        
        # Step 2: Combined symptom analysis
        if visual_analysis:
            symptom_analysis = analyze_symptoms_with_image(user_input, visual_analysis)
        else:
            symptom_analysis = analyze_symptoms_direct(user_input)
        
        # Parse results
        symptom_data = json.loads(symptom_analysis)
        visual_data = json.loads(visual_analysis) if visual_analysis else {}
        
        # Step 3: Severity analysis with combined symptoms
        combined_symptoms = symptom_data.get("combined_symptoms", 
                                            symptom_data.get("extracted_symptoms", []))
        
        if combined_symptoms:
            severity_analysis = analyze_symptom_severity(", ".join(combined_symptoms))
            severity_data = json.loads(severity_analysis)
        else:
            severity_data = {"success": False, "message": "No symptoms for severity analysis"}
        
        # Step 4: Compile comprehensive result
        result = {
            "user_input": user_input,
            "has_image": bool(image_data_b64),
            "visual_analysis": visual_data,
            "symptom_analysis": symptom_data,
            "severity_analysis": severity_data,
            "diseases": symptom_data.get("diseases", []),
            "all_symptoms": {
                "text_symptoms": symptom_data.get("text_symptoms", []),
                "visual_symptoms": symptom_data.get("visual_symptoms", []),
                "combined_symptoms": combined_symptoms
            },
            "analysis_summary": {
                "total_diseases_found": len(symptom_data.get("diseases", [])),
                "visual_confidence": visual_data.get("confidence", 0.0),
                "has_visual_support": symptom_data.get("has_visual_analysis", False),
                "overall_severity": severity_data.get("overall_severity", 0.0),
                "severity_level": severity_data.get("overall_level", "Unknown")
            },
            "success": True,
            "message": "Complete multimodal analysis performed"
        }
        
        return safe_json_dumps(result)
        
    except Exception as e:
        return safe_json_dumps({
            "user_input": user_input,
            "has_image": bool(image_data_b64),
            "diseases": [],
            "success": False,
            "message": f"Error in multimodal analysis: {str(e)}"
        })

# ===============================
# EXISTING TOOLS (unchanged)
# ===============================

def clean_user_input(user_input: str) -> str:
    """Lightly clean user input to remove conversation noise while preserving medical terms."""
    cleaned = user_input.lower().strip()
    
    conversation_patterns = [
        r'\b(hello|hi|hey)\b',
        r'\b(doctor|doc)\b', 
        r'\b(i think|i believe|i feel like)\b',
        r'\b(what should i do|can you help|please help)\b',
        r'\b(i have been|i am|i\'m)\s+(experiencing|having|feeling)\b'
    ]
    
    for pattern in conversation_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', cleaned)
    
    return cleaned

@tool
def analyze_symptoms_direct(user_input: str) -> str:
    """
    Direct symptom analysis using FAISS without explicit symptom extraction.
    """
    if not _FAISS_SYMPTOM_INDEX:
        return safe_json_dumps({
            "diseases": [],
            "extracted_symptoms": [],
            "success": False,
            "message": "Disease database not available"
        })

    if not user_input.strip():
        return safe_json_dumps({
            "diseases": [],
            "extracted_symptoms": [],
            "success": False,
            "message": "No input provided"
        })

    cleaned_input = clean_user_input(user_input)

    try:
        results = _FAISS_SYMPTOM_INDEX.similarity_search_with_score(
            cleaned_input, k=15
        )

        disease_results = []
        found_symptoms = set()

        for doc, sim_score in results:
            sim_score = float(sim_score)

            if sim_score <= 1.5:
                disease_name = doc.metadata.get('disease', 'unknown')
                confidence_score = max(0.0, 1.0 - (sim_score / 2.0))
                confidence_percentage = round(confidence_score * 100, 1)
                
                matched_content = doc.page_content.lower()
                content_symptoms = [s.strip() for s in re.split(r'[,;]', matched_content) if s.strip()]
                found_symptoms.update(content_symptoms)

                result_dict = {
                    'disease': str(disease_name),
                    'similarity_score': sim_score,
                    'confidence_score': confidence_score,
                    'confidence_percentage': confidence_percentage,
                    'matched_symptoms': content_symptoms,
                    'matched_content': str(doc.page_content),
                    'all_metadata': doc.metadata
                }

                disease_results.append(result_dict)

        disease_results.sort(key=lambda x: -x['confidence_score'])

        seen_diseases = set()
        unique_results = []
        for result in disease_results:
            disease = result['disease']
            if disease not in seen_diseases:
                seen_diseases.add(disease)
                unique_results.append(result)

        extracted_symptoms = list(found_symptoms)

        return safe_json_dumps({
            "diseases": unique_results[:5],
            "extracted_symptoms": extracted_symptoms,
            "total_matches": len(results),
            "good_matches": len(disease_results),
            "search_query": cleaned_input,
            "original_input": user_input,
            "success": True,
            "message": f"Found {len(unique_results[:5])} diseases via FAISS"
        })

    except Exception as e:
        return safe_json_dumps({
            "diseases": [],
            "extracted_symptoms": [],
            "success": False,
            "message": f"Error in direct symptom analysis: {e}"
        })

@tool
def analyze_symptom_severity(symptoms_list: str) -> str:
    """Analyzes the severity of symptoms using FAISS similarity and severity database."""
    if not _FAISS_SEVERITY_INDEX or _DF_DISEASE_SYMPTOM_SEVERITY is None:
        return json.dumps({
            "severity_scores": {},
            "overall_severity": 4.0,
            "overall_level": "Moderate",
            "success": False,
            "message": "Severity database not available"
        })

    symptoms = [s.strip() for s in symptoms_list.split(',') if s.strip()]

    if not symptoms:
        return json.dumps({
            "severity_scores": {},
            "overall_severity": 4.0,
            "overall_level": "Moderate",
            "success": False,
            "message": "No symptoms provided"
        })

    def classify_severity(severity_score: float) -> str:
        if severity_score <= 2:
            return "Low"
        elif severity_score <= 3:
            return "Mild"
        elif severity_score <= 4:
            return "Moderate"
        elif severity_score <= 5:
            return "High"
        elif severity_score <= 6:
            return "Very High"
        else:
            return "Severe"

    severity_scores = {}

    for symptom in symptoms:
        try:
            results = _FAISS_SEVERITY_INDEX.similarity_search_with_score(symptom, k=1)

            if results:
                matched_doc = results[0][0]
                matched_symptom = matched_doc.page_content.strip()

                exact_match = _DF_DISEASE_SYMPTOM_SEVERITY[
                    (_DF_DISEASE_SYMPTOM_SEVERITY['symptom'].str.lower().str.strip() == matched_symptom.lower())
                ]

                if len(exact_match) > 0:
                    severity_scores[symptom] = float(exact_match['severity'].iloc[0])
                else:
                    partial_matches = _DF_DISEASE_SYMPTOM_SEVERITY[
                        (_DF_DISEASE_SYMPTOM_SEVERITY['symptom'].str.lower().str.contains(matched_symptom.lower(), na=False))
                    ]

                    if len(partial_matches) > 0:
                        avg_severity = partial_matches['severity'].mean()
                        severity_scores[symptom] = float(avg_severity)
                    else:
                        severity_scores[symptom] = 4.0
            else:
                severity_scores[symptom] = 4.0

        except Exception:
            severity_scores[symptom] = 4.0

    if severity_scores:
        valid_scores = [score for score in severity_scores.values() if score is not None]
        if valid_scores:
            avg_severity = sum(valid_scores) / len(valid_scores)
            overall_level = classify_severity(avg_severity)
        else:
            avg_severity = 4.0
            overall_level = "Moderate"
    else:
        avg_severity = 4.0
        overall_level = "Moderate"

    return json.dumps({
        "severity_scores": {k: float(v) for k, v in severity_scores.items()},
        "overall_severity": float(avg_severity),
        "overall_level": overall_level,
        "success": True,
        "message": f"Analyzed severity for {len(symptoms)} symptoms"
    })

@tool
def get_disease_precautions(disease_name: str) -> str:
    """Gets precautions and preventive measures for a specific disease."""
    if _DF_DISEASE_PRECAUTIONS is None:
        return json.dumps({
            "precautions": "Precautions database not available.",
            "success": False,
            "message": "Database not available"
        })

    def format_text_properly(text: str) -> str:
        if not text or not text.strip():
            return text

        medical_acronyms = {
            'hiv': 'HIV', 'aids': 'AIDS', 'copd': 'COPD', 'uti': 'UTI',
            'std': 'STD', 'sti': 'STI', 'tb': 'TB', 'ct': 'CT', 'mri': 'MRI',
            'ecg': 'ECG', 'ekg': 'EKG', 'bp': 'BP', 'iv': 'IV', 'er': 'ER',
            'icu': 'ICU', 'or': 'OR', 'covid': 'COVID', 'sars': 'SARS',
            'mrsa': 'MRSA', 'dna': 'DNA', 'rna': 'RNA'
        }

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        formatted_sentences = []

        for sentence in sentences:
            if not sentence.strip():
                continue

            sentence = sentence.strip()
            if sentence:
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()

            words = sentence.split()
            formatted_words = []

            for word in words:
                clean_word = re.sub(r'[^\w]', '', word.lower())
                if clean_word in medical_acronyms:
                    formatted_word = re.sub(re.escape(clean_word), medical_acronyms[clean_word], word, flags=re.IGNORECASE)
                    formatted_words.append(formatted_word)
                else:
                    formatted_words.append(word)

            formatted_sentence = ' '.join(formatted_words)

            if formatted_sentence and not formatted_sentence[-1] in '.!?':
                formatted_sentence += '.'

            formatted_sentences.append(formatted_sentence)

        return ' '.join(formatted_sentences)

    try:
        precautions = _DF_DISEASE_PRECAUTIONS[
            (_DF_DISEASE_PRECAUTIONS['disease'].str.lower() == disease_name.lower())
        ]['precautions'].iloc[0]

        formatted_precautions = format_text_properly(precautions)

        return json.dumps({
            "precautions": formatted_precautions,
            "success": True,
            "message": f"Found precautions for {disease_name}"
        })

    except (IndexError, KeyError):
        return json.dumps({
            "precautions": "No specific precautions found for this condition.",
            "success": False,
            "message": f"No precautions found for {disease_name}"
        })

@tool
def get_disease_description(disease_name: str) -> str:
    """Gets detailed description for a specific disease."""
    if _DF_DISEASE_SYMPTOM_DESCRIPTION is None:
        return json.dumps({
            "description": "Description database not available.",
            "success": False,
            "message": "Database not available"
        })

    def format_text_properly(text: str) -> str:
        if not text or not text.strip():
            return text

        medical_acronyms = {
            'hiv': 'HIV', 'aids': 'AIDS', 'copd': 'COPD', 'uti': 'UTI',
            'std': 'STD', 'sti': 'STI', 'tb': 'TB', 'ct': 'CT', 'mri': 'MRI',
            'ecg': 'ECG', 'ekg': 'EKG', 'bp': 'BP', 'iv': 'IV', 'er': 'ER',
            'icu': 'ICU', 'or': 'OR', 'covid': 'COVID', 'sars': 'SARS',
            'mrsa': 'MRSA', 'dna': 'DNA', 'rna': 'RNA'
        }

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        formatted_sentences = []

        for sentence in sentences:
            if not sentence.strip():
                continue

            sentence = sentence.strip()
            if sentence:
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()

            words = sentence.split()
            formatted_words = []

            for word in words:
                clean_word = re.sub(r'[^\w]', '', word.lower())
                if clean_word in medical_acronyms:
                    formatted_word = re.sub(re.escape(clean_word), medical_acronyms[clean_word], word, flags=re.IGNORECASE)
                    formatted_words.append(formatted_word)
                else:
                    formatted_words.append(word)

            formatted_sentence = ' '.join(formatted_words)

            if formatted_sentence and not formatted_sentence[-1] in '.!?':
                formatted_sentence += '.'

            formatted_sentences.append(formatted_sentence)

        return ' '.join(formatted_sentences)

    try:
        disease_col = None
        desc_col = None

        for col in _DF_DISEASE_SYMPTOM_DESCRIPTION.columns:
            if col.lower() in ['disease', 'diseases']:
                disease_col = col
                break

        for col in _DF_DISEASE_SYMPTOM_DESCRIPTION.columns:
            if col.lower() in ['description', 'desc', 'descriptions']:
                desc_col = col
                break

        if disease_col is None or desc_col is None:
            return json.dumps({
                "description": f"Unable to find appropriate columns. Available: {list(_DF_DISEASE_SYMPTOM_DESCRIPTION.columns)}",
                "success": False,
                "message": "Column structure issue"
            })

        description = _DF_DISEASE_SYMPTOM_DESCRIPTION[
            (_DF_DISEASE_SYMPTOM_DESCRIPTION[disease_col].str.lower() == disease_name.lower())
        ][desc_col].iloc[0]

        formatted_description = format_text_properly(description)

        return json.dumps({
            "description": formatted_description,
            "success": True,
            "message": f"Found description for {disease_name}"
        })

    except (IndexError, KeyError):
        return json.dumps({
            "description": f"No detailed description found for {disease_name}.",
            "success": False,
            "message": f"No description found for {disease_name}"
        })

# ===============================
# TOOL UTILITIES
# ===============================

def get_enhanced_medical_tools():
    """
    Get all Kickstart HealthIQ tools including image analysis capabilities.
    
    Returns:
        List of medical tools with image analysis
    """
    return [
        # New multimodal tools
        analyze_medical_image,              # NEW: Image analysis
        analyze_symptoms_with_image,        # NEW: Combined text+image analysis  
        complete_multimodal_analysis,       # NEW: Complete multimodal analysis
        
        # Enhanced existing tools
        analyze_symptoms_direct,            # Enhanced with better integration
        analyze_symptom_severity,           # Existing: For compatibility
        get_disease_precautions,           # Existing: Unchanged
        get_disease_description            # Existing: Unchanged
    ]

def check_resources_available() -> Dict[str, bool]:
    """Check which resources are available including vision model."""
    return {
        'faiss_symptom_index': _FAISS_SYMPTOM_INDEX is not None,
        'faiss_severity_index': _FAISS_SEVERITY_INDEX is not None,
        'df_disease_precautions': _DF_DISEASE_PRECAUTIONS is not None,
        'df_disease_symptom_description': _DF_DISEASE_SYMPTOM_DESCRIPTION is not None,
        'df_disease_symptom_severity': _DF_DISEASE_SYMPTOM_SEVERITY is not None,
        'vision_model': _VISION_MODEL is not None
    }

def test_enhanced_tools():
    """Test Kickstart HealthIQ tools including image analysis."""
    print("=== TESTING Kickstart HealthIQ MEDICAL TOOLS ===")
    
    resources = check_resources_available()
    print(f"Resources available: {resources}")
    
    if not all(list(resources.values())[:-1]):  # All except vision model
        print("❌ Core resources not available. Please call set_global_resources() first.")
        return False
    
    # Test text-only analysis
    print("\n--- Testing text analysis ---")
    try:
        result = analyze_symptoms_direct("I have a fever and headache")
        data = json.loads(result)
        if data.get("success"):
            print("✅ Text analysis working")
        else:
            print(f"⚠️ Text analysis failed: {data.get('message')}")
    except Exception as e:
        print(f"❌ Text analysis error: {e}")
    
    # Test image analysis (if vision model available)
    if resources['vision_model']:
        print("\n--- Testing image analysis ---")
        try:
            # Test with empty image (should handle gracefully)
            result = analyze_medical_image("", "test context")
            data = json.loads(result)
            if not data.get("success") and "No image data" in data.get("message", ""):
                print("✅ Image analysis handles empty input correctly")
            else:
                print("⚠️ Image analysis didn't handle empty input as expected")
        except Exception as e:
            print(f"❌ Image analysis error: {e}")
    else:
        print("\n⚠️ Vision model not configured - image analysis not available")
    
    print("\n✅ Kickstart HealthIQ tools test completed!")
    return True

if __name__ == "__main__":
    print("Kickstart HealthIQ Tools with Image Analysis loaded successfully!")
    print("New functions:")
    print("- analyze_medical_image(image_data_b64, additional_context)")  
    print("- analyze_symptoms_with_image(user_input, visual_analysis)")
    print("- complete_multimodal_analysis(user_input, image_data_b64, additional_context)")
    print("Test function: test_enhanced_tools()")