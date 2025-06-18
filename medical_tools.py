"""
Streamlined Medical Tools - Direct FAISS Approach
=================================================

Works directly with FAISS similarity search.

"""

import json
import re
import warnings
from typing import Dict

import numpy as np
import pandas as pd
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
import nltk


def safe_json_dumps(obj):
    """Safely serialize objects to JSON, converting numpy types to
    Python types."""
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


def set_global_resources(faiss_symptom_index: FAISS,
                         faiss_severity_index: FAISS,
                         df_disease_precautions: pd.DataFrame,
                         df_disease_symptom_description: pd.DataFrame,
                         df_disease_symptom_severity: pd.DataFrame):
    """
    Set global resources for tools to access.

    Args:
        faiss_symptom_index: FAISS index for symptom-disease matching
        faiss_severity_index: FAISS index for severity analysis
        df_disease_precautions: DataFrame with disease precautions
        df_disease_symptom_description: DataFrame with disease descriptions
        df_disease_symptom_severity: DataFrame with symptom severity scores
    """
    global _FAISS_SYMPTOM_INDEX, _FAISS_SEVERITY_INDEX
    global _DF_DISEASE_PRECAUTIONS
    global _DF_DISEASE_SYMPTOM_DESCRIPTION, _DF_DISEASE_SYMPTOM_SEVERITY

    _FAISS_SYMPTOM_INDEX = faiss_symptom_index
    _FAISS_SEVERITY_INDEX = faiss_severity_index
    _DF_DISEASE_PRECAUTIONS = df_disease_precautions
    _DF_DISEASE_SYMPTOM_DESCRIPTION = df_disease_symptom_description
    _DF_DISEASE_SYMPTOM_SEVERITY = df_disease_symptom_severity


def check_resources_available() -> Dict[str, bool]:
    """Check which resources are available."""
    return {
        'faiss_symptom_index': _FAISS_SYMPTOM_INDEX is not None,
        'faiss_severity_index': _FAISS_SEVERITY_INDEX is not None,
        'df_disease_precautions': _DF_DISEASE_PRECAUTIONS is not None,
        'df_disease_symptom_description':
            _DF_DISEASE_SYMPTOM_DESCRIPTION is not None,
        'df_disease_symptom_severity':
            _DF_DISEASE_SYMPTOM_SEVERITY is not None
    }


def clean_user_input(user_input: str) -> str:
    """
    Lightly clean user input to remove conversation noise while preserving
    medical terms.
    """
    # Convert to lowercase
    cleaned = user_input.lower().strip()

    # Remove common conversation starters but keep medical content
    conversation_patterns = [
        r'\b(hello|hi|hey)\b',
        r'\b(doctor|doc)\b',
        r'\b(i think|i believe|i feel like)\b',
        r'\b(what should i do|can you help|please help)\b',
        r'\b(i have been|i am|i\'m)\s+(experiencing|having|feeling)\b'
    ]

    for pattern in conversation_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # Remove leading/trailing punctuation but keep internal punctuation
    cleaned = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', cleaned)

    return cleaned


@tool
def analyze_symptoms_direct(user_input: str) -> str:
    """
    Direct symptom analysis using FAISS without explicit symptom extraction.
    This is the main function that replaces both extract_symptoms and
    find_probable_diseases.

    Args:
        user_input: Raw user input describing symptoms

    Returns:
        JSON string containing diseases, extracted symptoms, and metadata
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

    # Light cleaning of input
    cleaned_input = clean_user_input(user_input)

    try:
        # Search FAISS directly - it handles the complexity for us
        results = _FAISS_SYMPTOM_INDEX.similarity_search_with_score(
            cleaned_input, k=15
        )

        # Process FAISS results
        disease_results = []
        found_symptoms = set()  # Track symptoms found in matches

        for doc, sim_score in results:
            sim_score = float(sim_score)

            # Good similarity threshold (adjust based on your data)
            if sim_score <= 1.5:  # Lower scores = better matches
                disease_name = doc.metadata.get('disease', 'unknown')

                # Calculate confidence from similarity score
                # Convert similarity to confidence (0-100%)
                confidence_score = max(0.0, 1.0 - (sim_score / 2.0))
                confidence_percentage = round(confidence_score * 100, 1)

                # Extract symptoms from the matched content
                matched_content = doc.page_content.lower()
                content_symptoms = [s.strip() for s in
                                    re.split(r'[,;]', matched_content)
                                    if s.strip()]
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

        # Sort by confidence (highest first)
        disease_results.sort(key=lambda x: -x['confidence_score'])

        # Remove duplicate diseases (keep highest confidence)
        seen_diseases = set()
        unique_results = []
        for result in disease_results:
            disease = result['disease']
            if disease not in seen_diseases:
                seen_diseases.add(disease)
                unique_results.append(result)

        # Extract unique symptoms from all matches
        extracted_symptoms = list(found_symptoms)

        return safe_json_dumps({
            "diseases": unique_results[:5],  # Top 5 diseases
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
def analyze_symptom_severity_from_matches(analysis_result: str) -> str:
    """
    Analyze severity using symptoms extracted from FAISS matches.

    Args:
        analysis_result: JSON string from analyze_symptoms_direct

    Returns:
        JSON string containing severity analysis
    """
    try:
        # Parse the analysis result
        data = json.loads(analysis_result)

        if not data.get("success", False):
            return json.dumps({
                "severity_scores": {},
                "overall_severity": 4.0,
                "overall_level": "Moderate",
                "success": False,
                "message": "No valid symptoms to analyze"
            })

        # Get symptoms from FAISS matches
        symptoms = data.get("extracted_symptoms", [])

        if not symptoms:
            return json.dumps({
                "severity_scores": {},
                "overall_severity": 4.0,
                "overall_level": "Moderate",
                "success": False,
                "message": "No symptoms extracted from matches"
            })

        # Use existing severity analysis function
        symptoms_str = ", ".join(symptoms)
        return analyze_symptom_severity(symptoms_str)

    except Exception as e:
        return json.dumps({
            "severity_scores": {},
            "overall_severity": 4.0,
            "overall_level": "Moderate",
            "success": False,
            "message": f"Error in severity analysis: {e}"
        })


@tool
def analyze_symptom_severity(symptoms_list: str) -> str:
    """
    Analyzes the severity of symptoms using FAISS similarity and severity
    database. (Kept for compatibility)

    Args:
        symptoms_list: Comma-separated string of symptoms

    Returns:
        JSON string containing severity analysis
    """
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
        else:  # 7
            return "Severe"

    severity_scores = {}

    for symptom in symptoms:
        try:
            # Use FAISS to find the closest matching symptom
            results = _FAISS_SEVERITY_INDEX.similarity_search_with_score(
                symptom, k=1)

            if results:
                matched_doc = results[0][0]
                matched_symptom = matched_doc.page_content.strip()

                # Lookup severity in CSV
                exact_match = _DF_DISEASE_SYMPTOM_SEVERITY[
                    (_DF_DISEASE_SYMPTOM_SEVERITY['symptom'].str.lower()
                     .str.strip() == matched_symptom.lower())
                ]

                if len(exact_match) > 0:
                    severity_scores[symptom] = \
                        float(exact_match['severity'].iloc[0])
                else:
                    # Fallback: partial matching
                    partial_matches = _DF_DISEASE_SYMPTOM_SEVERITY[
                        (_DF_DISEASE_SYMPTOM_SEVERITY['symptom'].str.lower()
                         .str.contains(matched_symptom.lower(), na=False))
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

    # Calculate overall severity
    if severity_scores:
        valid_scores = [score for score in severity_scores.values()
                        if score is not None]
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
def complete_medical_analysis(user_input: str) -> str:
    """
    One-stop medical analysis function that does everything:
    1. Direct FAISS search for diseases
    2. Symptom extraction from matches
    3. Severity analysis
    4. All in one streamlined call

    Args:
        user_input: User's raw symptom description

    Returns:
        JSON string with complete analysis
    """
    try:
        # Step 1: Direct symptom analysis via FAISS
        analysis_result = analyze_symptoms_direct(user_input)
        analysis_data = json.loads(analysis_result)

        if not analysis_data.get("success", False):
            return analysis_result

        # Step 2: Severity analysis using extracted symptoms
        severity_result = analyze_symptom_severity_from_matches(
            analysis_result)
        severity_data = json.loads(severity_result)

        # Step 3: Combine everything into comprehensive result
        complete_result = {
            "user_input": user_input,
            "diseases": analysis_data.get("diseases", []),
            "extracted_symptoms": analysis_data.get("extracted_symptoms", []),
            "severity_analysis": severity_data,
            "search_metadata": {
                "total_faiss_matches": analysis_data.get("total_matches", 0),
                "good_matches": analysis_data.get("good_matches", 0),
                "search_query": analysis_data.get("search_query", ""),
            },
            "success": True,
            "message": "Complete medical analysis performed"
        }

        return safe_json_dumps(complete_result)

    except Exception as e:
        return safe_json_dumps({
            "user_input": user_input,
            "diseases": [],
            "extracted_symptoms": [],
            "severity_analysis": {"success": False},
            "success": False,
            "message": f"Error in complete analysis: {e}"
        })


# Keep existing functions for compatibility
@tool
def get_disease_precautions(disease_name: str) -> str:
    """
    Gets precautions and preventive measures for a specific disease.
    (Unchanged from original)
    """
    if _DF_DISEASE_PRECAUTIONS is None:
        return json.dumps({
            "precautions": "Precautions database not available.",
            "success": False,
            "message": "Database not available"
        })

    def format_text_properly(text: str) -> str:
        if not text or not text.strip():
            return text

        # Medical acronyms
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
                sentence = sentence[0].upper() + sentence[1:] \
                    if len(sentence) > 1 else sentence.upper()

            words = sentence.split()
            formatted_words = []

            for word in words:
                clean_word = re.sub(r'[^\w]', '', word.lower())
                if clean_word in medical_acronyms:
                    formatted_word = re.sub(re.escape(clean_word),
                                            medical_acronyms[clean_word],
                                            word, flags=re.IGNORECASE)
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
            (_DF_DISEASE_PRECAUTIONS['disease'].str.lower() ==
             disease_name.lower())
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
    """
    Gets detailed description for a specific disease.
    (Unchanged from original)
    """
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
                sentence = sentence[0].upper() + sentence[1:] \
                    if len(sentence) > 1 else sentence.upper()

            words = sentence.split()
            formatted_words = []

            for word in words:
                clean_word = re.sub(r'[^\w]', '', word.lower())
                if clean_word in medical_acronyms:
                    formatted_word = re.sub(re.escape(clean_word),
                                            medical_acronyms[clean_word],
                                            word, flags=re.IGNORECASE)
                    formatted_words.append(formatted_word)
                else:
                    formatted_words.append(word)

            formatted_sentence = ' '.join(formatted_words)

            if formatted_sentence and not formatted_sentence[-1] in '.!?':
                formatted_sentence += '.'

            formatted_sentences.append(formatted_sentence)

        return ' '.join(formatted_sentences)

    try:
        # Find disease and description columns
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
                "description": (
                    f"Unable to find appropriate columns. Available: "
                    f"{list(_DF_DISEASE_SYMPTOM_DESCRIPTION.columns)}"
                ),
                "success": False,
                "message": "Column structure issue"
            })

        description = _DF_DISEASE_SYMPTOM_DESCRIPTION[
            (_DF_DISEASE_SYMPTOM_DESCRIPTION[disease_col].str.lower() ==
             disease_name.lower())
        ][desc_col].iloc[0]

        formatted_description = format_text_properly(description)

        return json.dumps({
            "description": formatted_description,
            "success": True,
            "message": f"Found description for {disease_name}"
        })

    except (IndexError, KeyError):
        return json.dumps({
            "description": f"No detailed description found for "
                          f"{disease_name}.",
            "success": False,
            "message": f"No description found for {disease_name}"
        })


# ===============================
# TOOL UTILITIES
# ===============================

def get_medical_tools():
    """
    Get all medical tools for use in LangChain agent.
    Now includes the new streamlined functions.

    Returns:
        List of medical tools
    """
    return [
        analyze_symptoms_direct,           # NEW: Main function
        complete_medical_analysis,         # NEW: One-stop analysis
        analyze_symptom_severity,          # Existing: For compatibility
        get_disease_precautions,          # Existing: Unchanged
        get_disease_description           # Existing: Unchanged
    ]


def test_tools():
    """Test that all tools are working properly."""
    print("=== STARTING STREAMLINED TOOL TESTS ===")
    print("Testing streamlined medical tools...")

    # Check if resources are available
    resources = check_resources_available()
    print(f"Resources available: {resources}")

    if not all(resources.values()):
        print("❌ Not all resources are available. Please call "
              "set_global_resources() first.")
        return False

    # Test complete medical analysis with realistic cases
    print("\n--- Testing complete_medical_analysis ---")
    test_cases = [
        "I have a headache and fever",
        "I'm experiencing fever, headache, chills",
        "I feel nauseous and have body aches",
        "fever, headache, chills, fatigue",
        "My ears burning, my skin is red and itchy and my nose is running",
        "I have itching, skin rash, nodal skin eruptions"
    ]

    for test_case in test_cases:
        try:
            print(f"\nTesting: '{test_case}'")
            result = complete_medical_analysis(test_case)
            result_data = json.loads(result)

            if result_data.get("success"):
                diseases = result_data.get("diseases", [])
                symptoms = result_data.get("extracted_symptoms", [])
                print(f"✅ Found {len(diseases)} diseases, "
                      f"{len(symptoms)} symptoms")

                # Show top disease
                if diseases:
                    top_disease = diseases[0]
                    print(f"   Top match: {top_disease['disease']} "
                          f"({top_disease['confidence_percentage']}%)")
            else:
                print(f"⚠️ Analysis failed: {result_data.get('message')}")

        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    # Test individual functions
    print("\n--- Testing get_disease_precautions ---")
    try:
        result = get_disease_precautions("drug reaction")
        result_data = json.loads(result)
        if result_data.get("success"):
            print("✅ Precautions retrieved successfully")
        else:
            print(f"⚠️ Precautions failed: {result_data.get('message')}")
    except Exception as e:
        print(f"❌ Precautions error: {e}")

    print("\n--- Testing get_disease_description ---")
    try:
        result = get_disease_description("drug reaction")
        result_data = json.loads(result)
        if result_data.get("success"):
            print("✅ Description retrieved successfully")
        else:
            print(f"⚠️ Description failed: {result_data.get('message')}")
    except Exception as e:
        print(f"❌ Description error: {e}")

    print("\n=== STREAMLINED TOOL TESTS COMPLETED ===")
    print("✅ All streamlined tools appear to be working correctly!")
    return True


def get_tool_descriptions():
    """Get descriptions of all available tools."""
    tools = get_medical_tools()
    return {tool.name: tool.description for tool in tools}


if __name__ == "__main__":
    print("Streamlined Medical Tools loaded successfully!")
    print("Main function: complete_medical_analysis(user_input)")
    print("Test function: test_tools()")

