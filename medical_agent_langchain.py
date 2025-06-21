"""
Enhanced Agentic Medical Assistant with Image Analysis using LangChain
======================================================================

This module extends the original medical assistant to include image analysis
capabilities using OpenAI's Vision API, while maintaining the existing
agentic architecture and conversation memory.
"""

import warnings
from typing import Optional

import pandas as pd
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

# Import enhanced tools - CORRECTED IMPORT
from medical_tools import get_enhanced_medical_tools, set_global_resources

warnings.filterwarnings('ignore')

class AgenticMedicalAssistant:  # CHANGED: Remove "Enhanced" to match original
    """Enhanced LangChain-based agentic medical assistant with image analysis."""

    def __init__(self,
                 faiss_symptom_index: FAISS,
                 faiss_severity_index: FAISS,
                 df_disease_precautions: pd.DataFrame,
                 df_disease_symptom_description: pd.DataFrame,
                 df_disease_symptom_severity: pd.DataFrame,
                 vision_model: Optional[ChatOpenAI] = None,
                 model_name: str = "gpt-3.5-turbo",
                 use_memory: bool = True):

        # Set global resources for tools (including vision model)
        set_global_resources(
            faiss_symptom_index, faiss_severity_index,
            df_disease_precautions, df_disease_symptom_description,
            df_disease_symptom_severity, vision_model
        )

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.1
        )
        
        # Store vision model reference
        self.vision_model = vision_model
        self.has_vision = vision_model is not None

        # Get enhanced tools
        self.tools = get_enhanced_medical_tools()

        # Enhanced system prompt for multimodal analysis
        system_prompt = """You are an advanced medical assistant AI that can analyze both text descriptions AND medical images to provide comprehensive health assessments.

**MULTIMODAL CAPABILITIES:**
- Text analysis: Symptom descriptions, medical histories
- Image analysis: Visual symptoms like rashes, burns, discoloration, swelling
- Combined analysis: Text + image for comprehensive diagnosis

**ANALYSIS WORKFLOW:**

For TEXT-ONLY queries:
1. Use analyze_symptoms_direct() with the user's description
2. Get disease descriptions for top matches using get_disease_description()
3. Format response as before

For IMAGE-ONLY requests ("analyze this image", "what do you see"):
1. Use analyze_medical_image() with the provided image
2. Use analyze_symptoms_with_image() with extracted visual symptoms
3. Get disease descriptions for matches
4. Format focusing on visual findings

For TEXT + IMAGE queries:
1. Use complete_multimodal_analysis() with both text and image
2. This combines visual and textual symptom analysis
3. Get disease descriptions for top matches
4. Format highlighting both text and visual findings

**RESPONSE FORMAT:**
Based on your [symptoms/image/symptoms and image], here are the most likely conditions:

1. **[Disease Name]**
   Confidence: [X]%
   [Visual Support: Supported by image analysis] (if applicable)
   
   [Disease description]

2. **[Disease Name]**
   Confidence: [X]%
   
   [Disease description]

**Visual Findings:** (if image provided)
- [List key visual symptoms identified]
- [Note any concerning features]

Would you like precautions for any of these conditions?

**CRITICAL FORMATTING RULES:**
- Do NOT show raw JSON or tool outputs
- DO highlight when findings are supported by both text AND image
- DO mention specific visual findings when available
- DO ask about precautions at the end
- DO include medical disclaimer for image analysis

**IMAGE ANALYSIS DISCLAIMER:**
When image analysis is involved, add: "⚠️ Visual analysis is for informational purposes only. This tool cannot replace professional medical examination of skin or other conditions."

**EXAMPLES:**

User: "I have a rash on my arm" + uploads image
Response: 
Based on your symptoms and the uploaded image, here are the most likely conditions:

1. **Contact Dermatitis**
   Confidence: 78%
   Visual Support: Supported by image analysis
   
   Contact dermatitis is an inflammatory skin condition...

**Visual Findings:**
- Red, inflamed patch visible on arm
- Defined borders suggesting contact reaction
- No signs of infection or severe inflammation

Would you like precautions for contact dermatitis?

⚠️ Visual analysis is for informational purposes only. This tool cannot replace professional medical examination of skin conditions.
"""

        # Create prompt template
        if use_memory:
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
        else:
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])

        # Create memory
        if use_memory:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        else:
            self.memory = None

        # Create agent
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )

        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=20,
            early_stopping_method="generate",
            return_intermediate_steps=True
        )

    def chat(self, user_input: str) -> str:
        """
        Standard chat interface for text-only interactions.
        
        Args:
            user_input: User's message/question
            
        Returns:
            Assistant's response
        """
        try:
            response = self.agent_executor.invoke({"input": user_input})
            return response["output"]
        except Exception as e:
            error_msg = (
                f"I apologize, but I encountered an error: {str(e)}. "
                f"Please try rephrasing your question or consult with a "
                f"healthcare professional."
            )
            return error_msg

    def chat_with_image(self, user_input: str, image_data_b64: str, 
                       additional_context: str = "") -> str:
        """
        Enhanced chat interface for multimodal interactions with images.
        
        Args:
            user_input: User's text description
            image_data_b64: Base64 encoded image data
            additional_context: Additional context about the image
            
        Returns:
            Assistant's response combining text and image analysis
        """
        if not self.has_vision:
            return (
                "I understand you've uploaded an image, but I don't currently have "
                "image analysis capabilities configured. I can still help analyze "
                "your text description of symptoms. Please describe what you see "
                "in the image and I'll do my best to help."
            )
        
        try:
            # Construct enhanced input for multimodal analysis
            enhanced_input = f"""
            User text input: {user_input}
            
            Image provided: Yes (base64 data available)
            Additional context: {additional_context}
            
            Please use complete_multimodal_analysis() with the following parameters:
            - user_input: "{user_input}"
            - image_data_b64: "{image_data_b64}"
            - additional_context: "{additional_context}"
            
            Then provide disease descriptions for the top matches found.
            """
            
            response = self.agent_executor.invoke({"input": enhanced_input})
            return response["output"]
            
        except Exception as e:
            error_msg = (
                f"I apologize, but I encountered an error analyzing your image and symptoms: {str(e)}. "
                f"Please try again or consult with a healthcare professional for proper evaluation."
            )
            return error_msg

    def analyze_image_only(self, image_data_b64: str, additional_context: str = "") -> str:
        """
        Analyze only the image without additional text input.
        
        Args:
            image_data_b64: Base64 encoded image data
            additional_context: Additional context about the image
            
        Returns:
            Assistant's response focusing on visual findings
        """
        if not self.has_vision:
            return (
                "I apologize, but I don't currently have image analysis capabilities "
                "configured. Please describe what you see in the image and I'll help "
                "analyze the symptoms."
            )
        
        try:
            # Focus purely on image analysis
            image_only_input = f"""
            Please analyze the uploaded medical image only.
            
            Image provided: Yes (base64 data available)
            Additional context: {additional_context}
            
            Use analyze_medical_image() with:
            - image_data_b64: "{image_data_b64}"
            - additional_context: "{additional_context}"
            
            Then use the visual findings to find possible conditions and provide descriptions.
            Focus your response on what you can see in the image.
            """
            
            response = self.agent_executor.invoke({"input": image_only_input})
            return response["output"]
            
        except Exception as e:
            error_msg = (
                f"I apologize, but I encountered an error analyzing the image: {str(e)}. "
                f"Please try again or consult with a healthcare professional."
            )
            return error_msg

    def reset_conversation(self):
        """Reset the conversation memory."""
        if self.memory is not None:
            self.memory.clear()

    def get_capabilities(self) -> dict:
        """Get information about assistant capabilities."""
        return {
            "text_analysis": True,
            "image_analysis": self.has_vision,
            "multimodal_analysis": self.has_vision,
            "memory": self.memory is not None,
            "vision_model": str(self.vision_model.model_name) if self.vision_model else None,
            "available_tools": [tool.name for tool in self.tools]
        }


def create_medical_assistant(  # CHANGED: Back to original function name
    faiss_symptom_index: FAISS,
    faiss_severity_index: FAISS,
    df_disease_precautions: pd.DataFrame,
    df_disease_symptom_description: pd.DataFrame,
    df_disease_symptom_severity: pd.DataFrame,
    vision_model: Optional[ChatOpenAI] = None,
    model_name: str = "gpt-3.5-turbo",
    use_memory: bool = True
) -> AgenticMedicalAssistant:  # CHANGED: Back to original class name
    """
    Factory function to create an enhanced medical assistant with image analysis.

    Args:
        faiss_symptom_index: FAISS index for symptom-disease matching
        faiss_severity_index: FAISS index for severity analysis
        df_disease_precautions: DataFrame with disease precautions
        df_disease_symptom_description: DataFrame with disease descriptions
        df_disease_symptom_severity: DataFrame with symptom severity scores
        vision_model: Optional OpenAI vision model for image analysis
        model_name: OpenAI model name for text processing
        use_memory: Whether to use conversation memory

    Returns:
        Configured AgenticMedicalAssistant instance
    """
    return AgenticMedicalAssistant(  # CHANGED: Back to original class name
        faiss_symptom_index=faiss_symptom_index,
        faiss_severity_index=faiss_severity_index,
        df_disease_precautions=df_disease_precautions,
        df_disease_symptom_description=df_disease_symptom_description,
        df_disease_symptom_severity=df_disease_symptom_severity,
        vision_model=vision_model,
        model_name=model_name,
        use_memory=use_memory
    )


# ===============================
# USAGE EXAMPLES AND TESTING
# ===============================

def example_usage():
    """Example of how to use the Kickstart HealthIQ assistant."""
    print("Kickstart HealthIQ Usage Examples:")
    print("=" * 50)
    
    # This would be called after loading your data and indices
    """
    # Load your resources first
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_symptom = FAISS.load_local("indices/faiss_symptom_index", embeddings)
    faiss_severity = FAISS.load_local("indices/faiss_severity_index", embeddings)
    
    # Load DataFrames
    df_precautions = pd.read_csv("data/disease_precautions.csv")
    df_descriptions = pd.read_csv("data/disease_symptom_description.csv")
    df_severity = pd.read_csv("data/disease_symptom_severity.csv")
    
    # Initialize vision model
    vision_model = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    # Create enhanced assistant
    assistant = create_medical_assistant(
        faiss_symptom_index=faiss_symptom,
        faiss_severity_index=faiss_severity,
        df_disease_precautions=df_precautions,
        df_disease_symptom_description=df_descriptions,
        df_disease_symptom_severity=df_severity,
        vision_model=vision_model,
        model_name="gpt-3.5-turbo",
        use_memory=True
    )
    
    # Usage examples:
    
    # 1. Text-only analysis
    response1 = assistant.chat("I have a fever and headache")
    print("Text Analysis:", response1)
    
    # 2. Image-only analysis
    with open("rash_photo.jpg", "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    response2 = assistant.analyze_image_only(
        image_data_b64=image_data,
        additional_context="Appeared 2 days ago, slightly itchy"
    )
    print("Image Analysis:", response2)
    
    # 3. Combined text + image analysis
    response3 = assistant.chat_with_image(
        user_input="I have a rash that appeared suddenly",
        image_data_b64=image_data,
        additional_context="No pain but some itching"
    )
    print("Multimodal Analysis:", response3)
    
    # 4. Check capabilities
    capabilities = assistant.get_capabilities()
    print("Capabilities:", capabilities)
    """
    
    print("\nKey Features:")
    print("- Text symptom analysis (original functionality)")
    print("- Medical image analysis (NEW)")
    print("- Combined multimodal analysis (NEW)")
    print("- Maintains conversation memory")
    print("- Agentic tool selection")
    print("- Diagnostic medical responses")


if __name__ == "__main__":
    print("Enhanced Agentic Medical Assistant loaded successfully!")
    print("=" * 60)
    print("Key Classes:")
    print("- AgenticMedicalAssistant: Main assistant class")
    print("\nKey Functions:")
    print("- create_medical_assistant(): Factory function")
    print("- chat(): Text-only analysis")
    print("- chat_with_image(): Multimodal analysis")
    print("- analyze_image_only(): Image-only analysis")
    print("\nNew Capabilities:")
    print("✅ Medical image analysis")
    print("✅ Visual symptom extraction")
    print("✅ Multimodal diagnosis")
    print("✅ Enhanced formatting for visual findings")
    print("✅ Backward compatibility with existing functionality")
    
    example_usage()