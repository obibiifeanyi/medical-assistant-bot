"""
Streamlined Medical Agent with Proper Tool Flow
==============================================

This version ensures the agent:
1. Uses analyze_medical_image to extract visible symptoms
2. Uses analyze_symptoms_direct with the extracted symptoms  
3. Uses get_disease_description for the identified diseases
4. Provides clean, focused medical responses
"""

import warnings
from typing import Optional
import json

import pandas as pd
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

# Import enhanced tools with debugging
from medical_tools import get_enhanced_medical_tools, set_global_resources

warnings.filterwarnings('ignore')

class AgenticMedicalAssistant:
    """Streamlined medical assistant with proper tool workflow."""

    def __init__(self,
                 faiss_symptom_index: FAISS,
                 faiss_severity_index: FAISS,
                 df_disease_precautions: pd.DataFrame,
                 df_disease_symptom_description: pd.DataFrame,
                 df_disease_symptom_severity: pd.DataFrame,
                 vision_model: Optional[ChatOpenAI] = None,
                 model_name: str = "gpt-3.5-turbo",
                 use_memory: bool = True):

        print(f"🚀 Initializing Streamlined AgenticMedicalAssistant...")

        # Set global resources for tools
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
        print(f"🔧 Tools loaded: {[tool.name for tool in self.tools]}")

        # Streamlined system prompt focused on proper tool workflow
        system_prompt = """You are a medical assistant AI that uses specialized tools to analyze symptoms and provide medical information.

**CRITICAL WORKFLOW FOR IMAGE ANALYSIS:**

When a user uploads an image and asks for analysis:

1. **FIRST**: Use analyze_medical_image() to extract visible symptoms from the image
2. **THEN**: Use analyze_symptoms_direct() with the extracted visible symptoms
3. **FINALLY**: Use get_disease_description() for the top disease matches

**RESPONSE FORMAT:**
Based on the image analysis, I found these visible symptoms: [list symptoms]

Using medical analysis, here are the most likely conditions:

1. **[Disease Name]**
   Confidence: [X]%
   
   [Disease description]

2. **[Disease Name]**  
   Confidence: [X]%
   
   [Disease description]

Would you like precautions for any of these conditions?

**EXAMPLE WORKFLOW:**
User: "Analyze this skin image"
1. Call analyze_medical_image() → gets "erythematous papules, vesicular lesions"
2. Call analyze_symptoms_direct("erythematous papules, vesicular lesions") → gets diseases
3. Call get_disease_description() for top diseases
4. Format clean response

**FOR TEXT-ONLY SYMPTOMS:**
1. Use analyze_symptoms_direct() with user's text
2. Use get_disease_description() for top matches
3. Format response

**IMPORTANT:**
- Always extract symptoms from images FIRST
- Always use the medical database tools for diagnosis
- Keep responses focused and medical
- Don't include verbose image analysis details
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
            verbose=True,  # Keep verbose for debugging
            handle_parsing_errors=True,
            max_iterations=10,
            early_stopping_method="generate",
            return_intermediate_steps=True
        )
        
        print("✅ Streamlined AgenticMedicalAssistant initialized!")

    def chat(self, user_input: str) -> str:
        """
        Streamlined chat interface with proper tool workflow.
        
        Args:
            user_input: User's message/question
            
        Returns:
            Assistant's response
        """
        print(f"\n🎯 STREAMLINED CHAT REQUEST: {user_input}")
        
        try:
            response = self.agent_executor.invoke({"input": user_input})
            
            # Log tool usage for debugging
            if 'intermediate_steps' in response:
                steps = response['intermediate_steps']
                print(f"🔍 Tools used: {len(steps)} steps")
                for i, step in enumerate(steps):
                    if hasattr(step, '__len__') and len(step) >= 2:
                        action, observation = step[0], step[1]
                        print(f"  {i+1}. {action.tool}")
            
            final_output = response.get("output", "No output generated")
            return final_output
            
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."
            print(f"❌ Chat error: {error_msg}")
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
            "streamlined_workflow": True,
            "memory": self.memory is not None,
            "vision_model": str(self.vision_model.model_name) if self.vision_model else None,
            "available_tools": [tool.name for tool in self.tools]
        }


def create_medical_assistant(
    faiss_symptom_index: FAISS,
    faiss_severity_index: FAISS,
    df_disease_precautions: pd.DataFrame,
    df_disease_symptom_description: pd.DataFrame,
    df_disease_symptom_severity: pd.DataFrame,
    vision_model: Optional[ChatOpenAI] = None,
    model_name: str = "gpt-3.5-turbo",
    use_memory: bool = True
) -> AgenticMedicalAssistant:
    """
    Factory function to create a streamlined medical assistant.

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
    print("🏭 Creating streamlined medical assistant...")
    
    return AgenticMedicalAssistant(
        faiss_symptom_index=faiss_symptom_index,
        faiss_severity_index=faiss_severity_index,
        df_disease_precautions=df_disease_precautions,
        df_disease_symptom_description=df_disease_symptom_description,
        df_disease_symptom_severity=df_disease_symptom_severity,
        vision_model=vision_model,
        model_name=model_name,
        use_memory=use_memory
    )

if __name__ == "__main__":
    print("Streamlined Medical Agent with Proper Tool Flow loaded!")
    print("=" * 60)
    print("Key Features:")
    print("- Focused symptom extraction from images")
    print("- Proper tool workflow: image → symptoms → diseases → descriptions")
    print("- Clean, medical-focused responses")
    print("- Streamlined user experience")