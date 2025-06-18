"""
Agentic Medical Assistant using LangChain.

This module provides a conversational AI medical assistant that can analyze
symptoms, match diseases, assess severity, and provide medical recommendations
using LangChain's agent framework with OpenAI function calling.
"""

# Standard library imports
import warnings

# Third-party imports
import pandas as pd
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

# Local imports
from medical_tools import get_medical_tools, set_global_resources

warnings.filterwarnings('ignore')


# ===============================
# LANGCHAIN AGENT
# ===============================

class AgenticMedicalAssistant:
    """LangChain-based agentic medical assistant."""

    def __init__(self,
                 faiss_symptom_index: FAISS,
                 faiss_severity_index: FAISS,
                 df_disease_precautions: pd.DataFrame,
                 df_disease_symptom_description: pd.DataFrame,
                 df_disease_symptom_severity: pd.DataFrame,
                 model_name: str = "gpt-3.5-turbo",
                 use_memory: bool = True):

        # Set global resources for tools
        set_global_resources(
            faiss_symptom_index, faiss_severity_index,
            df_disease_precautions, df_disease_symptom_description,
            df_disease_symptom_severity
        )

        # Initialize LLM - Will use OPENAI_API_KEY environment variable
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.1
        )

        # Define tools - Import from medical_tools module
        self.tools = get_medical_tools()

        # Create system prompt - MUCH more aggressive and explicit
        system_prompt = """You are a medical assistant AI.

When a user describes symptoms, you MUST:
1. Use the extract_symptoms tool with their exact words
2. Pass their complete message to extract_symptoms exactly as they wrote it
3. Do NOT modify, paraphrase, or change any part of their input

Example: If user says "I have a fever and headache", pass "I have a fever \
and headache" to extract_symptoms.

CRITICAL: Use their actual text, not placeholder text like "{input}"."""

        # Create prompt template with conditional memory support
        # Use a simpler approach that works with OpenAI function calling
        if use_memory:
            # Prompt with memory (chat_history)
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history",
                                    optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
        else:
            # Prompt without memory (no chat_history)
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])

        # Create memory (optional)
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

        # Create agent executor with conditional memory
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,  # This will be None if use_memory=False
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=20,
            early_stopping_method="generate",
            return_intermediate_steps=True
        )

    def chat(self, user_input: str) -> str:
        """
        Main chat interface for the medical assistant.

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

    def reset_conversation(self):
        """Reset the conversation memory."""
        if self.memory is not None:
            self.memory.clear()


# ===============================
# USAGE EXAMPLE
# ===============================

def create_medical_assistant(
    faiss_symptom_index: FAISS,
    faiss_severity_index: FAISS,
    df_disease_precautions: pd.DataFrame,
    df_disease_symptom_description: pd.DataFrame,
    df_disease_symptom_severity: pd.DataFrame,
    model_name: str = "gpt-3.5-turbo",
    use_memory: bool = True
) -> AgenticMedicalAssistant:
    """
    Factory function to create a medical assistant.

    Args:
        faiss_symptom_index: FAISS index for symptom-disease matching
        faiss_severity_index: FAISS index for severity analysis
        df_disease_precautions: DataFrame with disease precautions
        df_disease_symptom_description: DataFrame with disease descriptions
        df_disease_symptom_severity: DataFrame with symptom severity scores
        model_name: OpenAI model name (requires OPENAI_API_KEY env variable)
        use_memory: Whether to use conversation memory (set False for debug)

    Returns:
        Configured AgenticMedicalAssistant instance
    """
    return AgenticMedicalAssistant(
        faiss_symptom_index=faiss_symptom_index,
        faiss_severity_index=faiss_severity_index,
        df_disease_precautions=df_disease_precautions,
        df_disease_symptom_description=df_disease_symptom_description,
        df_disease_symptom_severity=df_disease_symptom_severity,
        model_name=model_name,
        use_memory=use_memory

    )
