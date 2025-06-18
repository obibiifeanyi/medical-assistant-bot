# Medical Assistant Chatbot 🏥

A conversational AI medical assistant that can analyze symptoms, identify possible conditions, and provide health information using LangChain and OpenAI.

## Features

- **Symptom Analysis**: Describe your symptoms in natural language
- **Disease Matching**: Uses FAISS vector search to find matching conditions
- **Severity Assessment**: Evaluates the severity of reported symptoms
- **Medical Information**: Provides descriptions and precautions for conditions
- **Conversational Interface**: Maintains context throughout the conversation

## How to Use

1. **Enter your OpenAI API Key** in the sidebar
2. **Describe your symptoms** in the chat interface
3. **Ask follow-up questions** about conditions, precautions, or severity
4. **Clear conversation** when starting a new consultation

## Important Disclaimers

⚠️ **This is not a replacement for professional medical advice**
- Always consult with qualified healthcare providers for proper diagnosis
- This tool provides general information only
- Do not make medical decisions based solely on this tool's output
- In case of emergency, contact emergency services immediately

## Technology Stack

- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-3.5/GPT-4
- **Framework**: LangChain
- **Vector Search**: FAISS
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)

## Data Sources

The medical knowledge base includes:
- Disease-symptom relationships
- Symptom severity scores
- Disease descriptions
- Precautionary measures

## Privacy

- Your OpenAI API key is only stored in your session
- Conversations are not permanently stored
- No personal health information is collected

## Local Development

To run locally:

```bash
# Clone the repository
git clone <your-repo-url>

# Install dependencies
pip install -r requirements.txt

# Set up your OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Run the app
streamlit run app.py
```

---

**Remember**: This tool is for educational and informational purposes only. Always seek professional medical advice for health concerns.