---
title: Medical Assistant Bot
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.29.0
app_file: app.py
pinned: false
---

# Agentic AI : Medical Assistant Chatbot 🏥!

A conversational AI medical assistant that can analyze symptoms, identify possible conditions, and provide health information using LangChain and OpenAI. 

## 🧠 Design Philosophy

While today's LLMs are capable of impersonating medical professionals on their own, this **agentic multimodal system** demonstrates a more disciplined, secure, and scalable design:

> It constrains the model’s outputs to **only** the information it has been explicitly given — both in **content** *and* **context**.

This architecture enables:

- Domain-specific tuning for medical scenarios  
- Reduced hallucinations through contextual grounding  
- Intelligent interaction with both **textual symptoms** and **medical images**

The result? A smarter, safer, and more controllable diagnostic assistant — **purpose-built for trust**.



## 🔗 Key Resources

- [Quick Market Analysis for "Medibots"](https://huggingface.co/spaces/bott-wa/medical-assistant-bot/blob/main/docs/market_analysis.md)
- [Data Massage Therapy](https://huggingface.co/spaces/bott-wa/medical-assistant-bot/blob/main/docs/data-etl-analysis.md)
- [High-Level Architecture ](https://huggingface.co/spaces/bott-wa/medical-assistant-bot/blob/main/docs/architecture.md)

- [Jammin' Demo](https://huggingface.co/spaces/bott-wa/medical-assistant-bot)

## Features

- **Symptom & Image Analysis**: Understands natural language input and images using OpenAI Vision
- **Disease Matching**: Uses FAISS vector search to find matching conditions
- **Severity Assessment**: Evaluates the severity of reported symptoms
- **Medical Information**: Provides descriptions and precautions for conditions
- **Conversational Interface**: Maintains context throughout the conversation

## How to Use

1. **Describe your symptoms** in the chat interface or upload an image
2. **Ask follow-up questions** about conditions, precautions, or severity
3. **Clear conversation** when starting a new consultation

## Technology Stack

- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-3.5 Turbo /GPT-4.1 (vision)
- **Framework**: LangChain
- **Vector Search**: FAISS
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Image Support**: OpenAI Vision

## Data Sources

The medical knowledge base used here is small but usable includes:
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
git clone https://github.com/wbott/medical-assistant-bot
cd medical-assistant-bot

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key:
# On Linux/macOS:
export OPENAI_API_KEY="your-key-here"

# On Windows CMD:
set OPENAI_API_KEY=your-key-here

# On Windows PowerShell:
$env:OPENAI_API_KEY="your-key-here"

# Run the app
streamlit run app.py
```

## Important Disclaimers

⚠️ **This is not a replacement for professional medical advice**
- Always consult with qualified healthcare providers for proper diagnosis
- This tool provides general information only
- Do not make medical decisions based solely on this tool's output
- In case of emergency, contact emergency services immediately



