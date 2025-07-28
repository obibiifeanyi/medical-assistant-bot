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

- [Market Analysis / Requirements Gathering for "Medical Chatbots"](https://huggingface.co/spaces/bott-wa/medical-assistant-bot/blob/main/docs/market_analysis.md)
- [Data Extract, Transform, Load](https://huggingface.co/spaces/bott-wa/medical-assistant-bot/blob/main/docs/data-etl-analysis.md)
- [High-Level Architecture](https://huggingface.co/spaces/bott-wa/medical-assistant-bot/blob/main/docs/architecture.md)
- [Live Demo](https://huggingface.co/spaces/bott-wa/medical-assistant-bot)

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

### Option 1: Direct Python Setup

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

### Option 2: Docker Deployment

For containerized deployment with Docker:

```bash
# Quick start with Docker Compose
cp .env.example .env  # Edit with your OPENAI_API_KEY
docker-compose up medical-assistant

# Or build and run manually
./scripts/docker/build.sh
docker run -p 8501:8501 -e OPENAI_API_KEY=your-key medical-assistant-bot:latest
```

**Development mode with hot reload:**
```bash
docker-compose --profile dev up medical-assistant-dev
```

**Access the application:**
- Streamlit UI: http://localhost:8501
- Development mode: http://localhost:8502

## Enterprise Deployment

### Amazon SageMaker Integration

Deploy to AWS SageMaker for production-scale inference:

```bash
# Build SageMaker-compatible image
docker build --target sagemaker -t medical-assistant-sagemaker:latest .

# Deploy to SageMaker (requires AWS credentials)
python scripts/docker/deploy-sagemaker.py \
    --image-uri <your-ecr-uri> \
    --endpoint-name medical-assistant-endpoint \
    --instance-type ml.t2.medium
```

**SageMaker Features:**
- REST API endpoints (`/ping`, `/invocations`, `/health`)
- Auto-scaling based on traffic
- Multiple instance types (ml.t2.medium to ml.c5.xlarge)
- Enterprise security with IAM roles

**Test SageMaker deployment:**
```bash
# Test local SageMaker container
python scripts/docker/test-sagemaker.py --endpoint-type local

# Test live SageMaker endpoint
python scripts/docker/test-sagemaker.py --endpoint-type sagemaker
```

For detailed deployment instructions, see [CLAUDE.md](CLAUDE.md#-docker-deployment).

## 🧪 Testing Framework

The project includes a comprehensive testing suite with three different testing frameworks to ensure reliability and quality across all components.

### Testing Frameworks Overview

#### 1. **Pytest (Python Unit Tests)** ✅
Comprehensive Python unit and integration testing for core functionality.

**Coverage includes:**
- Core medical analysis functions (`src/medical_tools.py`)
- Image processing capabilities (`src/vision_tools.py`)
- Agent functionality (`src/medical_agent_langchain.py`)
- JSON serialization and data validation
- Error handling and edge cases
- Parametrized tests for multiple scenarios

**Test files:**
- `tests/test_medical_tools.py` - Medical analysis functions
- `tests/test_vision_tools.py` - Vision and image processing
- `tests/test_medical_agent.py` - Agent and chat functionality

#### 2. **Robot Framework (End-to-End Testing)** 🤖
Keyword-driven testing for complete user workflows and system integration.

**Coverage includes:**
- System health checks and file validation
- Complete symptom analysis workflows
- Image upload and processing validation
- Conversation memory and context testing
- Error handling and recovery scenarios
- Performance benchmarking
- Medical disclaimer verification

**Test files:**
- `tests/robot/medical_assistant.robot` - Main test suite
- `tests/robot_keywords/MedicalAssistantKeywords.py` - Custom keywords

#### 3. **JUnit (Java Integration Testing)** ☕
Cross-language integration testing for system-level validation.

**Coverage includes:**
- Python-Java process execution testing
- HTTP API interaction validation
- Performance and memory usage monitoring
- Cross-platform compatibility testing
- System integration scenarios

**Test files:**
- `tests/java/MedicalAssistantIntegrationTest.java` - Integration tests
- `pom.xml` - Maven configuration

### Running Tests

#### Prerequisites
```bash
# Install testing dependencies
pip install -r requirements.txt

# The following testing packages are included:
# - pytest>=7.0.0
# - pytest-html>=3.1.0
# - pytest-cov>=4.0.0
# - pytest-mock>=3.10.0
# - robotframework>=6.0.0
# - robotframework-requests>=0.9.0
```

#### Individual Framework Commands

**Run Pytest (Python Unit Tests):**
```bash
# Basic test run
pytest tests/

# With verbose output and coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_medical_tools.py -v

# Run with HTML report
pytest tests/ --html=test-results/pytest-report.html --self-contained-html
```

**Run Robot Framework (End-to-End Tests):**
```bash
# Basic test run
robot tests/robot/medical_assistant.robot

# With custom output directory
robot --outputdir test-results/robot tests/robot/medical_assistant.robot

# Run specific test tags
robot --include smoke tests/robot/medical_assistant.robot

# Dry run (syntax check only)
robot --dryrun tests/robot/medical_assistant.robot
```

**Run JUnit Tests (requires Java 11+ and Maven):**
```bash
# Install Java and Maven first
sudo apt update
sudo apt install openjdk-11-jdk maven

# Run all tests
mvn test

# Run with specific profile
mvn test -P integration

# Fast tests only (skip slow tests)
mvn test -P fast
```

#### Comprehensive Test Runner

**Run all testing frameworks at once:**
```bash
# Execute complete test suite with reporting
python test_runner.py
```

This will:
- Run all pytest tests with coverage reporting
- Execute Robot Framework syntax validation
- Check Java/Maven environment availability
- Generate a comprehensive test report in `test-results/comprehensive-test-report.md`

#### Test Configuration Files

- **`pytest.ini`** - Pytest configuration with coverage settings
- **`robot.yaml`** - Robot Framework configuration
- **`pom.xml`** - Maven configuration for Java tests

#### Test Results and Reporting

All test results are saved to the `test-results/` directory:

```
test-results/
├── pytest-report.html          # Pytest HTML report
├── coverage-html/               # Coverage report
├── robot/                       # Robot Framework reports
│   ├── report.html
│   ├── log.html
│   └── output.xml
├── junit/                       # JUnit test reports
└── comprehensive-test-report.md # Combined report
```

#### Continuous Integration

The testing framework is designed for CI/CD integration:

```bash
# CI-friendly command (non-interactive)
pytest tests/ --tb=short --junit-xml=test-results/junit.xml
robot --outputdir test-results/robot --exitonfailure tests/robot/
```

#### Test Development Guidelines

When adding new features:

1. **Write unit tests first** (pytest) for individual functions
2. **Add integration tests** (Robot Framework) for user workflows  
3. **Include error handling tests** for edge cases
4. **Update test documentation** in this section
5. **Run full test suite** before committing changes

**Mock Testing:**
- Tests include comprehensive mocking for external dependencies
- API keys not required for most unit tests
- Offline testing capabilities for development environments

For detailed testing architecture and guidelines, see [CLAUDE.md](CLAUDE.md).

## Important Disclaimers

⚠️ **This is not a replacement for professional medical advice**
- Always consult with qualified healthcare providers for proper diagnosis
- This tool provides general information only
- Do not make medical decisions based solely on this tool's output
- In case of emergency, contact emergency services immediately



