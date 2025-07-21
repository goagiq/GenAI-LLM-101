# GenAI-LLM-101: Large Language Model Engineering Fundamentals

A comprehensive collection of Jupyter notebooks and Python scripts designed to teach the fundamentals of Large Language Model (LLM) engineering using modern frameworks like LangChain, OpenAI, Groq, and Hugging Face Transformers.

> **📝 Note**: These Jupyter notebooks are designed to run on **Google Colab** and include Colab-specific configurations for API key management and library installations. While they can be adapted for local environments, the easiest way to get started is by opening them directly in Google Colab.

## 📚 Course Overview

This repository contains practical examples and tutorials covering key concepts in LLM engineering, from basic prompting techniques to advanced RAG (Retrieval-Augmented Generation) implementations. The materials are designed for both beginners and intermediate practitioners looking to understand modern LLM development patterns.

## 🎯 Learning Objectives

- Understand LLM fundamentals: temperature, tokenization, and prompt engineering
- Master LangChain framework for building LLM applications
- Implement conversation memory and context management
- Build text summarization and sentiment analysis systems
- Work with vector embeddings and databases
- Create RAG pipelines for knowledge augmentation
- Compare different LLM providers (OpenAI, Groq, Local models)

## 📁 Repository Structure

### Core Learning Modules

#### 🔥 **Temperature & Tokenization**
- `100_Temperature_&_Tokenization.ipynb` - Understanding how temperature affects LLM output and text tokenization
- `01_temperature_&_tokenization.py` - Python script version

#### 🎯 **Prompt Engineering**
- `101_PromptTemplate_OLD_2024_08.ipynb` - Legacy prompt template implementation
- `002_prompttemplate.py` - Modern prompt template abstraction using LangChain

#### 💬 **Message Types & Context**
- `103_SystemMessage_vs_HumanMessage.ipynb` - Understanding system vs human messages
- `003_systemmessage_vs_humanmessage.py` - Python implementation

#### 🧠 **Conversation Memory**
- `104_ConversationMemory.ipynb` - Implementing conversation memory for context retention
- `004_conversationmemory.py` - Python script version

#### 📝 **Text Summarization**
- `105_Summarization.ipynb` - Text and PDF summarization techniques
- `005_summarization.py` - Advanced summarization with web scraping and aspect-based sentiment analysis

#### 🔍 **Vector Embeddings & Databases**
- `007_vector_embeddings.py` - Working with vector embeddings using Deep Lake
- `106_vectordatabase.py` - Vector database implementation for RAG systems

### Specialized Applications

#### 🎭 **Sentiment Analysis**
- `02_Groq_AspectBasedSentiment_(CPU).ipynb` - Aspect-based sentiment analysis using Groq API
- `03_Llama3AspectBasedSentiment_(slow_CPU).ipynb` - Local Llama 3.2 implementation

#### 🚀 **Advanced RAG Implementation**
- `01_Main(GPU).ipynb` - Complete RAG pipeline using Haystack and Ollama with local models

## 🛠️ Technical Stack

### Primary Frameworks
- **LangChain**: Framework for LLM application development
- **Haystack**: Advanced RAG pipeline implementation
- **OpenAI**: GPT models (GPT-3.5-turbo, GPT-4o-mini)
- **Groq**: High-performance LLM inference
- **Transformers**: Hugging Face transformers library

### LLM Providers & Models
- **OpenAI**: GPT-3.5-turbo, GPT-4o-mini, GPT-3.5-turbo-instruct
- **Groq**: Llama3-70b-8192
- **Local Models**: Llama-3.2-1B-Instruct, Llama3.2:3b-instruct-fp16
- **Ollama**: Local model serving

### Vector Databases & Tools
- **Deep Lake**: ActiveLoop's vector database
- **FAISS**: Facebook AI Similarity Search
- **PyPDF**: PDF processing
- **NLTK**: Natural language processing
- **Tiktoken**: OpenAI tokenization

## 🚀 Getting Started

### Prerequisites

1. **Python 3.8+** with pip
2. **API Keys** (keep these secure):
   - OpenAI API Key ([Get here](https://platform.openai.com/api-keys))
   - Groq API Key ([Get here](https://console.groq.com/))
   - ActiveLoop API Token ([Get here](https://app.activeloop.ai/))

### Installation

**For Google Colab** (Recommended):
- No installation required! The notebooks include all necessary `!pip install` commands
- Libraries are installed automatically when you run the setup cells
- API keys are managed through Colab's secure Secrets feature

**For Local Development** (Advanced):
```bash
# Core dependencies
pip install langchain langchain-community langchain-openai
pip install openai groq transformers
pip install haystack-ai ollama-haystack
pip install deeplake[enterprise] faiss-cpu
pip install pypdf tiktoken nltk
pip install torch numpy pandas

# For local models
pip install ollama gpt4all
```

### Quick Start

1. **Open in Google Colab** (Recommended):
   - Click on any `.ipynb` file in this repository
   - Click "Open in Colab" button or copy the file to your Google Drive
   - Follow the notebook instructions for API key setup

2. **Set up your API keys in Google Colab**:
   - Use Colab's built-in Secrets feature (🔑 icon in left sidebar)
   - Add your keys: `OPENAI_API_KEY`, `GROQ_API_KEY`, etc.
   - The notebooks are configured to use `userdata.get()` for secure access

3. **Start with basic temperature exploration**:
   ```bash
   # In Google Colab, simply run the cells in:
   100_Temperature_&_Tokenization.ipynb
   ```

4. **Try the main RAG implementation**:
   ```bash
   # For advanced users, explore:
   01_Main(GPU).ipynb
   ```

## 📖 Learning Path

### Beginner Track
1. **Temperature & Tokenization** → Understanding LLM basics
2. **Prompt Templates** → Structured prompting
3. **System vs Human Messages** → Context control
4. **Conversation Memory** → Maintaining context

### Intermediate Track
5. **Text Summarization** → Practical applications
6. **Vector Embeddings** → Semantic search foundations
7. **Sentiment Analysis** → Real-world NLP tasks

### Advanced Track
8. **Vector Databases** → RAG implementation
9. **Multi-Provider Comparison** → OpenAI vs Groq vs Local
10. **Production RAG Pipeline** → Complete system design

## 🔧 Key Features

### Multi-Provider Support
- **Cloud APIs**: OpenAI, Groq for high-performance inference
- **Local Models**: Ollama, Transformers for privacy and cost control
- **Hybrid Approaches**: Combining cloud and local models

### Comprehensive Examples
- **String Summarization**: Basic text processing
- **PDF Processing**: Document analysis workflows
- **Web Scraping**: Real-time content analysis
- **Aspect-Based Sentiment**: Detailed opinion mining

### Production-Ready Patterns
- **Error Handling**: Robust API interaction patterns
- **Token Management**: Cost optimization strategies
- **Memory Efficiency**: Conversation context management
- **Scalable Architecture**: RAG pipeline design

## 💡 Use Cases Demonstrated

1. **Document Intelligence**: PDF summarization and analysis
2. **Customer Feedback Analysis**: Aspect-based sentiment mining
3. **Knowledge Base Integration**: RAG for enterprise data
4. **Conversational AI**: Context-aware chatbots
5. **Content Generation**: Template-based text creation

## 📊 Performance Considerations

### Model Selection Guide
- **GPT-4o-mini**: Cost-effective, high-quality general tasks
- **GPT-3.5-turbo**: Balanced performance and cost
- **Groq Llama3-70b**: Ultra-fast inference for production
- **Local Llama3.2**: Privacy-first, offline capable

### Cost Optimization
- Token counting and management examples
- Temperature tuning for deterministic vs creative outputs
- Efficient prompt design patterns

## 🔒 Security & Best Practices

- **API Key Management**: 
  - **Google Colab**: Use Colab's built-in Secrets feature (🔑 icon) for secure storage
  - **Local Development**: Use environment variables and secure storage
- **Rate Limiting**: Handling API quotas and limits
- **Error Handling**: Robust failure recovery patterns
- **Data Privacy**: Local model options for sensitive data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive examples with documentation
4. Test with multiple LLM providers
5. Submit a pull request

## 📄 License

This project is intended for educational purposes. Please review the licensing terms of individual LLM providers and frameworks used.

## 👤 Author

**Repository maintained by**: [goagiq](https://github.com/goagiq)  
**Course materials**: GenAI-LLM-101 - Large Language Model Engineering Fundamentals

## 🆘 Troubleshooting

### Common Issues

**Google Colab Specific**:
- Use Colab Secrets (🔑 icon) instead of environment variables for API keys
- Libraries install automatically via `!pip` commands in notebooks
- GPU/TPU access may be limited based on Colab tier

**API Key Errors**:
- Ensure API keys are properly set in Colab Secrets or environment variables
- Check API key validity and account credits

**Model Loading Issues**:
- For local models, ensure sufficient RAM/VRAM
- Check Ollama service status for local deployments

**Dependency Conflicts**:
- In Colab: Restart runtime after major library installations
- For local: Use virtual environments for isolation
- Check specific version requirements in individual notebooks

### Getting Help

- Review individual notebook documentation
- Check framework-specific documentation
- Open issues for bugs or enhancement requests

## 🔄 Recent Updates

- **2025-07-21**: Repository cleaned and all external references removed
- **2025**: Updated to latest LangChain API patterns
- **2024**: Added Groq integration and local model support
- **2024**: Expanded RAG implementations with Haystack

---

**Note**: This repository represents a comprehensive learning journey through modern LLM engineering. Start with the basics and progressively work through more advanced concepts. Each notebook is designed to be self-contained while building upon previous concepts.
