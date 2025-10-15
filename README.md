# NLP Experiments: RAG & Agentic AI

This repository contains two Natural Language Processing experiments demonstrating advanced AI capabilities using Google's Gemini API.

## 📋 Table of Contents
- [Experiments Overview](#experiments-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Experiment 1: RAG-Based Gen-AI Tool](#experiment-1-rag-based-gen-ai-tool)
- [Experiment 2: Image Captioning Agent](#experiment-2-image-captioning-agent)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Author](#author)

## 🔬 Experiments Overview

### Experiment 7: Image Captioning Agent
An Agentic AI system that generates intelligent captions for images using Google's Gemini 1.5 Flash model. The agent can analyze images and provide simple, detailed, or custom-prompted captions.

### Experiment 8: Quantum Computing RAG System
A Retrieval Augmented Generation (RAG) system focused on quantum computing knowledge. Uses FAISS vector database and sentence transformers to provide grounded, accurate responses to quantum computing queries.

## 📦 Prerequisites

- Python 3.8 or higher
- Google Gemini API Key
- Internet connection for API calls

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/Jash-Vora/NLP-7th-8th-exp.git
```

2. **Create a virtual environment** (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

### Requirements File
Create a `requirements.txt` file with the following dependencies:
```
google-generativeai>=0.3.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
numpy>=1.24.0
requests>=2.31.0
beautifulsoup4>=4.12.0
PyPDF2>=3.0.0
pillow>=10.0.0
python-dotenv>=1.0.0
```

## ⚙️ Configuration

1. **Get your Gemini API Key**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create or sign in to your Google account
   - Generate an API key

2. **Create a `.env` file** in the project root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_GEMINI_API_KEY=your_gemini_api_key_here
```

**Important**: Never commit your `.env` file to version control. Add it to `.gitignore`:
```bash
echo ".env" >> .gitignore
```

## 🤖 Experiment 1: RAG-Based Gen-AI Tool

### Description
A Retrieval Augmented Generation system that combines information retrieval with generative AI to answer questions about quantum computing. The system uses:
- **FAISS** for efficient vector similarity search
- **Sentence Transformers** for generating embeddings
- **Gemini 2.5 Flash** for generating contextual responses

### Features
- Initialize knowledge base with quantum computing fundamentals
- Add documents from multiple sources (text, PDF, URLs)
- Chunk documents for efficient retrieval
- Retrieve top-k relevant chunks for queries
- Generate grounded responses using retrieved context
- Persistent vector storage (saved to disk)

### Running Experiment 1

```bash
python exp7.py
```

### Usage Examples

**Basic Usage:**
```python
from exp7 import QuantumRAGAgent

# Initialize agent
agent = QuantumRAGAgent()

# Initialize with quantum computing knowledge
agent.initialize_quantum_knowledge_base()

# Query the system
result = agent.query("What is quantum superposition?")
print(result['response'])
```

**Adding Documents:**
```python
# Add text
agent.add_text("Your quantum computing text here", 
               metadata={"topic": "custom"})

# Add PDF
agent.add_pdf("quantum_paper.pdf", 
              metadata={"source": "research_paper"})

# Add URL
agent.add_url("https://example.com/quantum-article",
              metadata={"source": "web"})
```

**Advanced Querying:**
```python
# Get detailed results
result = agent.query("Explain Shor's algorithm", n_results=5)
print(f"Response: {result['response']}")
print(f"Sources: {result['sources']}")
print(f"Retrieved chunks: {result['retrieved_docs']}")

# Get knowledge base statistics
stats = agent.get_collection_stats()
print(f"Total chunks: {stats['total_chunks']}")
```

### Vector Database
The FAISS index is automatically saved to `./faiss_index/` and persists between runs. To reset:
```bash
rm -rf faiss_index/
```

## 🖼️ Experiment 2: Image Captioning Agent

### Description
An Agentic AI system that analyzes images and generates descriptive captions using Google's Gemini 1.5 Flash vision model. Supports single images and batch processing.

### Features
- Generate simple, concise captions
- Generate detailed, descriptive captions
- Custom prompt-based captioning
- Batch process entire folders
- Support for multiple image formats (JPG, PNG, BMP, GIF, TIFF)
- Automatic image format conversion

### Running Experiment 2

```bash
python exp8.py
```

### Usage Examples

**Basic Usage:**
```python
from exp8 import GeminiImageCaptionAgent

# Initialize agent
agent = GeminiImageCaptionAgent()

# Simple caption
caption = agent.generate_simple_caption("image.jpg")
print(caption)

# Detailed caption
detailed = agent.generate_detailed_caption("image.jpg")
print(detailed)
```

**Custom Prompts:**
```python
# Custom caption with specific instructions
custom_caption = agent.generate_caption(
    "image.jpg",
    prompt="Describe this image focusing on colors and composition",
    max_tokens=100,
    temperature=0.8
)
print(custom_caption)
```

**Batch Processing:**
```python
# Process all images in a folder
results = agent.batch_caption_images(
    "./images",
    output_file="captions.txt"
)

for filename, caption in results.items():
    print(f"{filename}: {caption}")
```

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- GIF (.gif)
- TIFF (.tiff)


## 🔧 Troubleshooting

### Common Issues

**1. API Key Not Found**
```
Error: Gemini API key not found
```
**Solution:** Ensure your `.env` file exists and contains the correct API key.

**2. Module Not Found**
```
ModuleNotFoundError: No module named 'faiss'
```
**Solution:** Install missing dependencies:
```bash
pip install faiss-cpu
```

**3. Image File Not Found**
```
FileNotFoundError: Image file not found
```
**Solution:** Verify the image path is correct and the file exists.

**4. API Quota Exceeded**
```
API quota exceeded
```
**Solution:** Check your Gemini API usage limits at [Google AI Studio](https://makersuite.google.com/).

**5. Response Blocked by Safety Filters**
```
Response was blocked by safety filters
```
**Solution:** Try rephrasing your query or adjust the content being processed.

### Performance Tips

1. **RAG System:**
   - Adjust `chunk_size` and `overlap` for better retrieval
   - Increase `n_results` for more context (but slower)
   - Use specific queries for better results

2. **Image Captioning:**
   - Lower `temperature` (0.3-0.5) for more consistent captions
   - Higher `temperature` (0.7-0.9) for more creative captions
   - Reduce `max_tokens` for shorter captions

## 🎯 Key Concepts

### Retrieval Augmented Generation (RAG)
- Combines information retrieval with generative AI
- Reduces hallucination by grounding responses
- Uses latest data without model retraining
- Efficient vector similarity search with FAISS

### Agentic AI
- Autonomous systems that perceive, plan, and act
- Goal-oriented behavior
- Integration with external tools and APIs
- Context-aware decision making

## 📝 Example Outputs

### RAG System Query
```
Query: What is quantum superposition?
Response: Quantum superposition is a fundamental principle where qubits 
can exist in a combination of |0⟩ and |1⟩ states simultaneously, unlike 
classical bits that can only be in one state at a time.
Sources used: 2
Retrieved chunks: 5
```

### Image Captioning
```
Simple: A modern office workspace with a laptop and coffee cup.
Detailed: The image shows a bright, minimalist office workspace featuring 
a silver laptop on a white desk, accompanied by a white ceramic coffee 
cup. Natural light streams through a nearby window, creating a productive 
and calm atmosphere.
```

## 👤 Author

**Jash Vora**
- UID: 2022600065
- Branch: AIML
- Batch: A1
- Course: Natural Language Processing

## 📄 License

This project is created for educational purposes as part of NLP coursework.

## 🙏 Acknowledgments

- Google Gemini API for powerful AI capabilities
- FAISS library for efficient vector search
- Sentence Transformers for quality embeddings
- Beautiful Soup for web scraping

## 📞 Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review [Google AI Studio Documentation](https://ai.google.dev/docs)
3. Open an issue in this repository

---

**Note**: Ensure you comply with Google's API usage policies and rate limits when using these experiments.
