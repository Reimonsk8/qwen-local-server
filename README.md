# Qwen Local Server

![Qwen Architecture](assets/qwen.png)

A full-featured local server implementation for running the **Qwen** (Qwen2.5) large language model with **GPU acceleration**, **quantization support** and **OpenAI-compatible API**.

Perfect for developers who want to run powerful open-source LLMs locally—securely, privately, and efficiently.

## ✨ Features

- ✅ **OpenAI-compatible REST API** (`/v1/chat/completions`, `/v1/completions`)
- 🚀 **GPU-accelerated inference** via Hugging Face Transformers
- ⚡ **4-bit & 8-bit quantization** support for low-memory GPUs
- 🔒 **API key authentication** (configurable via `.env`)
- 📦 **Automatic model downloading**
- 🧪 **CLI utility** for testing and model management
- 🧠 **Supports Qwen2.5 series** models
- 🔄 **Streaming responses** support
- 🌐 **CORS-enabled** for web frontend integration

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (NVIDIA, ≥8GB VRAM recommended; 6GB+ with quantization)
- Node.js 18+ and npm (for `qwencoder` tokenizer CLI)
- Git

## 🚀 Quick Start

1. **Install the qwencoder CLI tool**
    First, ensure you have Node.js and npm installed. Then install qwencoder globally so it’s available system-wide. This tool uses the tokenizer configuration from your local model path (as defined in your .env file):

    ```bash
    npm install -g qwencoder
    ```

2. **Clone the repository**
   ```bash
   git clone https://github.com/Reimonsk8/qwen-local-server.git
   cd qwen-local-server
   ```

3. **Set up Python environment**
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/macOS
   # source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Copy and configure environment variables
   cp .env-example .env
   # Edit .env to set your API key and model path if needed
   ```

## 📥 Downloading the Model

### Option A: Manual Download (Recommended for control)
1. Create models directory:
   ```bash
   mkdir -p models
   ```

2. Download the model from Hugging Face Hub:
   - [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
   - [Qwen Model Hub](https://huggingface.co/Qwen)

3. Place the downloaded files in the following structure:
   ```
   models/
   └── Qwen2.5-7B-Instruct/
       ├── config.json
       ├── model.safetensors          # or pytorch_model.bin
       ├── tokenizer.json
       ├── tokenizer_config.json
       ├── vocab.json
       ├── special_tokens_map.json
       └── generation_config.json
   ```

   > **Note**: Make sure all tokenizer files are present to avoid errors. The tokenizer may fail without `tokenizer_config.json` and other supporting files.

## 🚀 Running the Server

### Native Installation
```bash
python serve_qwen_local.py
```
Server will be available at: http://localhost:8000/v1

### Python (OpenAI SDK)
```python
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-secret-api-key-here"
)

        completion = client.chat.completions.create(
            model="qwen2.5-7b-instruct",  # Default model, matches the one in .env
            messages=[{"role": "user", "content": "Write a Python function to reverse a string."}],
            stream=False  # or True for streaming
        )

print(completion.choices[0].message.content)
```

### curl
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-secret-api-key-here" \
  -H "Content-Type: application/json" \
            "model": "qwen2.5-7b-instruct",  # Using the default model
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

{{ ... }}

The following environment variables can be set to customize the server's behavior:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required but value can be anything) | API key |
| `OPENAI_MODEL` | `qwen2.5-7b-instruct` | Model to use |
| `MAX_MODEL_LENGTH` | `4096` | Maximum context length |
| `LOAD_IN_4BIT` | `true` | Enable 4-bit quantization |
| `LOAD_IN_8BIT` | `false` | Enable 8-bit quantization (mutually exclusive with 4-bit) |
| `DEVICE` | `cuda` | `cuda` for GPU or `cpu` (not recommended) |
| `HOST` | `0.0.0.0` | Host to bind the server to |
{{ ... }}

```bash
python -m pytest tests/
```

## 📁 Project Structure

```
qwen-local-server/
├── serve_qwen_local.py    # Main FastAPI server
├── download_model.py      # Model downloader
├── test_api.py            # API test utility
├── requirements.txt       # Python dependencies
├── .env-example           # Example environment variables
├── assets/                # Static assets (images, etc.)
├── models/                # Store downloaded models here
└── tests/                 # Test suite
```

> **Note on Model Variants**:
> - `Qwen2.5-7B-Instruct`: General-purpose instruction model (recommended default)
> - `Qwen2.5-Coder-7B-Instruct`: Specialized for code generation tasks
> Update the `OPENAI_MODEL` in `.env` and model path as needed for different variants.
## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgements
{{ ... }}
- Qwen Team for the amazing open-source models
- Hugging Face for the Transformers library and model hosting
- OpenAI for the API specification
