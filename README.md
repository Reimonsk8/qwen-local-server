# Qwen Local Server

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

1. **Clone the repository**
   ```bash
   git clone https://github.com/Reimonsk8/qwen-local-server.git
   cd qwen-local-server
   ```

2. **Set up Python environment**
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/macOS
   # source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
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
       ├── model.safetensors  # or pytorch_model.bin
       └── tokenizer.json
   ```

### Option B: Auto-download via CLI (experimental)
```bash
python download_model.py --model-name Qwen/Qwen2.5-7B-Instruct --output-dir ./models
```
*Note: Requires Hugging Face token if the model is gated.*

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
    model="qwen2.5-coder-7b-instruct",
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
  -d '{
    "model": "qwen2.5-coder-7b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

### Streaming (SSE)
Set "stream": true to receive tokens in real-time via Server-Sent Events (SSE).

### CLI Tools

#### Tokenizer
```bash
qwencoder "Hello, world!" --model ./models/Qwen2.5-7B-Instruct
```

#### Test Server Health
```bash
python test_api.py --base-url http://localhost:8000/v1 --api-key your-secret-api-key-here
```

## ⚙️ Configuration Options (.env)

The following environment variables can be set to customize the server's behavior:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | Your API key for authentication |
| `OPENAI_MODEL` | `qwen2.5-7b-instruct` | Model to use |
| `MODEL_PATH` | `./models/Qwen2.5-7B-Instruct` | Path to model files |
| `MAX_MODEL_LENGTH` | `4096` | Maximum context length |
| `LOAD_IN_4BIT` | `true` | Enable 4-bit quantization |
| `LOAD_IN_8BIT` | `false` | Enable 8-bit quantization (mutually exclusive with 4-bit) |
| `DEVICE` | `cuda` | `cuda` for GPU or `cpu` (not recommended) |
| `HOST` | `0.0.0.0` | Host to bind the server to |
| `PORT` | `8000` | Port to run the server on |

## 🔒 Security Note

Never expose the server publicly without authentication.

## 🐞 Troubleshooting

If you encounter any issues, refer to the following table for solutions:

| Issue | Solution |
|-------|----------|
| **CUDA Out of Memory** | Enable `LOAD_IN_4BIT=true` or reduce `MAX_MODEL_LENGTH` |
| **Model not found** | Verify `MODEL_PATH` points to the correct directory and file structure |
| **Slow inference on CPU** | Use GPU for better performance; quantization has limited benefits on CPU |
| **Tokenizer errors** | Ensure `tokenizer.json` and `vocab.json` exist in the model directory |

## 🧪 Testing

Run unit and integration tests:

```bash
python -m pytest tests/
```

```
├── download_model.py      # Model downloader
├── test_api.py            # API test utility
├── .env-example           # Example environment variables
├── models/                # Store downloaded models here
└── tests/                 # Test suite
```
## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgements

- Qwen Team for the amazing open-source models
- Hugging Face for the Transformers library and model hosting
- OpenAI for the API specification
