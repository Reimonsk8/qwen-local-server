#!/usr/bin/env python3
"""
Qwen2.5-Coder-7B-Instruct - Local GPU Server (4-bit)
"""

import os
import json
import time
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError, field_validator
import uvicorn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer
)
from threading import Thread

# === Environment Setup ===
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["SAFETENSORS_FAST_GPU"] = "1"

# === Configuration ===
MODEL_NAME = "."  # Load from current directory
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_4BIT = True

# === Main Execution ===
if __name__ == "__main__":
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        print(f"🎮 GPU: {gpu_name} ({vram} GB VRAM)")
        print(f"🔧 CUDA Version: {torch.version.cuda}")
        print(f"🔥 PyTorch CUDA Available: {torch.cuda.is_available()}")
        torch.cuda.empty_cache()
        print("🧹 GPU cache cleared")
        DEVICE = "cuda"
        print(f"✅ Forcing GPU usage: {DEVICE}")
    else:
        print("⚠️  No GPU detected – will run on CPU (slow)")
        DEVICE = "cpu"

    print("🧠 Loading model and tokenizer from local folder...")
    print(f"📍 Model path: {os.path.abspath(MODEL_NAME)}")
    print(f"🖥️  Device: {DEVICE}")
    if USE_4BIT and DEVICE == "cuda":
        print("📦 Using 4-bit quantization (bitsandbytes) for GPU")
    elif DEVICE == "cuda":
        print("🚀 Using full precision on GPU")
    else:
        print("⚠️  Running on CPU (no quantization)")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

        if USE_4BIT and DEVICE == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map="auto" if DEVICE == "cuda" else None,
                trust_remote_code=True,
            )
            if DEVICE == "cpu":
                model = model.to("cpu")

        model.eval()
        print("✅ Model loaded successfully!")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        exit(1)

    app = FastAPI(title="Qwen2.5-Coder-7B-Instruct (Local)")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        if request.url.path == "/v1/chat/completions":
            body = await request.body()
            body_str = body.decode('utf-8')
            if len(body_str) > 100000:
                print(f"⚠️ Large request detected: {len(body_str)} characters")
                try:
                    body_json = json.loads(body_str)
                    print(f"🔍 Request structure: model={body_json.get('model', 'N/A')}, messages={len(body_json.get('messages', []))}")
                    if 'tools' in body_json and len(body_json['tools']) > 0:
                        print(f"🔍 First tool: {json.dumps(body_json['tools'][0], indent=2)[:500]}...")
                except:
                    print(f"🔍 Raw request body (first 500 chars): {body_str[:500]}...")
            else:
                try:
                    body_json = json.loads(body_str)
                    print(f"🔍 Request: {json.dumps(body_json, indent=2)}")
                except:
                    print(f"🔍 Raw request body: {body_str}")
        response = await call_next(request)
        return response

    class ContentPart(BaseModel):
        type: str = Field(..., description="Content part type")
        text: str = Field(..., description="Content part text")

    class Message(BaseModel):
        role: str = Field(..., description="The role of the message sender")
        content: str | list[ContentPart] = Field(..., description="The content of the message")

    class ToolFunction(BaseModel):
        name: str = Field(..., description="Function name")
        description: str = Field(default="", description="Function description")
        parameters: dict = Field(default_factory=dict, description="Function parameters")

    class Tool(BaseModel):
        type: str = Field(default="function", description="Tool type")
        function: ToolFunction = Field(..., description="Tool function definition")

    class ChatRequest(BaseModel):
        model: str = Field(default="qwen2.5-coder-7b-instruct", description="Model identifier")
        messages: list[Message] = Field(..., min_length=1, description="List of messages")
        max_tokens: int = Field(default=256, ge=1, le=4096, description="Maximum tokens to generate")
        temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
        top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Top-p sampling parameter")
        stream: bool = Field(default=False, description="Whether to stream the response")
        tools: list | None = Field(default=None, description="Tools available to the model")
        
        @field_validator('tools', mode='before')
        @classmethod
        def validate_tools(cls, v):
            if v is None:
                return None
            if isinstance(v, list):
                print(f"🔧 Tools received: {len(v)} tools")
                for i, tool in enumerate(v):
                    if isinstance(tool, dict):
                        tool_name = tool.get('function', {}).get('name', f'tool_{i}')
                        print(f"🔧 Tool {i}: {tool_name}")
            return v
        
        @field_validator('temperature')
        @classmethod
        def validate_temperature(cls, v):
            if v < 0.0:
                raise ValueError("temperature must be >= 0.0")
            return v

        class Config:
            extra = "allow"

    @app.get("/")
    async def root():
        return {"status": "running", "mode": "local", "device": DEVICE}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.body()
            body_str = body.decode('utf-8')
            try:
                request_data = json.loads(body_str)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
            
            if 'messages' not in request_data:
                raise HTTPException(status_code=400, detail="Missing 'messages' field")
            
            try:
                chat_request = ChatRequest(**request_data)
            except ValidationError as e:
                print(f"❌ Validation error details: {e}")
                raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")
            
            print(f"🔍 Received request model: {chat_request.model}")
            print(f"🔍 Request messages count: {len(chat_request.messages)}")
            if chat_request.tools:
                print(f"🔍 Request tools count: {len(chat_request.tools)}")
            
            if not chat_request.messages:
                raise HTTPException(status_code=400, detail="At least one message is required")
            
            valid_roles = {"user", "assistant", "system"}
            messages = []
            for msg in chat_request.messages:
                if msg.role not in valid_roles:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Invalid role '{msg.role}'. Must be one of: {', '.join(valid_roles)}"
                    )
                
                if isinstance(msg.content, str):
                    if not msg.content.strip():
                        raise HTTPException(status_code=400, detail="Message content cannot be empty")
                    messages.append({"role": msg.role, "content": msg.content})
                elif isinstance(msg.content, list):
                    if not msg.content:
                        raise HTTPException(status_code=400, detail="Message content cannot be empty")
                    text_parts = []
                    for part in msg.content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                text = part.get("text", "")
                                if text.strip():
                                    text_parts.append(text)
                        elif hasattr(part, 'type') and hasattr(part, 'text'):
                            if part.type == "text" and part.text.strip():
                                text_parts.append(part.text)
                    full_text = "".join(text_parts)
                    if not full_text.strip():
                        raise HTTPException(status_code=400, detail="Message content cannot be empty")
                    messages.append({"role": msg.role, "content": full_text})
            
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to format messages: {str(e)}")

            do_sample = chat_request.temperature > 0.0
            gen_temperature = chat_request.temperature if do_sample else 1.0

            input_tensors = tokenizer(prompt, return_tensors="pt").to(DEVICE)

            if chat_request.stream:
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                
                generation_kwargs = dict(
                    input_ids=input_tensors.input_ids,
                    attention_mask=input_tensors.attention_mask,
                    max_new_tokens=chat_request.max_tokens,
                    temperature=gen_temperature,
                    top_p=chat_request.top_p,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer,
                )

                thread = Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()

                async def generate_stream():
                    # Initial assistant role chunk
                    init_chunk = {
                        "id": f"chatcmpl-{int(time.time() * 1000)}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": chat_request.model,
                        "choices": [{
                            "delta": {"role": "assistant"},
                            "index": 0,
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(init_chunk)}\n\n"

                    for new_text in streamer:
                        if new_text:
                            chunk = {
                                "id": f"chatcmpl-{int(time.time() * 1000)}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": chat_request.model,
                                "choices": [{
                                    "delta": {"content": new_text},
                                    "index": 0,
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"

                    # Final stop chunk
                    final_chunk = {
                        "id": f"chatcmpl-{int(time.time() * 1000)}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": chat_request.model,
                        "choices": [{
                            "delta": {},
                            "index": 0,
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    generate_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no"
                    }
                )
            else:
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_tensors.input_ids,
                        attention_mask=input_tensors.attention_mask,
                        max_new_tokens=chat_request.max_tokens,
                        temperature=gen_temperature,
                        top_p=chat_request.top_p,
                        do_sample=do_sample,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                prompt_len = input_tensors.input_ids.shape[1]
                generated_ids = outputs[0]
                reply_ids = generated_ids[prompt_len:]
                reply = tokenizer.decode(reply_ids, skip_special_tokens=True).strip()

                prompt_tokens = prompt_len
                completion_tokens = reply_ids.shape[0]

                response_data = {
                    "id": f"chatcmpl-{int(time.time() * 1000)}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": chat_request.model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                }
                return response_data

        except HTTPException:
            raise
        except Exception as e:
            print(f"Internal error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request: Request, exc: ValidationError):
        print(f"❌ Validation error: {exc}")
        return {"error": {"message": "Validation error", "details": exc.errors(), "type": "validation_error"}}

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        print(f"❌ General exception: {exc}")
        return {"error": {"message": f"Internal error: {str(exc)}", "type": "internal_error"}}

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "device": DEVICE,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }

    PORT = 8001
    print("🚀 Starting server on http://localhost:8001")
    print("🎯 Use /v1/chat/completions for inference")
    print("=" * 50)
    uvicorn.run(app, host="127.0.0.1", port=PORT)