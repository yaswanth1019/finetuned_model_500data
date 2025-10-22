from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
from datetime import datetime
import time
import os
import logging
import sys
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
model = None
tokenizer = None
model_loaded = False
model_load_time = 0
server_start_time = None

# Production Configuration - Only Fine-tuned Mistral
BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.1"
ADAPTER_REPO = "Thala1019/tirumala-mistral-7b"
HF_TOKEN = os.getenv("HF_TOKEN")

logger.info("🕉️ Production mode: Using fine-tuned Mistral-7B for Tirumala temple")

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="Question about Tirumala temple")
    max_tokens: Optional[int] = Field(150, ge=10, le=500, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, ge=0.1, le=2.0, description="Generation temperature")

class QuestionResponse(BaseModel):
    answer: str
    question: str
    timestamp: str
    processing_time: float
    model_info: str

def load_model_blocking():
    """Load your fine-tuned Mistral model - Production only"""
    global model, tokenizer, model_loaded, model_load_time
    
    logger.info("=" * 60)
    logger.info("🕉️ LOADING TIRUMALA TEMPLE CHATBOT MODEL")
    logger.info("=" * 60)
    
    load_start = time.time()
    
    try:
        logger.info(f"📦 Base Model: {BASE_MODEL_NAME}")
        logger.info(f"🎯 Fine-tuned Adapter: {ADAPTER_REPO}")
        logger.info(f"🔑 HF Token: {'✅ Set' if HF_TOKEN else '❌ Not set'}")
        
        # Validate HF Token
        if not HF_TOKEN:
            logger.error("❌ HF_TOKEN is required for your fine-tuned model!")
            logger.error("Set HF_TOKEN environment variable with your Hugging Face token")
            logger.error("Get token from: https://huggingface.co/settings/tokens")
            sys.exit(1)
        
        # Import libraries
        logger.info("📚 Importing transformers and dependencies...")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        import torch
        
        # Check device capabilities
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        
        logger.info(f"💻 Device: {device}")
        logger.info(f"🖥️ GPU Count: {gpu_count}")
        logger.info(f"💾 GPU Memory: {gpu_memory:.1f} GB" if gpu_memory > 0 else "💾 Using CPU")
        
        # Production: Load fine-tuned Mistral model
        logger.info("⚙️ Setting up 4-bit quantization for production...")
        
        # Quantization config to reduce 15GB model to ~4GB
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load base Mistral model first
        logger.info("📥 Downloading and loading base Mistral-7B model...")
        logger.info("⏳ This may take 3-5 minutes for first download...")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            token=HF_TOKEN,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        logger.info("📖 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_NAME,
            token=HF_TOKEN
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'right'
        
        logger.info("✅ Base Mistral-7B model and tokenizer loaded!")
        
        # Load your fine-tuned adapter - This is the key part!
        logger.info("🎯 Loading your Tirumala temple fine-tuned adapter...")
        logger.info(f"📡 Downloading from: {ADAPTER_REPO}")
        
        try:
            model = PeftModel.from_pretrained(
                base_model, 
                ADAPTER_REPO,
                token=HF_TOKEN
            )
            logger.info("🎉 Fine-tuned Tirumala adapter loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Failed to load fine-tuned adapter: {str(e)}")
            logger.error("💡 Check if your Hugging Face repo is accessible")
            logger.error(f"💡 Verify repo exists: https://huggingface.co/{ADAPTER_REPO}")
            sys.exit(1)
        
        # Set model to evaluation mode
        model.eval()
        
        # Test the model with a Tirumala-specific question
        logger.info("🧪 Testing fine-tuned model with Tirumala temple question...")
        
        test_prompts = [
            "What are the temple visiting hours?",
            "Tell me about Tirumala temple darshan.",
            "How to book tickets for Tirumala?"
        ]
        
        for i, test_prompt in enumerate(test_prompts[:1]):  # Test just one for startup speed
            # Format prompt like your training data
            formatted_prompt = f"Question: {test_prompt}\nAnswer:"
            inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True)
            
            # Move to device if using GPU
            if torch.cuda.is_available() and hasattr(model, 'device'):
                device_obj = next(model.parameters()).device
                inputs = {k: v.to(device_obj) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            test_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = test_output[len(formatted_prompt):].strip()
            logger.info(f"🧪 Test Q: {test_prompt}")
            logger.info(f"🧪 Test A: {answer[:100]}...")
        
        model_loaded = True
        model_load_time = time.time() - load_start
        
        logger.info("=" * 60)
        logger.info("🎉 TIRUMALA TEMPLE CHATBOT READY!")
        logger.info(f"⏱️ Model Load Time: {model_load_time:.2f} seconds")
        logger.info("🧠 Model: Fine-tuned Mistral-7B for Tirumala Temple")
        logger.info("💾 Memory Usage: ~4GB (4-bit quantized)")
        logger.info("🎯 Specialization: Tirumala Temple Q&A")
        logger.info("=" * 60)
        
    except Exception as e:
        error_msg = f"❌ CRITICAL ERROR loading model: {str(e)}"
        logger.error(error_msg)
        logger.error("🛑 Server will not start!")
        logger.error("💡 Check your HF_TOKEN and internet connection")
        logger.error(f"💡 Verify model repo: https://huggingface.co/{ADAPTER_REPO}")
        sys.exit(1)

def generate_answer(prompt: str, max_tokens: int = 150, temperature: float = 0.7) -> str:
    """Generate answer using your fine-tuned Tirumala model"""
    try:
        # Format prompt like your training data
        formatted_prompt = f"Question: {prompt}\nAnswer:"
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True)
        
        # Move to device if using GPU
        import torch
        if torch.cuda.is_available() and hasattr(model, 'device'):
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                top_k=50,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer part
        answer = full_response[len(formatted_prompt):].strip()
        
        # Clean up answer
        if '\n' in answer:
            answer = answer.split('\n')[0].strip()
        
        # Remove common artifacts
        answer = answer.replace("Question:", "").replace("Answer:", "").strip()
        
        return answer if answer else "I apologize, but I couldn't generate a proper response. Please try rephrasing your question about Tirumala temple."
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return "I encountered an error while processing your question. Please try again."

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP - Load model before starting server
    logger.info("🚀 Tirumala Temple Chatbot startup initiated...")
    
    # This blocks until model is fully loaded
    load_model_blocking()
    
    global server_start_time
    server_start_time = time.time()
    
    logger.info("✅ Server ready to answer questions about Tirumala temple!")
    
    yield
    
    # SHUTDOWN
    logger.info("🛑 Tirumala Temple Chatbot shutdown...")

# Create FastAPI app
app = FastAPI(
    title="🕉️ Tirumala Temple AI Chatbot",
    description="Fine-tuned Mistral-7B model specialized for Tirumala temple information and guidance. Powered by Thala1019/tirumala-mistral-7b",
    version="1.0.0-production",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    uptime = time.time() - server_start_time if server_start_time else 0
    return {
        "message": "🕉️ Om Namo Venkatesaya - Tirumala Temple AI Chatbot",
        "status": "✅ Ready - Specialized for Tirumala temple information",
        "model_name": f"{BASE_MODEL_NAME}",
        "fine_tuned_adapter": ADAPTER_REPO,
        "model_type": "Fine-tuned Mistral-7B for Tirumala temple Q&A",
        "model_load_time": f"{model_load_time:.2f}s",
        "uptime": f"{uptime:.2f}s",
        "specialization": "Darshan, Temple timings, Accommodation, Prasadam, Temple history",
        "endpoints": {
            "health": "/health",
            "ask": "/ask", 
            "docs": "/docs",
            "temple-info": "/temple-info"
        }
    }

@app.get("/health")
async def health_check():
    import torch
    uptime = time.time() - server_start_time if server_start_time else 0
    
    return {
        "status": "ready",
        "model_loaded": True,
        "model_name": BASE_MODEL_NAME,
        "fine_tuned_adapter": ADAPTER_REPO,
        "model_load_time": f"{model_load_time:.2f}s",
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "server_uptime": f"{uptime:.2f}s",
        "specialization": "Tirumala Temple Information",
        "memory_usage": "~4GB (4-bit quantized)",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask questions about Tirumala temple - powered by fine-tuned Mistral-7B"""
    
    start_time = time.time()
    
    # Generate answer using your fine-tuned model
    answer = generate_answer(
        request.question, 
        request.max_tokens, 
        request.temperature
    )
    
    processing_time = time.time() - start_time
    
    return QuestionResponse(
        answer=answer,
        question=request.question,
        timestamp=datetime.now().isoformat(),
        processing_time=round(processing_time, 2),
        model_info=f"Tirumala-specialized Mistral-7B (loaded in {model_load_time:.1f}s)"
    )

@app.get("/temple-info")
async def temple_info():
    """Get common Tirumala temple information"""
    return {
        "temple_name": "Sri Venkateswara Temple",
        "location": "Tirumala, Andhra Pradesh, India",
        "deity": "Lord Venkateswara (Balaji)",
        "significance": "One of the most visited and richest temples in the world",
        "model_info": {
            "base_model": BASE_MODEL_NAME,
            "fine_tuned_adapter": ADAPTER_REPO,
            "specialization": "Trained on Tirumala temple-specific data"
        },
        "common_questions": [
            "What are the temple visiting hours?",
            "How to book darshan tickets online?",
            "What are the different types of darshan?",
            "Where to stay in Tirumala?",
            "What is the significance of Tirumala temple?",
            "How to reach Tirumala from major cities?",
            "What are the temple dress code requirements?",
            "What prasadam is available at the temple?",
            "What are the accommodation options in Tirumala?",
            "How to perform special sevas at the temple?"
        ],
        "sample_usage": {
            "endpoint": "/ask",
            "method": "POST",
            "payload": {
                "question": "What are the temple visiting hours?",
                "max_tokens": 150,
                "temperature": 0.7
            }
        }
    }

if __name__ == "__main__":
    logger.info("🕉️ Starting Tirumala Temple Chatbot with fine-tuned Mistral-7B...")
    port = int(os.getenv("PORT", 8000))
    
    # Server will only start AFTER model is loaded
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )
