import gradio as gr
from enum import StrEnum
from pathlib import Path
import hashlib
import openai
from gradio.themes import Soft
import logging
import time
from abc import ABC, abstractmethod
import warnings

# Suppress h11 protocol warnings (known Gradio issue)
warnings.filterwarnings("ignore", message=".*h11.*")
warnings.filterwarnings("ignore", message=".*LocalProtocolError.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Reduce noise from third-party libraries
logging.getLogger("h11").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

import datetime

def log_llm_interaction(chunk_num: int, total_chunks: int, query: str, response: str, log_file: str = "llm_interactions.log"):
    """
    Query/Response Logging
    Log every query sent to the LLM and its response to a separate file for analysis.
    
    Args:
        chunk_num: Current chunk number (1-indexed)
        total_chunks: Total number of chunks
        query: The prompt/query sent to the model
        response: The model's response
        log_file: Path to the log file (default: llm_interactions.log)
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "="*100 + "\n")
        f.write(f"TIMESTAMP: {timestamp}\n")
        f.write(f"CHUNK: {chunk_num} / {total_chunks}\n")
        f.write("="*100 + "\n\n")
        
        f.write(">>> QUERY SENT TO MODEL:\n")
        f.write("-"*100 + "\n")
        f.write(query)
        f.write("\n" + "-"*100 + "\n\n")
        
        f.write("<<< RESPONSE FROM MODEL:\n")
        f.write("-"*100 + "\n")
        f.write(response)
        f.write("\n" + "-"*100 + "\n\n")

def clear_llm_log(log_file: str = "llm_interactions.log"):
    """Clear the LLM interaction log file at the start of a new run."""
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"LLM INTERACTION LOG\n")
            f.write(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*100 + "\n\n")
        logging.info(f"* LLM interaction log initialized: {log_file}")
    except Exception as e:
        logging.warning(f"!  Could not initialize LLM log: {e}")

# Hardware Detection
MLX_AVAILABLE = False
CUDA_AVAILABLE = False
CPU_ONLY = False
DEVICE_TYPE = "unknown"

# Check for MLX (Apple Silicon)
try:
    import mlx_whisper
    MLX_AVAILABLE = True
    DEVICE_TYPE = "MLX"
    logging.info("+ MLX (Apple Silicon) detected and available")
except ImportError:
    logging.info("- MLX not available")

# Check for CUDA and PyTorch
try:
    import torch
    if torch.cuda.is_available():
        CUDA_AVAILABLE = True
        DEVICE_TYPE = "CUDA"
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        logging.info(f"+ CUDA available with {device_count} device(s): {device_name}")
        logging.info(f"   GPU Memory: {total_memory:.2f} GB")
    else:
        if not MLX_AVAILABLE:
            CPU_ONLY = True
            DEVICE_TYPE = "CPU"
            logging.info("!  CUDA not available, using CPU mode")
except ImportError:
    if not MLX_AVAILABLE:
        CPU_ONLY = True
        DEVICE_TYPE = "CPU"
        logging.warning("!  PyTorch not installed, defaulting to CPU mode")

# Attempt to import Hugging Face transformers
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import pipeline
    HF_AVAILABLE = True
    logging.info(f"+ Hugging Face Transformers available (using {DEVICE_TYPE})")
except ImportError:
    HF_AVAILABLE = False
    logging.warning("- Hugging Face Transformers not available")

# Check for BitsAndBytes (needed for 4-bit quantization on CUDA)
BITSANDBYTES_AVAILABLE = False
if CUDA_AVAILABLE and HF_AVAILABLE:
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes
        BITSANDBYTES_AVAILABLE = True
        logging.info("+ BitsAndBytes available for 4-bit quantization")
    except ImportError:
        logging.warning("-  BitsAndBytes not available - 4-bit quantization disabled")
        logging.warning("!  Install with: pip install bitsandbytes")

# Ensure at least one backend is available
if not MLX_AVAILABLE and not HF_AVAILABLE:
    raise ImportError("Please install either 'mlx-whisper' or 'transformers' with 'torch' to use this module.")

# Log final hardware configuration
logging.info("=" * 30)
logging.info(f"HARDWARE CONFIGURATION: {DEVICE_TYPE}")
logging.info(f"  MLX Available: {MLX_AVAILABLE}")
logging.info(f"  CUDA Available: {CUDA_AVAILABLE}")
logging.info(f"  CPU Only: {CPU_ONLY}")
logging.info(f"  Hugging Face: {HF_AVAILABLE}")
logging.info("=" * 30)

# Abstract Interfaces
class TranscriptionService(ABC):
    @abstractmethod
    def transcribe(self, audio_file: str) -> str:
        raise NotImplementedError("TranscriptionService: Subclasses must implement this method")

class SummarizationService(ABC):
    @abstractmethod
    def summarize(self, prompt: str) -> str:
        raise NotImplementedError("SummarizationService: Subclasses must implement this method")

# MLX Adapter implementation
class MLXWhisperAdapter(TranscriptionService):
    def __init__(self, model_name: str):
        if not MLX_AVAILABLE:
            raise ImportError("MLX is not available on this system.")
        self.model_name = model_name

    def transcribe(self, audio_file: str) -> str:
        logging.info("Transcribing with MLX Whisper (Apple Silicon): %s", self.model_name)
        result = mlx_whisper.transcribe(audio_file, path_or_hf_repo=self.model_name)
        text = result.get("text", "")
        if isinstance(text, list):
            text = " ".join(str(chunk) for chunk in text)
        return str(text)

class HuggingFaceWhisperAdapter(TranscriptionService):
    """Supports both CUDA and CPU backends with long-form audio support (longer than 30s)"""
    def __init__(self, model_name: str, lazy_load: bool = False):
        if not HF_AVAILABLE:
            raise ImportError("Hugging Face transformers not available.")
        self.model_name = model_name
        self.lazy_load = lazy_load
        self.pipe = None
        
        # Determine device and configuration
        if CUDA_AVAILABLE:
            self.device = 0
            self.torch_dtype = torch.float16
            self.device_name = torch.cuda.get_device_name(0)
            # Enable chunking for long audio files on CUDA (! super important for long files)
            self.chunk_length_s = 30  # Process in 30-second chunks
            self.batch_size = 8  # Batch size for CUDA
            logging.info(f"* Whisper adapter configured for CUDA on {self.device_name}")
            logging.info(f"  Long-form audio: enabled (chunk_length={self.chunk_length_s}s, batch_size={self.batch_size})")
        else:
            self.device = -1  # CPU
            self.torch_dtype = torch.float32
            self.device_name = "CPU"
            self.chunk_length_s = 30
            self.batch_size = 4 # Smaller batch for CPU
            logging.info("* Whisper adapter configured for CPU (this may be 30-100x slower than with CUDA/MLX)")
                
        if not lazy_load: # Load immediately if not lazy loading, means: the weights are loaded into memory with the first inference
            self._load_model()
    
    def _load_model(self):
        """Load the Whisper model into memory"""
        if self.pipe is not None:
            logging.info("!  Whisper model already loaded, skipping")
            return
        
        logging.info(f"* Loading Whisper model: {self.model_name}")
        log_cuda_memory_usage("Before Whisper Load") # a lot of VRAM memory testing in the code (there is)
        
        try:
            # For CUDA with long audio support, we use return_timestamps and chunking
            pipeline_kwargs = {
                "model": self.model_name,
                "device": self.device,
                "torch_dtype": self.torch_dtype,
            }
            
            # Add chunking parameters for long-form audio
            if CUDA_AVAILABLE:
                pipeline_kwargs["chunk_length_s"] = self.chunk_length_s
                pipeline_kwargs["batch_size"] = self.batch_size
            
            self.pipe = pipeline("automatic-speech-recognition", **pipeline_kwargs)
            logging.info(f"+ Whisper model loaded successfully on {self.device_name}")
            log_cuda_memory_usage("After Whisper Load") # we need to control the VRAM usage all the time for (laptop's) embedded systems
        except Exception as e:
            logging.error(f"- Failed to load Whisper model: {e}")
            raise
    
    def unload_model(self):
        """Unload the Whisper model and free VRAM (CUDA only)"""
        if self.pipe is None:
            return
        
        logging.info("*  Unloading Whisper model from memory")
        log_cuda_memory_usage("Before Whisper Unload")
        
        # Delete the pipeline
        del self.pipe
        self.pipe = None
        
        # For CUDA: aggressive memory cleanup
        if CUDA_AVAILABLE:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logging.info("+ CUDA cache cleared after Whisper unload")
        
        log_cuda_memory_usage("After Whisper Unload")

    def transcribe(self, audio_file: str) -> str:
        # Load model if not already loaded
        if self.pipe is None:
            self._load_model()
        
        logging.info(f"* Transcribing with Hugging Face Whisper ({self.device_name}): {self.model_name}")
        log_cuda_memory_usage("Before Transcription")
        
        try:
            # For CUDA: use return_timestamps for better long-form handling
            if CUDA_AVAILABLE:
                # Build generate_kwargs - only include supported parameters
                generate_kwargs = {
                    "language": "english",  # Can be made configurable
                    "task": "transcribe",
                }
                
                # Add quality parameters (some models don't support all)
                # These are generally supported by most Whisper models
                try:
                    result = self.pipe(
                        audio_file,
                        return_timestamps=True,  # Better for long audio
                        generate_kwargs=generate_kwargs
                    )
                except Exception as e:
                    # If that fails, try without return_timestamps
                    logging.warning(f"!  Timestamps not supported, retrying without: {e}")
                    result = self.pipe(audio_file, generate_kwargs=generate_kwargs)
                # Extract text from chunks if timestamps are returned
                if isinstance(result, dict) and "chunks" in result:
                    text = " ".join([chunk["text"] for chunk in result["chunks"]])
                elif isinstance(result, dict) and "text" in result:
                    text = result["text"]
                else:
                    text = str(result) # just in case - should not happen, but..
            else:
                # just for CPU
                result = self.pipe(audio_file)
                text = result["text"]
            
            logging.info(f"+ Transcription completed on {self.device_name}")
            log_cuda_memory_usage("After Transcription") # to clarify: no messages generated without CUDA in this function
            return text
        except Exception as e:
            logging.error(f"- Transcription failed: {e}")
            raise

class CloudWhisperAdapter(TranscriptionService):
    def __init__(self, api_key: str, model_name: str = "whisper-1"):
        self.api_key = api_key
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=self.api_key)

    def transcribe(self, audio_file: str) -> str:
        logging.debug("Transcribing with Cloud Whisper: %s", self.model_name)
        with open(audio_file, "rb") as audio:
            transcript = self.client.audio.transcriptions.create(
                model=self.model_name,
                file=audio
            )
        return transcript.text

class LocalOpenAIAdapter(SummarizationService):
    def __init__(self, base_url: str, api_key: str, model_name: str):
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

    def summarize(self, prompt: str) -> str:
        logging.debug("Summarizing with Local OpenAI: %s", self.model_name)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content or "No summary generated."

class HuggingFaceLLMAdapter(SummarizationService):
    def __init__(self, model_name: str, lazy_load: bool = False):
        if not HF_AVAILABLE:
            raise ImportError("Hugging Face transformers not available.")
        
        self.model_name = model_name
        self.lazy_load = lazy_load
        self.pipe = None
        self.model = None
        self.tokenizer = None
        
        # Always use 4-bit quantization on CUDA
        self.use_4bit = CUDA_AVAILABLE and BITSANDBYTES_AVAILABLE
        
        # Determine device and configuration
        if CUDA_AVAILABLE:
            self.device = 0
            self.device_name = torch.cuda.get_device_name(0)
            self.torch_dtype = torch.float16
            
            # Estimate model size
            model_size_gb = self._estimate_model_size(model_name)
            
            logging.info(f"* LLM adapter configured for CUDA on {self.device_name}")
            if self.use_4bit:
                logging.info(f"   Mode: 4-bit NF4 quantization (for 8GB VRAM)")
                logging.info(f"   * Estimated VRAM: ~{model_size_gb:.1f} GB (model) + ~1.5 GB (context/overhead)")
                logging.info(f"   * Total estimated: ~{model_size_gb + 1.5:.1f} GB / 8.0 GB ({((model_size_gb + 1.5)/8.0)*100:.0f}%)")
            else:
                logging.warning(f"   *  Mode: FP16 (BitsAndBytes not available - will use more VRAM!)")
                logging.warning(f"!  Install with: pip install bitsandbytes") # another warning for the user to make it working faster
        else:
            # CPU mode
            self.device = -1
            self.torch_dtype = torch.float32
            self.device_name = "CPU"
            logging.info("* LLM adapter configured for CPU (this may be slower)")
        
        if not lazy_load:
            self._load_model()
    
    def _clean_summary_artifacts(self, text: str) -> str:
        """
        Remove structural artifacts and formatting noise from model output.
        This ensures clean, consistent summaries without meta-commentary in the final output (!).
        """
        import re
        
        # Remove common structural labels and headers
        artifacts = [
            r'\*\*Previous Summary\*\*:?\s*',
            r'\*\*New Content Added\*\*:?\s*',
            r'\*\*Extended Summary\*\*:?\s*',
            r'\*\*Updated Summary\*\*:?\s*',
            r'\*\*Instructions\*\*:?\s*',
            r'\*\*TASK\*\*:?\s*',
            r'\*\*YOUR PREVIOUS SUMMARY\*\*:?\s*',
            r'\*\*NEW LECTURE CONTENT TO ADD\*\*:?\s*',
            r'> \*\*Key Points:\*\*\s*',
            r'> \*\*Takeaways:\*\*\s*',
            r'^---+\s*$',  # Horizontal rules
            r'^\*\*Output\*\*\s*$',
            r'^Output ONLY the extended summary.*?:\s*$',
        ]
        
        cleaned = text
        for pattern in artifacts:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove meta-commentary phrases
        meta_phrases = [
            r'Here is the (?:updated|extended|complete) summary:?\s*',
            r'Below is the (?:updated|extended|complete) summary:?\s*',
            r'This (?:updated|extended|complete) summary (?:includes|incorporates|integrates).*?\.\s*',
            r'I have (?:extended|updated|incorporated).*?\.\s*',
            r'The summary has been (?:extended|updated).*?\.\s*',
        ]
        
        for pattern in meta_phrases:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
        
        # Clean up excessive whitespace to spare the memory
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)  # Max 2 consecutive newlines
        cleaned = re.sub(r'^\s+', '', cleaned, flags=re.MULTILINE)  # Remove leading spaces per line
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _estimate_model_size(self, model_name: str) -> float:
        """Estimate model VRAM usage in GB based on model name (4-bit quantized) 
           It's needed for choosing the right model for limited VRAM systems"""

        # Extract parameter size from model name
        if "7b" in model_name.lower() or "7-b" in model_name.lower():
            # 7B model in 4-bit: ~3.5 GB base + ~2GB context = 5.5GB (TOO BIG for 8GB)
            return 5.5
        elif "phi-3" in model_name.lower() or "3.8b" in model_name.lower():
            # Phi-3-mini (3.8B) in 4-bit: ~2.0 GB base + context
            return 2.0
        elif "3b" in model_name.lower() or "3-b" in model_name.lower():
            # 3B model in 4-bit: ~1.5 GB
            return 1.5
        elif "1.5b" in model_name.lower():
            # 1.5B model in 4-bit: ~0.8 GB
            return 0.8
        elif "0.5b" in model_name.lower():
            # 0.5B model in 4-bit: ~0.3 GB
            return 0.3
        else:
            # Unknown, assume moderate size
            return 2.5
    
    def _load_model(self):
        """Load the LLM model into memory"""
        if self.pipe is not None:
            logging.info("*  LLM model already loaded, skipping")
            return
        
        logging.info(f"* Loading LLM model: {self.model_name}")
        log_cuda_memory_usage("Before LLM Load")
        
        try:
            if CUDA_AVAILABLE and self.use_4bit:
                # 4-bit quantization configuration for limited VRAM
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,  # Double quantization saves ~0.4GB
                    bnb_4bit_quant_type="nf4"  # NF4 is more memory efficient than FP4
                )
                
                # Check if Flash Attention 2 is available (requires transformers >= 4.36)
                use_flash_attn = False
                try:
                    import transformers
                    from packaging import version
                    if version.parse(transformers.__version__) >= version.parse("4.36.0"):
                        use_flash_attn = True
                except:
                    pass
                
                logging.info(f"   * MEMORY OPTIMIZATIONS ENABLED:")
                logging.info(f"      • 4-bit NF4 quantization (4x compression)")
                logging.info(f"      • Double quantization (extra ~0.4GB savings)")
                logging.info(f"      • Flash Attention 2: {'Available' if use_flash_attn else 'Not available (transformers < 4.36)'}")
                logging.info(f"      • Low CPU memory usage")
                logging.info(f"      • GPU-only device placement (no CPU offload)")
                
                # Force load model and tokenizer (separately) entirely onto GPU 0 (avoid CPU offload/shared memory)
                # IMPORTANT insights: device_map needs string "cuda:0" not integer for quantized models!
                model_kwargs = {
                    "quantization_config": quantization_config,
                    "device_map": "cuda:0",  # Force GPU-only
                    "trust_remote_code": True,
                    "torch_dtype": torch.float16,  # Will be overridden by quantization_config
                    "low_cpu_mem_usage": True,  # Minimize CPU RAM during loading
                }
                
                # Try loading with Flash Attention 2 if available, fall back if not installed
                # Flash Attention 2 can speed up attention computations significantly,
                #   but it's hard to be installed in some systems - so we try/catch it here (not successfully).
                flash_attn_loaded = False # it's turned off temporarily
                if use_flash_attn:
                    try:
                        model_kwargs["attn_implementation"] = "flash_attention_2"
                        logging.info(f"      • Attempting to load with Flash Attention 2")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            **model_kwargs
                        )
                        flash_attn_loaded = True
                        logging.info(f"      + Flash Attention 2 enabled")
                    except Exception as e:
                        if "flash_attn" in str(e).lower():
                            logging.info(f"      ⚠️  Flash Attention 2 not available (package not installed), using standard attention")
                            # Remove flash attention and try again
                            model_kwargs.pop("attn_implementation", None)
                        else:
                            raise  # Re-raise if it's a different error
                
                # Load without Flash Attention if it failed or wasn't available
                if not flash_attn_loaded:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        **model_kwargs
                    )
                
                # VERIFY 4-bit quantization was actually applied
                logging.info("=" * 30)
                logging.info("* VERIFYING 4-BIT QUANTIZATION:") # some models may fail to load in 4-bit for various reasons (!)
                logging.info("=" * 30)
                
                # Check model dtype
                first_param = next(self.model.parameters())
                actual_dtype = first_param.dtype
                logging.info(f"   Model parameter dtype: {actual_dtype}")
                
                # Check if quantization config is present
                if hasattr(self.model, 'config') and hasattr(self.model.config, 'quantization_config'):
                    quant_config = self.model.config.quantization_config
                    logging.info(f"   + Quantization config present: {quant_config}")
                    if hasattr(quant_config, 'load_in_4bit'):
                        logging.info(f"   + load_in_4bit = {quant_config.load_in_4bit}")
                    if hasattr(quant_config, 'bnb_4bit_quant_type'):
                        logging.info(f"   + Quantization type = {quant_config.bnb_4bit_quant_type}")
                else:
                    logging.error(f"   - NO QUANTIZATION CONFIG FOUND - Model loaded in {actual_dtype}!")
                    logging.error(f"   - This means 4-bit quantization FAILED!")
                
                # Approximate the actual model size in memory
                model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
                model_size_gb = model_size_mb / 1024
                logging.info(f"   Model size in memory: {model_size_gb:.2f} GB")
                
                if model_size_gb > 3.0:
                    logging.error(f"   - Model is {model_size_gb:.2f}GB - TOO LARGE! 4-bit should be <2GB for 1.5B model")
                    logging.error(f"   - 4-bit quantization likely FAILED - model is in FP16!")
                else:
                    logging.info(f"   + Model size looks correct for 4-bit quantization")
                
                logging.info("=" * 30)
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    trust_remote_code=True
                )
                
                # Set padding token if not exists
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Get model's max context length
                max_length = getattr(self.model.config, 'max_position_embeddings', 
                                   getattr(self.model.config, 'max_sequence_length', 4096))
                logging.info(f"   Model context length: {max_length} tokens")
                
                # C H E C K (!): if model is on GPU
                model_device = next(self.model.parameters()).device
                logging.info(f"   Model loaded on device: {model_device}")
                if "cpu" in str(model_device):
                    logging.error("   - MODEL IS ON CPU! This should not happen!")
                else:
                    logging.info(f"   + Model confirmed on GPU: {model_device}")
                
                # Create pipeline WITHOUT device parameter (model already placed via device_map)
                # When using accelerate/device_map, you CANNOT specify device in pipeline!
                self.pipe = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    # NO device parameter - accelerate handles device placement!
                )
            else:
                # Standard fp16 for CUDA or fp32 for CPU
                if not BITSANDBYTES_AVAILABLE and self.use_4bit:
                    logging.warning("!  BitsAndBytes not available, loading model in FP16 instead (big VRAM usage)")
                
                self.pipe = pipeline(
                    "text-generation",
                    model=self.model_name,
                    device=self.device,
                    torch_dtype=self.torch_dtype,
                )
            
            logging.info(f"+ LLM model loaded successfully on {self.device_name}")
            log_cuda_memory_usage("After LLM Load")
        except Exception as e:
            logging.error(f"- Failed to load LLM model: {e}")
            raise
    
    def unload_model(self):
        """Unload the LLM model and free VRAM (CUDA only)"""
        if self.pipe is None:
            return
        
        logging.info("*  Unloading LLM model from memory")
        log_cuda_memory_usage("Before LLM Unload")
        
        # Delete all references
        del self.pipe
        self.pipe = None
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # For CUDA: aggressive memory cleanup
        if CUDA_AVAILABLE:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logging.info("+ CUDA cache cleared after LLM unload")
        
        log_cuda_memory_usage("After LLM Unload")

    def summarize(self, prompt: str) -> str:
        # Load model if not already loaded
        if self.pipe is None:
            self._load_model()
        
        logging.info(f"* Summarizing with Hugging Face pipeline ({self.device_name}): {self.model_name}")
        logging.info(f"   Prompt length: {len(prompt)} characters")
        logging.info(f"   Starting generation...")
        
        logging.info("=" * 30)
        logging.info("* GPU MEMORY CHECK 1: After Model Load")
        logging.info("=" * 30)

        mem_info = get_cuda_memory_info()
        if "error" not in mem_info:
            allocated_gb = mem_info['allocated_gb']
            total_gb = mem_info['total_gb']
            used_percent = (allocated_gb / total_gb) * 100
            logging.info(f"   Allocated: {allocated_gb:.2f} GB ({used_percent:.1f}%)")
            logging.info(f"   Free:      {mem_info['free_gb']:.2f} GB / {total_gb:.2f} GB")
            
            if used_percent > 85: # model is only one part, we need to put into the VRAM also KVCACHE (could be bigger) and other stuff
                logging.warning(f"   ! WARNING: Model alone uses {used_percent:.1f}% of VRAM!")
            else:
                logging.info(f"   + Model loaded, using {used_percent:.1f}% of VRAM")
        logging.info("=" * 30)
        
        # Check prompt length and tokenize
        if self.pipe.tokenizer:
            prompt_tokens = len(self.pipe.tokenizer.encode(prompt))
            logging.info(f"* Input tokens: {prompt_tokens:,}")
            
            # Get model's max context
            max_length = getattr(self.model.config if self.model else self.pipe.model.config, 
                               'max_position_embeddings', 4096)
            logging.info(f"* Model max context: {max_length:,} tokens")
            logging.info(f"* Context usage: {prompt_tokens:,} / {max_length:,} ({(prompt_tokens/max_length)*100:.1f}%)")
            
            # BALANCED: Auto-truncate to prevent KV cache OOM while maximizing GPU utilization
            # KV cache memory = input_tokens × 2 (K+V) × num_layers × hidden_dim × 2 bytes (fp16)
            # For 8GB VRAM with 1.08GB model: max ~6500 input tokens (~4GB KV cache + 1GB model + 1.5GB overhead)
            max_input_tokens = 6500  # BALANCED for 8GB VRAM (75-80% GPU utilization, safe headroom)
            
            if prompt_tokens > max_input_tokens:
                logging.warning(f"!  Prompt ({prompt_tokens:,} tokens) exceeds VRAM safe limit ({max_input_tokens:,})")
                logging.warning(f"   * AUTO-TRUNCATING to prevent KV cache OOM...")
                logging.warning(f"   * KV cache would use ~{(prompt_tokens/1000)*1.0:.1f}GB - TOO MUCH for 8GB GPU!")
                
                # Tokenize and truncate
                tokens = self.pipe.tokenizer.encode(prompt, truncation=True, max_length=max_input_tokens)
                prompt = self.pipe.tokenizer.decode(tokens, skip_special_tokens=True)
                prompt_tokens = len(tokens)
                
                logging.warning(f"   *  Truncated to {prompt_tokens:,} tokens (KV cache: ~{(prompt_tokens/1000)*1.0:.1f}GB)")
                logging.warning(f"   * TIP: Use chunked summarization for long transcripts!")
            elif prompt_tokens > 2000:
                logging.warning(f"*  Large prompt ({prompt_tokens:,} tokens) - KV cache will use ~{(prompt_tokens/1000)*1.0:.1f}GB")
                logging.warning(f"   Consider using chunked summarization for better memory efficiency")
        
        # === MEMORY CHECK 2: After Tokenization (before generation) ===
        logging.info("=" * 30)
        logging.info("* GPU MEMORY CHECK 2: After Tokenization (Before Generation)")
        logging.info("=" * 30)
        mem_info_before_gen = get_cuda_memory_info()
        if "error" not in mem_info_before_gen:
            allocated_gb = mem_info_before_gen['allocated_gb']
            total_gb = mem_info_before_gen['total_gb']
            used_percent = (allocated_gb / total_gb) * 100
            logging.info(f"   Allocated: {allocated_gb:.2f} GB ({used_percent:.1f}%)")
            logging.info(f"   Free:      {mem_info_before_gen['free_gb']:.2f} GB / {total_gb:.2f} GB")
            
            if used_percent > 90:
                logging.error(f"   ! CRITICAL: {used_percent:.1f}% VRAM used - likely to OOM!")
                logging.error(f"   The context ({prompt_tokens:,} tokens) may be too large for this GPU")
            elif used_percent > 80:
                logging.warning(f"   !  WARNING: {used_percent:.1f}% VRAM used - generation may be slow")
            else:
                logging.info(f"   + Ready for generation, {used_percent:.1f}% VRAM used")
        logging.info("=" * 30)
        
        try:
            logging.info("* Starting text generation with MEMORY-OPTIMIZED settings...")
            
            # AGGRESSIVE MEMORY OPTIMIZATIONS for generation
            # Phi-3 has DynamicCache.seen_tokens compatibility issue with transformers < 4.45
            # Disable cache for Phi-3 to avoid errors
            use_cache_setting = False if "phi" in self.model_name.lower() else True
            if not use_cache_setting:
                logging.info("   !  Disabling cache for Phi-3 compatibility (transformers version)")
            
            generate_kwargs = {
                "max_new_tokens": 1536,        # BALANCED for quality without OOM risk
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "truncation": True,
                "pad_token_id": self.pipe.tokenizer.eos_token_id,
                # OPTIMIZED FOR PERFORMANCE:
                "use_cache": True,             # ENABLED for faster generation (caches KV states)
                "num_beams": 1,                # Greedy decoding (beam search uses much more memory)
                "batch_size": 1,               # Single batch to minimize memory
            }
            
            # PyTorch memory optimizations
            if CUDA_AVAILABLE:
                # Enable gradient checkpointing if model supports it (trades compute for memory)
                if hasattr(self.model, 'gradient_checkpointing_enable'):
                    try:
                        self.model.gradient_checkpointing_enable()
                        logging.info("   + Gradient checkpointing enabled (saves VRAM)")
                    except:
                        pass  # Some models don't support this
                
                # Set memory efficient attention if available (use new API for PyTorch 2.9+)
                try:
                    torch.backends.cudnn.conv.fp32_precision = 'tf32'
                    torch.backends.cuda.matmul.fp32_precision = 'tf32'
                except AttributeError:
                    # Fall back to old API for older PyTorch versions
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
            
            # Generate with parameters optimized for summarization (Tokenize manually and move to GPU to ensure GPU execution)
            inputs = self.pipe.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
            
            # Move inputs to GPU explicitly
            if CUDA_AVAILABLE:
                inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
                logging.info(f"   + Inputs moved to GPU: {inputs['input_ids'].device}")
            
            # GPU EXECUTION CHECK
            logging.info("=" * 30)
            logging.info("* STARTING GPU GENERATION - WATCH GPU MEMORY NOW!")
            logging.info("=" * 30)
            
            # Get VRAM - important user info before the CPU performance could be frozen by the CPU/RAM usage
            mem_before = get_cuda_memory_info()
            vram_before = mem_before.get('allocated_gb', 0) if mem_before else 0
            logging.info(f"   VRAM before generation: {vram_before:.2f} GB")
            logging.info(f"   Expected VRAM during generation: ~{vram_before + (prompt_tokens/1000)*1.0:.2f} GB")
            logging.info(f"   ?  If GPU usage doesn't spike, generation is on CPU! CHECK IT OUT in your Task Manager!")
            logging.info("=" * 30)
            
            import time
            time.sleep(0.5)
            
            # Generate using model directly (bypass pipeline to ensure GPU usage)
            generation_start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=generate_kwargs["max_new_tokens"],
                    do_sample=generate_kwargs["do_sample"],
                    temperature=generate_kwargs["temperature"],
                    top_p=generate_kwargs["top_p"],
                    repetition_penalty=generate_kwargs["repetition_penalty"],
                    pad_token_id=generate_kwargs["pad_token_id"],
                    use_cache=generate_kwargs["use_cache"],
                    num_beams=generate_kwargs["num_beams"],
                )
            generation_time = time.time() - generation_start_time
            
            # GPU execution verification
            mem_after = get_cuda_memory_info()
            vram_after = mem_after.get('allocated_gb', 0) if mem_after else 0
            vram_increase = vram_after - vram_before
            tokens_per_sec = generate_kwargs['max_new_tokens'] / generation_time
            
            logging.info("=" * 30)
            logging.info("* GPU EXECUTION VERIFICATION")
            logging.info("=" * 30)
            logging.info(f"   VRAM before: {vram_before:.2f} GB")
            logging.info(f"   VRAM after:  {vram_after:.2f} GB")
            logging.info(f"   VRAM increase: {vram_increase:.2f} GB")
            logging.info(f"   Generation time: {generation_time:.1f}s")
            logging.info(f"   Tokens per second: {tokens_per_sec:.1f}")
            
            # Check if generation actually used GPU (use speed as primary indicator)
            # GPU: >15 tokens/s for 1.5B model, CPU: <5 tokens/s
            # Invoke error if generation is too slow (indicating CPU execution)
            if tokens_per_sec < 10:
                # Too slow - definitely CPU!
                logging.error("=" * 30)
                logging.error("! GPU EXECUTION FAILED!")
                logging.error("=" * 30)
                logging.error(f"   Generation speed: {tokens_per_sec:.1f} tokens/s")
                logging.error(f"   GPU should do >15 tokens/s, CPU does <5 tokens/s")
                logging.error("   ")
                logging.error("   * GENERATION IS RUNNING ON CPU, NOT GPU!")
                logging.error("   Stopping to avoid wasting time...")
                logging.error("=" * 30)
                raise RuntimeError("GPU execution verification failed - model is running on CPU despite being loaded on GPU. This is a critical bug.")
            else:
                logging.info(f"   + GPU EXECUTION CONFIRMED!")
                logging.info(f"   Speed: {tokens_per_sec:.1f} tokens/s (GPU threshold: >15)")
                if vram_increase > 0.1:
                    logging.info(f"   KV cache allocated: ~{vram_increase:.2f} GB")
                else:
                    logging.info(f"   Note: KV cache not visible in VRAM stats (4-bit quantization)")
                logging.info("=" * 30)
            
            # Decode output
            generated_text = self.pipe.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # extract the output text
            input_length = inputs['input_ids'].shape[1]
            generated_tokens_only = outputs[0][input_length:]  # Get only new tokens after input
            extracted_summary = self.pipe.tokenizer.decode(generated_tokens_only, skip_special_tokens=True)
            
            # Remove structural labels and formatting
            cleaned_summary = self._clean_summary_artifacts(extracted_summary)
            
            logging.info("=" * 30)
            logging.info("*  RESPONSE EXTRACTION & CLEANING")
            logging.info("=" * 30)
            logging.info(f"   Total output tokens: {len(outputs[0])}")
            logging.info(f"   Input tokens (prompt): {input_length}")
            logging.info(f"   Generated tokens (new): {len(generated_tokens_only)}")
            logging.info(f"   Raw extracted length: {len(extracted_summary)} chars")
            logging.info(f"   Cleaned summary length: {len(cleaned_summary)} chars")
            logging.info(f"   Preview: {cleaned_summary[:150].replace(chr(10), ' ')}...")
            logging.info("=" * 30)
            
            output = [{"generated_text": cleaned_summary}]
            
            # GPU memory check after the output
            logging.info("=" * 30)
            logging.info("* GPU MEMORY CHECK 3: After Generation")
            logging.info("=" * 30)
            mem_info_after = get_cuda_memory_info()
            if "error" not in mem_info_after:
                allocated_gb = mem_info_after['allocated_gb']
                total_gb = mem_info_after['total_gb']
                used_percent = (allocated_gb / total_gb) * 100
                logging.info(f"   Allocated: {allocated_gb:.2f} GB ({used_percent:.1f}%)")
                logging.info(f"   Peak usage: {used_percent:.1f}%")
            logging.info("=" * 30)
            
            logging.info(f"+ Summary generated successfully on {self.device_name}")
            return output[0]["generated_text"]
        except Exception as e:
            logging.error(f"! Summary generation failed: {e}")
            if "out of memory" in str(e).lower():
                logging.error("=" * 30)
                logging.error("! OUT OF MEMORY ERROR")
                logging.error("=" * 30)
                logging.error(f"   Model VRAM: ~{mem_info.get('allocated_gb', 0):.2f} GB")
                logging.error(f"   Input tokens: {prompt_tokens:,}")
                logging.error("   ")
                logging.error("   Solutions:")
                logging.error("   1. Use smaller model (Qwen 2.5-1.5B instead of Phi-3)")
                logging.error("   2. Reduce transcription length (split audio)")
                logging.error("   3. Use cloud API (no VRAM needed)")
                logging.error("=" * 30)
            raise

class GROQAdapter(SummarizationService):
    def __init__(self, api_key: str, model_name: str):
        self.client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key
        )
        self.model_name = model_name

    def summarize(self, prompt: str) -> str:
        logging.debug("Summarizing with GROQ: %s", self.model_name)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content or "No summary generated."

class LocalLLMAdapter(SummarizationService):
    def __init__(self, base_url: str, api_key: str, model_name: str):
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

    def summarize(self, prompt: str) -> str:
        logging.debug("Summarizing with Local LLM: %s", self.model_name)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content or "No summary generated."

# -------------------------
# Diagnostic Functions
# -------------------------
def get_device_diagnostics() -> str:
    """Get comprehensive device diagnostics for troubleshooting."""
    diagnostics = []
    diagnostics.append("="*30)
    diagnostics.append("DEVICE DIAGNOSTICS")
    diagnostics.append("="*30)
    
    # System info
    import platform
    diagnostics.append(f"OS: {platform.system()} {platform.release()}")
    diagnostics.append(f"Python: {platform.python_version()}")
    
    # Hardware availability
    diagnostics.append(f"\nHardware Configuration:")
    diagnostics.append(f"  Device Type: {DEVICE_TYPE}")
    diagnostics.append(f"  MLX Available: {MLX_AVAILABLE}")
    diagnostics.append(f"  CUDA Available: {CUDA_AVAILABLE}")
    diagnostics.append(f"  CPU Only: {CPU_ONLY}")
    diagnostics.append(f"  HF Transformers: {HF_AVAILABLE}")
    
    # CUDA details
    if CUDA_AVAILABLE:
        diagnostics.append(f"\nCUDA Configuration:")
        diagnostics.append(f"  PyTorch Version: {torch.__version__}")
        diagnostics.append(f"  CUDA Version: {torch.version.cuda}")
        diagnostics.append(f"  Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            diagnostics.append(f"  GPU {i}: {props.name}")
            diagnostics.append(f"    Memory: {props.total_memory / (1024**3):.2f} GB")
            diagnostics.append(f"    Compute Capability: {props.major}.{props.minor}")
    
    # MLX details
    if MLX_AVAILABLE:
        diagnostics.append(f"\nMLX Configuration:")
        diagnostics.append(f"  MLX Whisper Available")
    
    diagnostics.append("="*30)
    return "\n".join(diagnostics)

# Logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logging.info("Initializing Audio Transcription & Summarization application")
logging.info(get_device_diagnostics())

# OpenAI client (local)
client = openai.OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="dummy"
)
# Cache directory
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def sha256_of_file(file_path: str, block_size: int = 65536) -> str:
    """Utility Functions: Compute SHA256 hash of a file."""
    logging.debug("Computing SHA256 for file: %s", file_path)
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()

def get_cached_transcription(audio_file: str, force_refresh=False) -> str | None:
    """Return cached transcription if available and not forcing refresh."""
    logging.debug("Checking cache for file: %s", audio_file)
    digest = sha256_of_file(audio_file)
    cache_path = CACHE_DIR / f"{digest}.txt"
    logging.debug("Cache path: %s", cache_path)
    if cache_path.exists() and not force_refresh:
        logging.debug("Cache exists, attempting to read")
        try:
            content = cache_path.read_text(encoding="utf-8")
            logging.info(f"Loaded transcription {audio_file} from cache {digest}")
            return content
        except Exception as e:
            logging.warning(f"Unable to read cache: {e}")
    logging.debug("No valid cache found")
    return None

class WhisperModelChoice(StrEnum):
    """Enums"""
    # MLX models (Apple Silicon)
    WHISPER_MLX_TINY = "mlx-community/whisper-tiny"
    WHISPER_MLX_SMALL = "mlx-community/whisper-small"
    WHISPER_MLX_BASE = "mlx-community/whisper-base"
    WHISPER_MLX_LARGE = "mlx-community/whisper-large-v3-mlx"
    # Hugging Face models (CUDA/CPU) - Standard Whisper
    WHISPER_HF_TINY = "openai/whisper-tiny"
    WHISPER_HF_BASE = "openai/whisper-base"
    WHISPER_HF_SMALL = "openai/whisper-small"
    WHISPER_HF_MEDIUM = "openai/whisper-medium"
    WHISPER_HF_LARGE_V3 = "openai/whisper-large-v3"
    # * RECOMMENDED: Better quality models for lectures
    WHISPER_HF_LARGE_V3_TURBO = "openai/whisper-large-v3-turbo"  # * 8x faster, ~6GB, excellent quality
    DISTIL_WHISPER_LARGE_V3 = "distil-whisper/distil-large-v3"    # * 6x faster, ~3GB, 99% quality
    DISTIL_WHISPER_LARGE_V2 = "distil-whisper/distil-large-v2"    # Alternative, slightly faster
    # Cloud API
    WHISPER_CLOUD = "cloud/whisper-1"

class HuggingFaceLLMModelDefaultChoices(StrEnum):
    """Enums"""
    GRANITE_4_H_TINY = "ibm/granite-4-h-tiny"
    QWEN3_4B_2507 = "qwen/qwen3-4b-2507"
    GEMMA_3N_E4B = "google/gemma-3n-e4b"
    BAGUETTOTRON = 'PleIAs/Baguettotron'
    # 4-bit quantized models for CUDA (8GB VRAM, sequential loading)
    # Context Length Guide:
    # - Short transcripts (<15k tokens): Qwen 2.5-1.5B (32k context) ← Fast & safe
    # - Medium transcripts (15-30k tokens): Qwen 2.5-3B (32k context) ← Better quality
    # - Long transcripts (30-100k tokens): Qwen 2.5-7B-128K (128k context) ← LONG CONTEXT!
    # - Very long (100k+ tokens): Use cloud API
    QWEN2_5_1_5B_INSTRUCT = "qwen2.5-1.5b-instruct"         # 1.5B, 32k context, ~3-4GB, fast
    QWEN2_5_3B_INSTRUCT = "qwen2.5-3b-instruct-8k"          # 3B, 32k context, ~4-5GB, good quality
    QWEN2_5_7B_INSTRUCT_128K = "qwen2.5-7b-instruct-128k"   # 7B, 128k context!, ~6-7GB, LONG CONTEXT
    PHI3_MINI_128K = "phi-3-mini-128k"                       # 3.8B, 128k, ~6-7GB (alternative)
    MISTRAL_7B_INSTRUCT = "mistral-7b-instruct-8k"          # 7B, 32k, ~7-8GB (tight)

class BackendChoice(StrEnum):
    """Enums"""
    LOCAL_OPENAI = "Local OpenAI"
    HUGGING_FACE = "Hugging Face"
    GROQ = "GROQ"
    LOCAL_LLM = "Local LLM"

class PromptChoice(StrEnum):
    """Enums"""
    SUMMARY = "Summary"
    SUMMARY_BULLETS = "Summary with bullet points"
    FLASHCARDS = "Flashcards"

PROMPT_MAPPING: dict[str, str] = {
    PromptChoice.SUMMARY: """
You are an expert teaching assistant.

Here is a class transcription:
<START TRANSCRIPTION>
{transcription}
<END TRANSCRIPTION>

Step 1: Summarize the key points clearly and concisely.  
Step 2: Review your summary and improve it by making it more structured and easy to understand, without adding new information.  
Step 3: Output the refined summary. 

### Refined Summary

**1. Topic Overview**  
Concise statement of the main theme.

**2. Key Concepts Covered**  
- Bullet point summary of the main ideas and subtopics.

**3. Instructor’s Emphasis / Takeaways**  
- Highlight any repeated or stressed points.

**4. Contextual Notes (if relevant)**  
- Any dependencies, references, or frameworks mentioned.
""",
    PromptChoice.SUMMARY_BULLETS: "Provide a summary with bullet points for this info: {transcription}",
    PromptChoice.FLASHCARDS: """You are an expert teaching assistant. Here is a class transcription:

{transcription}

Step 1: Create flashcards with clear questions and answers based on this class.  
Step 2: Review each flashcard and improve the wording for clarity and accuracy, without adding new content.  
Step 3: Output the refined set of flashcards.
""",
    "Custom": "{transcription}"
}

def get_transcription_service(model: str, lazy_load: bool = False) -> TranscriptionService:
    """
    Transcription: Get transcription service adapter.
    
    Args:
        model: Model identifier
        lazy_load: If True (CUDA only), defer model loading until transcription time
    """
    if model.startswith("mlx-community/"):
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX models require Apple Silicon with mlx-whisper installed.")
        return MLXWhisperAdapter(model)
    elif model.startswith("openai/"):
        if not HF_AVAILABLE:
            raise RuntimeError("Hugging Face models require transformers and torch installed.")
        # Use lazy loading for CUDA to optimize memory
        return HuggingFaceWhisperAdapter(model, lazy_load=lazy_load and CUDA_AVAILABLE)
    elif model.startswith("cloud/"):
        # Assume API key is set in environment ("OPENAI_API_KEY") or config
        import os
        api_key = os.getenv("OPENAI_API_KEY", "")
        return CloudWhisperAdapter(api_key, model.split("/")[-1])
    else:
        raise ValueError(f"Unsupported transcription model: {model}")

def get_cuda_memory_info() -> dict:
    """Get CUDA memory information if available."""
    if not CUDA_AVAILABLE:
        return {"available": False}
    
    try:
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        free = total - reserved
        
        return {
            "available": True,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "free_gb": free,
            "device_name": torch.cuda.get_device_name(0)
        }
    except Exception as e:
        logging.error(f"Failed to get CUDA memory info: {e}")
        return {"available": True, "error": str(e)}

def get_recommended_whisper_model() -> str:
    """Get recommended Whisper model based on hardware."""
    if MLX_AVAILABLE:
        return WhisperModelChoice.WHISPER_MLX_BASE.value
    elif CUDA_AVAILABLE:
        # For 8GB VRAM: use large-v3-turbo (best quality, 8x faster than v3)
        return WhisperModelChoice.WHISPER_HF_LARGE_V3_TURBO.value
    elif HF_AVAILABLE:
        # CPU - recommend smaller models
        return WhisperModelChoice.WHISPER_HF_TINY.value
    else:
        return WhisperModelChoice.WHISPER_CLOUD.value

def get_available_whisper_models() -> list[str]:
    """Return list of available Whisper models based on hardware capabilities."""
    available_models = []
    
    if MLX_AVAILABLE:
        available_models.extend([
            WhisperModelChoice.WHISPER_MLX_TINY.value,
            WhisperModelChoice.WHISPER_MLX_SMALL.value,
            WhisperModelChoice.WHISPER_MLX_BASE.value,
            WhisperModelChoice.WHISPER_MLX_LARGE.value,
        ])
        logging.info("Added MLX Whisper models to available options")
    
    # Add Hugging Face models if available (works with both CUDA and CPU)
    if HF_AVAILABLE:
        available_models.extend([
            WhisperModelChoice.WHISPER_HF_TINY.value,
            WhisperModelChoice.WHISPER_HF_BASE.value,
            WhisperModelChoice.WHISPER_HF_SMALL.value,
            WhisperModelChoice.WHISPER_HF_MEDIUM.value,
            WhisperModelChoice.WHISPER_HF_LARGE_V3.value,
            # high-quality models - recommended for lecture recordings
            WhisperModelChoice.WHISPER_HF_LARGE_V3_TURBO.value,
            WhisperModelChoice.DISTIL_WHISPER_LARGE_V3.value,
            WhisperModelChoice.DISTIL_WHISPER_LARGE_V2.value,
        ])
        if CUDA_AVAILABLE:
            logging.info("Added Hugging Face Whisper models (CUDA accelerated)")
        else:
            logging.info("Added Hugging Face Whisper models (CPU mode)")
    
    # Cloud API is always available as fallback
    available_models.append(WhisperModelChoice.WHISPER_CLOUD.value)
    
    return available_models

def transcribe_file(audio_file: str, model: str, lazy_load: bool = False) -> str:
    """Transcribe audio using selected Whisper model."""
    logging.debug("Starting transcription with model: %s for file: %s", model, audio_file)
    start = time.time()
    try:
        service = get_transcription_service(model, lazy_load=lazy_load)
        text = service.transcribe(audio_file)
        logging.info(f"Transcription of file {audio_file} completed in {time.time() - start:.2f}s")
        return str(text)
    except Exception as e:
        logging.error("Error during transcription: %s", e)
        return f"Error during transcription: {e}"

def format_file_metadata(audio_file: str, digest: str | None = None) -> str:
    """Prompt/Metadata formatting for display"""
    logging.debug("Formatting metadata for file: %s", audio_file)
    p = Path(audio_file)
    if not p.exists():
        return ""
    size_kb = p.stat().st_size / 1024
    if digest is None:
        try:
            digest = sha256_of_file(audio_file)
        except FileNotFoundError:
            digest = "Unavailable"
    return (
        f"**File:** {p.name}  \n"
        f"**Size:** {size_kb:.1f} KB  \n"
        f"**SHA256:** `{digest}`"
    )

def fetch_models(base_url: str, api_key: str = "") -> list[str]:
    """Fetch available models from OpenAI-compatible server."""
    try:
        import requests
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response = requests.get(f"{base_url}/models", headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            return models
        else:
            logging.warning(f"Failed to fetch models: {response.status_code}")
            return []
    except Exception as e:
        logging.error(f"Error fetching models: {e}")
        return []

def get_default_hf_models() -> list[str]:
    """Get default Hugging Face models based on hardware"""
    if CUDA_AVAILABLE:
        # 4-bit quantized models for CUDA
        return [
            HuggingFaceLLMModelDefaultChoices.QWEN2_5_1_5B_INSTRUCT.value,     # BEST for 8GB: 1.5B, 32k, ~2-3GB
            HuggingFaceLLMModelDefaultChoices.PHI3_MINI_128K.value,            # LONG CONTEXT: 3.8B, 128k!, ~4-5GB (RECOMMENDED)
            HuggingFaceLLMModelDefaultChoices.QWEN2_5_3B_INSTRUCT.value,       # 3B, 32k, ~4-5GB
            # HuggingFaceLLMModelDefaultChoices.QWEN2_5_7B_INSTRUCT_128K.value,  # - TOO BIG: 7B uses `share`d memory (10GB+)
        ]
    else:
        # Standard models for CPU/MLX
        return [
            HuggingFaceLLMModelDefaultChoices.GRANITE_4_H_TINY.value,
            HuggingFaceLLMModelDefaultChoices.QWEN3_4B_2507.value,
            HuggingFaceLLMModelDefaultChoices.GEMMA_3N_E4B.value
        ]

def update_backend_settings(backend):
    if backend == BackendChoice.GROQ:
        return gr.update(visible=True, label="GROQ API Key"), gr.update(visible=False, value="https://api.groq.com/openai/v1"), gr.update(visible=False), gr.update(visible=True, choices=[])
    elif backend in [BackendChoice.LOCAL_OPENAI, BackendChoice.LOCAL_LLM]:
        default_url = "http://localhost:1234/v1" if backend == BackendChoice.LOCAL_OPENAI else "http://localhost:8000/v1"
        return gr.update(visible=False), gr.update(visible=True, value=default_url), gr.update(visible=True), gr.update(visible=True, choices=[])
    else:  # Hugging Face
        hf_models = get_default_hf_models()
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, choices=hf_models)

def fetch_and_update_models(base_url, api_key):
    models = fetch_models(base_url, api_key)
    if models:
        return gr.update(choices=models, value=models[0] if models else None)
    else:
        return gr.update(choices=[], value=None)

def update_prompt(prompt_type: str):
    logging.debug("Updating prompt for type: %s", prompt_type)
    return gr.update(
        value=PROMPT_MAPPING.get(prompt_type, PROMPT_MAPPING[PromptChoice.SUMMARY.value]),
        interactive=(prompt_type == "Custom")
    )

def get_summarization_service(backend: str, llm_model_choice: str, base_url: str = "", api_key: str = "", lazy_load: bool = False) -> SummarizationService:
    """
    Get summarization service adapter.
    
    Args:
        backend: Backend choice (Hugging Face, GROQ, etc.)
        llm_model_choice: Model identifier
        base_url: Base URL for API-based services
        api_key: API key for API-based services
        lazy_load: If True (CUDA only), defer model loading until summarisation time
    """
    if backend == BackendChoice.HUGGING_FACE:
        if not HF_AVAILABLE:
            raise ImportError("! Hugging Face not available. Please install transformers and torch.") # warning to install the needed libraries
        
        # Model mapping - includes both standard and 4-bit quantized models
        model_mapping = {
            # Standard models (for CPU/MLX)
            "ibm/granite-4-h-tiny": "ibm-granite/granite-4.0-h-tiny",
            "qwen/qwen3-4b-2507": "Qwen/Qwen3-4B",
            "google/gemma-3n-e4b": "google/gemma-3-4B",
            "PleIAs/Baguettotron": "PleIAs/Baguettotron",
            # 4-bit quantized models for CUDA (8GB VRAM, sequential loading)
            "qwen2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",           # 1.5B, 32k, ~3-4GB - the only which works well (for 8GB VRAM so far)
            "qwen2.5-3b-instruct-8k": "Qwen/Qwen2.5-3B-Instruct",            # 3B, 32k, ~4-5GB
            "qwen2.5-7b-instruct-128k": "Qwen/Qwen2.5-7B-Instruct",          # 7B, 128k!, ~6-7GB
            "phi-3-mini-128k": "microsoft/Phi-3-mini-128k-instruct",         # 3.8B, 128k, ~6-7GB
            "mistral-7b-instruct-8k": "mistralai/Mistral-7B-Instruct-v0.2",  # 7B, 32k, ~7-8GB
        }
        model_path = model_mapping.get(llm_model_choice)
        if not model_path:
            raise ValueError(f"Model {llm_model_choice} not supported for Hugging Face.")
        # Use lazy loading for CUDA to optimize memory
        return HuggingFaceLLMAdapter(model_path, lazy_load=lazy_load and CUDA_AVAILABLE)
    elif backend == BackendChoice.GROQ:
        if not api_key:
            import os
            api_key = os.getenv("GROQ_API_KEY", "")
        return GROQAdapter(api_key, llm_model_choice)
    elif backend == BackendChoice.LOCAL_LLM:
        if not base_url:
            base_url = "http://localhost:8000/v1"
        return LocalLLMAdapter(base_url, api_key or "dummy", llm_model_choice)
    else:  # Local OpenAI
        if not base_url:
            base_url = "http://localhost:1234/v1"
        return LocalOpenAIAdapter(base_url, api_key or "dummy", llm_model_choice)

def clean_transcription(text: str) -> str:
    """
    Clean transcription to reduce verbosity while preserving content.
    Removes excessive repetitions, filler words, and stuttering.
    """
    import re
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common filler words/sounds (but keep content)
    fillers = ['um', 'uh', 'er', 'ah', 'hmm', 'mm', 'mhm']
    for filler in fillers:
        # Only remove standalone fillers, not as part of words
        text = re.sub(rf'\b{filler}\b', '', text, flags=re.IGNORECASE)
    
    # Remove excessive repetitions (e.g., "the the the" -> "the")
    # But be careful not to remove intentional repetition
    words = text.split()
    cleaned_words = []
    prev_word = None
    repeat_count = 0
    
    for word in words:
        word_lower = word.lower()
        if word_lower == prev_word:
            repeat_count += 1
            if repeat_count < 2:  # Allow one repetition
                cleaned_words.append(word)
        else:
            cleaned_words.append(word)
            prev_word = word_lower
            repeat_count = 0
    
    text = ' '.join(cleaned_words)
    
    # Remove multiple punctuation (e.g., "..." -> ".")
    text = re.sub(r'([.!?,])\1+', r'\1', text)
    
    # Clean up spacing around punctuation
    text = re.sub(r'\s+([.!?,])', r'\1', text)
    text = re.sub(r'([.!?,])([A-Z])', r'\1 \2', text)
    
    return text.strip()

def safe_generate_summary(prompt: str, llm_model_choice: str, backend: str, base_url: str = "", api_key: str = "", lazy_load: bool = False) -> str:
    """Generate summary using selected backend with safe error handling."""
    logging.debug("Generating summary with backend: %s, model: %s", backend, llm_model_choice)
    try:
        service = get_summarization_service(backend, llm_model_choice, base_url, api_key, lazy_load=lazy_load)
        return service.summarize(prompt)
    except Exception as e:
        logging.error("Error generating summary: %s", e)
        return f"Error generating summary: {e}"

def clear_cuda_cache():
    """CUDA Memory Management: Clear CUDA cache if available."""
    if CUDA_AVAILABLE:
        try:
            torch.cuda.empty_cache()
            logging.info("* CUDA cache cleared")
        except Exception as e:
            logging.warning(f"Failed to clear CUDA cache: {e}")

def log_cuda_memory_usage(stage: str):
    """Log current CUDA memory usage."""
    if CUDA_AVAILABLE:
        mem_info = get_cuda_memory_info()
        if "error" not in mem_info:
            logging.info(
                f"* CUDA Memory [{stage}]: "
                f"Allocated: {mem_info['allocated_gb']:.2f}GB, "
                f"Reserved: {mem_info['reserved_gb']:.2f}GB, "
                f"Free: {mem_info['free_gb']:.2f}GB / {mem_info['total_gb']:.2f}GB"
            )

def chunk_text_by_tokens(text: str, max_tokens: int = 20000, overlap_tokens: int = 500) -> list[str]:
    """
    Chunked Summarization (MapReduce Pattern)
    Split text into overlapping chunks based on approximate token count.
    
    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk (default 20k for 32k models with margin - USE 32k models only)
        overlap_tokens: Overlap between chunks for context continuity
    
    Returns:
        List of text chunks
    """
    logging.info(f"      * Chunking: Splitting {len(text)} chars into {max_tokens}-token chunks...")
    
    # Rough estimate: 1 token ≈ 0.75 words - ASSUMPTION
    import time
    start_time = time.time()
    
    logging.info(f"      * Step 1/4: Splitting text into words...")
    words = text.split()
    logging.info(f"      * {len(words):,} words found ({time.time()-start_time:.2f}s)")
    
    words_per_chunk = int(max_tokens * 0.75)
    overlap_words = int(overlap_tokens * 0.75)
    
    if len(words) <= words_per_chunk:
        logging.info(f"      + Text fits in single chunk, no splitting needed")
        return [text]
    
    logging.info(f"      * Step 2/4: Calculating chunk boundaries...")
    logging.info(f"         Words per chunk: {words_per_chunk:,}")
    logging.info(f"         Overlap: {overlap_words:,} words")
    
    chunks = []
    start = 0
    chunk_count = 0
    
    logging.info(f"      * Step 3/4: Creating chunks...")
    while start < len(words):
        chunk_count += 1
        end = min(start + words_per_chunk, len(words))
        
        # Create the chunk
        chunk_words = words[start:end]
        chunk_text = ' '.join(chunk_words)
        chunks.append(chunk_text)
        
        # Log chunk details with preview
        first_words = ' '.join(chunk_words[:10])
        logging.info(f"         Chunk {chunk_count}: words [{start:,}:{end:,}] = {end-start:,} words")
        logging.info(f"         Starts with: \"{first_words}...\"")
        
        # If we've reached the end, stop
        if end >= len(words):
            logging.info(f"         + Reached end of text at word {end:,}")
            break
        
        # Move start with overlap (but ensure we always advance)
        next_start = end - overlap_words
        if next_start <= start:
            # Overlap too large, just move forward by at least 1 word
            next_start = start + max(1, words_per_chunk // 2)
            logging.warning(f"         !  Overlap too large, advancing by {next_start - start:,} words")
        
        advance_amount = next_start - start
        logging.info(f"         Next chunk will start at word {next_start:,} (advanced {advance_amount:,} words)")
        start = next_start
    
    elapsed = time.time() - start_time
    logging.info(f"      + Step 4/4: Chunking complete! Created {len(chunks)} chunks in {elapsed:.2f}s")
    
    return chunks

def summarize_in_chunks(llm_service, transcription: str, prompt_template: str, max_tokens_per_chunk: int = 5000) -> str:
    """
    Summarize long transcription using ITERATIVE REFINEMENT:
    1. Summarize first chunk
    2. For each subsequent chunk: combine previous summary + new chunk
    3. Iteratively build up the complete summary
    4. Extract only the final summary for output
    
    Args:
        llm_service: Summarization service
        transcription: Full transcription text
        prompt_template: Template with {transcription} placeholder (used for first chunk)
        max_tokens_per_chunk: Max tokens per chunk (default 5k for balanced 8GB GPU utilization)
    
    Returns:
        Final iteratively-refined summary
    """
    logging.info("="*30)
    logging.info("* CHUNKED SUMMARIZATION (Iterative Refinement)")
    logging.info("="*30)
    logging.info(f"   Transcription length: {len(transcription)} chars")
    logging.info(f"   Max tokens per chunk: {max_tokens_per_chunk}")
    
    # Clear the LLM interaction log for this new run
    clear_llm_log()
    
    # Split into chunks
    logging.info("   Splitting transcription into chunks...")
    chunks = chunk_text_by_tokens(transcription, max_tokens=max_tokens_per_chunk)
    num_chunks = len(chunks)
    logging.info(f"   + Chunking complete: {num_chunks} chunks created")
    
    logging.info(f"   Split into {num_chunks} chunks")
    for i, chunk in enumerate(chunks, 1):
        word_count = len(chunk.split())
        est_tokens = int(word_count * 1.3)
        # Show first 100 chars to verify chunks are different
        preview = chunk[:100].replace('\n', ' ')
        logging.info(f"   Chunk {i}: ~{word_count:,} words (~{est_tokens:,} tokens)")
        logging.info(f"      Preview: \"{preview}...\"")
    logging.info("="*30)
    
    if num_chunks == 1:
        # Single chunk - summarize directly using original template
        logging.info("+ Single chunk, using direct summarization")
        prompt = prompt_template.replace("{transcription}", transcription)
        return llm_service.summarize(prompt)
    
    # ITERATIVE REFINEMENT: Build summary progressively
    current_summary = None
    
    for i, chunk in enumerate(chunks, 1):
        # Calculate memory pressure for this chunk
        chunk_tokens = int(len(chunk.split()) * 1.3)
        estimated_kv_cache_gb = (chunk_tokens / 1000) * 1.0  # ~1GB per 1000 tokens
        
        logging.info("="*30)
        logging.info(f"* CHUNK {i}/{num_chunks} - Progress: {(i/num_chunks)*100:.0f}%")
        logging.info("="*30)
        logging.info(f"   Chunk size: ~{chunk_tokens:,} tokens")
        logging.info(f"   * Estimated memory pressure:")
        logging.info(f"      - Model: 1.08 GB (already loaded)")
        logging.info(f"      - KV cache: ~{estimated_kv_cache_gb:.1f} GB")
        logging.info(f"      - Total: ~{1.08 + estimated_kv_cache_gb:.1f} GB / 8.0 GB ({((1.08 + estimated_kv_cache_gb)/8.0)*100:.0f}%)")
        logging.info("="*30)
        
        if i == 1:
            # FIRST CHUNK: Generate initial summary using original instructions
            logging.info(f"   * Generating initial summary from first chunk...")
            first_prompt = prompt_template.replace("{transcription}", chunk)
            
            try:
                current_summary = llm_service.summarize(first_prompt)
                
                # LOG THE INTERACTION
                log_llm_interaction(i, num_chunks, first_prompt, current_summary)
                
                summary_preview = current_summary[:200].replace('\n', ' ')
                logging.info(f"   + Initial summary created ({len(current_summary)} chars): {summary_preview}...")
            except Exception as e:
                logging.error(f"   - Failed to summarize first chunk: {e}")
                return f"Error: Failed to create initial summary - {e}"
        
        else:
            # SUBSEQUENT CHUNKS: Combine previous summary + new chunk
            logging.info(f"   * Refining summary with new content...")
            summary_tokens = int(len(current_summary.split()) * 1.3)
            combined_tokens = chunk_tokens + summary_tokens
            combined_kv_cache_gb = (combined_tokens / 1000) * 1.0
            logging.info(f"   Combined input: ~{combined_tokens:,} tokens (summary + new chunk)")
            logging.info(f"   * KV cache for this iteration: ~{combined_kv_cache_gb:.1f} GB")
            
            refinement_prompt = f"""
You are an expert teaching assistant. You have a summary of a lecture so far, and now you need to extend it with new content from the next part of the lecture.

Previous summary:
{current_summary}

New lecture content:
{chunk}

Extend the summary by naturally adding the new information. Keep the same formatting style and structure. Do NOT add any labels like "Previous Summary" or "New Content Added" or "Extended Summary" - just provide the extended summary text directly, continuing from where the previous summary left off."""
            
            try:
                current_summary = llm_service.summarize(refinement_prompt)
                
                # LOG THE INTERACTION
                log_llm_interaction(i, num_chunks, refinement_prompt, current_summary)
                
                summary_preview = current_summary[:200].replace('\n', ' ')
                logging.info(f"   + Summary refined ({len(current_summary)} chars): {summary_preview}...")
            except Exception as e:
                logging.error(f"   ! Failed to refine with chunk {i}: {e}")
                logging.warning(f"   !  Continuing with previous summary...")
                # Continue with existing summary if refinement fails
    
    # FINAL STEP: Clean and extract the final summary
    logging.info("="*30)
    logging.info("* Final Summary Cleanup...")
    logging.info("="*30)
    
    # Apply comprehensive artifact cleaning
    # Note: We need to use a standalone cleaning function since we're outside the class
    import re
    
    final_summary = current_summary.strip()
    
    # Remove common structural labels and headers
    artifacts_patterns = [
        r'\*\*Previous Summary\*\*:?\s*',
        r'\*\*New Content Added\*\*:?\s*',
        r'\*\*Extended Summary\*\*:?\s*',
        r'\*\*Updated Summary\*\*:?\s*',
        r'\*\*Complete Summary\*\*:?\s*',
        r'\*\*Final Summary\*\*:?\s*',
        r'^---+\s*$',  # Horizontal rules
        r'^> ',  # Quote markers
    ]
    
    for pattern in artifacts_patterns:
        final_summary = re.sub(pattern, '', final_summary, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove simple text artifacts at start
    simple_artifacts = [
        "Updated Summary:",
        "Complete Summary:",
        "Final Summary:",
        "Extended Summary:",
        "Summary:",
        "Here is the updated summary:",
        "Here is the extended summary:",
        "Here is the complete summary:",
    ]
    
    for artifact in simple_artifacts:
        if final_summary.startswith(artifact):
            final_summary = final_summary[len(artifact):].strip()
    
    # Clean up excessive whitespace
    final_summary = re.sub(r'\n\s*\n\s*\n+', '\n\n', final_summary)  # Max 2 consecutive newlines
    final_summary = final_summary.strip()
    
    logging.info(f"+ Final summary complete: {len(final_summary)} characters")
    logging.info(f"   Preview: {final_summary[:300].replace(chr(10), ' ')}...")
    logging.info("="*30)
    logging.info("* DETAILED LOG: All queries and responses saved to 'llm_interactions.log'")
    logging.info("="*30)
    
    return final_summary

def generate_flashcards_from_chunks(llm_service, transcription: str, max_tokens_per_chunk: int = 5000) -> str:
    """
    Generate flashcards by processing transcription in chunks.
    Each chunk generates Q&A pairs, then all are combined.
    
    Args:
        llm_service: Summarization service to use
        transcription: Full transcription text
        max_tokens_per_chunk: Max tokens per chunk
    
    Returns:
        Combined flashcards in Q&A format
    """
    logging.info("="*60)
    logging.info("📇 FLASHCARD GENERATION")
    logging.info("="*60)
    logging.info(f"   Transcription length: {len(transcription)} chars")
    
    # Split into chunks
    chunks = chunk_text_by_tokens(transcription, max_tokens=max_tokens_per_chunk)
    num_chunks = len(chunks)
    logging.info(f"   + Split into {num_chunks} chunks")
    
    all_flashcards = []
    flashcard_number = 1
    
    for i, chunk in enumerate(chunks, 1):
        chunk_tokens = int(len(chunk.split()) * 1.3)
        
        logging.info("="*60)
        logging.info(f"📝 CHUNK {i}/{num_chunks} - Progress: {(i/num_chunks)*100:.0f}%")
        logging.info("="*60)
        logging.info(f"   Chunk size: ~{chunk_tokens:,} tokens")
        
        # Create flashcard generation prompt for this chunk
        flashcard_prompt = f"""You are an expert teaching assistant. Create flashcards from this lecture content.

Lecture content:
{chunk}

Generate flashcards as clear question-and-answer pairs. Format each flashcard EXACTLY like this:

Q: [Question based on the lecture content]
A: [Answer based on the lecture content]

Q: [Next question]
A: [Next answer]

Rules:
1. Create 5-10 flashcards from this content
2. Questions should test understanding of key concepts
3. Answers should be concise but complete
4. Use EXACTLY the format "Q: " and "A: " (with the colon and space)
5. Leave a blank line between each Q&A pair
6. No numbering, no extra labels, just Q: and A:

Output the flashcards directly:"""
        
        try:
            chunk_flashcards = llm_service.summarize(flashcard_prompt)
            
            # Clean up the response
            import re
            # Remove meta-commentary
            chunk_flashcards = re.sub(r'^(Here are|Below are).*?:\s*', '', chunk_flashcards, flags=re.IGNORECASE | re.MULTILINE)
            
            # Extract Q&A pairs and renumber them
            qa_pairs = []
            lines = chunk_flashcards.split('\n')
            current_q = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('Q:') or line.startswith('q:'):
                    if current_q:  # Save previous pair if exists
                        qa_pairs.append((current_q, None))
                    current_q = line[2:].strip()
                elif line.startswith('A:') or line.startswith('a:'):
                    if current_q:
                        qa_pairs.append((current_q, line[2:].strip()))
                        current_q = None
            
            # Format with numbering
            for q, a in qa_pairs:
                if a:  # Only add complete Q&A pairs
                    all_flashcards.append(f"**Q{flashcard_number}:** {q}\n\n**A{flashcard_number}:** {a}\n")
                    flashcard_number += 1
            
            logging.info(f"   + Generated {len(qa_pairs)} flashcards from this chunk")
            
        except Exception as e:
            logging.error(f"   ! Failed to generate flashcards from chunk {i}: {e}")
            continue
    
    # Combine all flashcards
    if all_flashcards:
        final_output = "# Flashcards\n\n" + "\n".join(all_flashcards)
        logging.info("="*30)
        logging.info(f"+ FLASHCARD GENERATION COMPLETE")
        logging.info(f"   Total flashcards: {flashcard_number - 1}")
        logging.info("="*30)
        return final_output
    else:
        logging.warning("!  No flashcards generated")
        return "No flashcards could be generated from the transcription."

def transform_to_bullet_points(summary: str, llm_model_choice: str, backend: str, base_url: str, api_key: str) -> str:
    """
    Transform a summary into a well-structured bullet point format.
    Groups content by meaning and presents as organized bullet points.
    
    Args:
        summary: The raw summary text to transform
        llm_model_choice: LLM model to use
        backend: Backend service (HuggingFace, OpenAI, etc.)
        base_url: Base URL for API
        api_key: API key
    
    Returns:
        Structured summary with bullet points grouped by topic
    """
    logging.info("   Transforming summary into structured bullet points...")
    
    bullet_transform_prompt = f"""You are an expert at organizing information. Take this summary and transform it into a well-structured bullet point format.

Original summary:
{summary}

Transform this into a clear, organized Markdown bullet point format following these EXACT rules:

1. For main section headers, use "## Header Name" (without colon at the end)
2. After each header, add a blank line
3. Then list bullet points with proper indentation:
   - Use "- " for main bullets under a header
   - Use "  - " (2 spaces + dash) for sub-bullets
   - Use "    - " (4 spaces + dash) for deeper sub-bullets
4. For numbered lists, use "1. ", "2. ", etc. with proper indentation
5. Add a blank line between different sections/headers
6. Keep bullet points concise and clear

EXAMPLE FORMAT:
## Section Name

- Main point one
- Main point two
  - Sub-point under two
  - Another sub-point
- Main point three

## Another Section

1. First numbered item
2. Second numbered item
   - Sub-point under item 2
3. Third numbered item

Output ONLY the organized Markdown (no meta-commentary, no "Here is...", just the content):"""
    
    try:
        service = get_summarization_service(backend, llm_model_choice, base_url, api_key, lazy_load=False)
        structured_summary = service.summarize(bullet_transform_prompt)
        
        # Clean up any artifacts
        import re
        structured_summary = re.sub(r'^(Here is|Below is|The organized).*?:\s*', '', structured_summary, flags=re.IGNORECASE)
        structured_summary = structured_summary.strip()
        
        # Fix common formatting issues
        # Remove colons from headers
        structured_summary = re.sub(r'^(##\s+[^:\n]+):\s*$', r'\1', structured_summary, flags=re.MULTILINE)
        
        # Ensure blank line after headers
        structured_summary = re.sub(r'^(##\s+.+)$\n(?![\n])', r'\1\n\n', structured_summary, flags=re.MULTILINE)
        
        # Ensure blank lines between sections (before new headers)
        structured_summary = re.sub(r'([^\n])\n(##\s+)', r'\1\n\n\2', structured_summary)
        
        # Fix bullet indentation: convert "°" or "◦" to proper Markdown
        structured_summary = re.sub(r'^[°◦]\s+', '  - ', structured_summary, flags=re.MULTILINE)
        
        logging.info(f"   + Structured bullet points created ({len(structured_summary)} chars)")
        return structured_summary
    except Exception as e:
        logging.error(f"   ! Failed to transform to bullet points: {e}")
        logging.warning("   Returning original summary")
        return summary

def transcribe_with_custom(audio, llm_model_choice, speech_to_text_model, prompt_template, backend, base_url, api_key, prompt_type):
    """Main workflow"""
    logging.info("="*30)
    logging.info(f"Starting transcription and summarization on {DEVICE_TYPE}")
    logging.info(f"Backend: {backend} | Whisper Model: {speech_to_text_model}")
    logging.info(f"Prompt Type: {prompt_type}")
    logging.info("="*30)
    
    # Log initial CUDA memory if available
    log_cuda_memory_usage("Workflow Start")
    
    if not audio:
        logging.warning("No audio file provided")
        return "No audio file provided.", "", ""
    
    try:
        digest = sha256_of_file(audio)
        logging.debug("Computed digest: %s", digest)
    except FileNotFoundError:
        logging.error("Audio file could not be found for hashing: %s", audio)
        return "Audio file could not be found for hashing.", "", ""

    # Check for cached transcription first
    logging.debug("Checking for cached transcription")
    transcription = get_cached_transcription(audio)
    
    # CUDA OPTIMIZATION: Sequential model loading/unloading
    whisper_service = None
    
    if transcription is None:
        # No cache - need to transcribe
        if CUDA_AVAILABLE and backend == BackendChoice.HUGGING_FACE:
            # === CUDA SEQUENTIAL PIPELINE ===
            logging.info("* CUDA Sequential Pipeline: Step 1 - Transcription")
            logging.info("=" * 30)
            
            try:
                # Step 1: Load Whisper model
                logging.info("* STEP 1.1: Loading Whisper model")
                whisper_service = get_transcription_service(speech_to_text_model, lazy_load=False)
                
                # Step 2: Transcribe audio with good quality
                logging.info("* STEP 1.2: Transcribing audio")
                transcription = whisper_service.transcribe(audio)
                logging.info(f"+ Transcription completed: {len(transcription)} characters")
                
                # Step 3: Unload Whisper and clear VRAM for LLM
                if hasattr(whisper_service, 'unload_model'):
                    # Show memory BEFORE cleanup
                    logging.info("=" * 30)
                    logging.info("* GPU MEMORY BEFORE CLEANUP:")
                    logging.info("=" * 30)
                    mem_info_before = get_cuda_memory_info()
                    if "error" not in mem_info_before:
                        allocated_before = mem_info_before['allocated_gb']
                        reserved_before = mem_info_before['reserved_gb']
                        free_before = mem_info_before['free_gb']
                        total_gb = mem_info_before['total_gb']
                        used_percent_before = (allocated_before / total_gb) * 100
                        
                        logging.info(f"   Allocated: {allocated_before:.2f} GB ({used_percent_before:.1f}%)")
                        logging.info(f"   Reserved:  {reserved_before:.2f} GB")
                        logging.info(f"   Free:      {free_before:.2f} GB / {total_gb:.2f} GB")
                    logging.info("=" * 30)
                    
                    logging.info("* STEP 1.3: Unloading Whisper to free VRAM")
                    whisper_service.unload_model()
                    # Extra cleanup to ensure VRAM is completely cleared
                    clear_cuda_cache()
                    # Brief pause to allow CUDA to fully release memory
                    time.sleep(0.5)
                    
                    # Show memory AFTER cleanup
                    logging.info("=" * 30)
                    logging.info("* GPU MEMORY AFTER CLEANUP:")
                    logging.info("=" * 30)
                    mem_info_after = get_cuda_memory_info()
                    if "error" not in mem_info_after:
                        allocated_after = mem_info_after['allocated_gb']
                        reserved_after = mem_info_after['reserved_gb']
                        free_after = mem_info_after['free_gb']
                        total_gb = mem_info_after['total_gb']
                        used_percent_after = (allocated_after / total_gb) * 100
                        
                        # Calculate how much was freed
                        if "error" not in mem_info_before:
                            freed_gb = allocated_before - allocated_after
                            logging.info(f"   Freed:     {freed_gb:.2f} GB")
                        
                        logging.info(f"   Allocated: {allocated_after:.2f} GB ({used_percent_after:.1f}%)")
                        logging.info(f"   Reserved:  {reserved_after:.2f} GB")
                        logging.info(f"   Free:      {free_after:.2f} GB / {total_gb:.2f} GB")
                        
                        if used_percent_after < 10:
                            logging.info("   + EXCELLENT - VRAM successfully freed!")
                        elif used_percent_after < 25:
                            logging.info("   + GOOD - Most VRAM freed")
                        else:
                            logging.info("   ! WARNING - VRAM cleanup may be incomplete")
                    logging.info("=" * 30)
                    logging.info("+ VRAM cleared, ready for next step")
                
            except Exception as e:
                logging.error(f"! Transcription failed: {e}")
                return f"Error during transcription: {e}", "", format_file_metadata(audio, digest)
        else:
            # Non-CUDA or non-HF backend: use simple approach
            transcription = transcribe_file(audio, speech_to_text_model)
    else:
        logging.info("+ Loaded transcription from cache")
    
    # Cache the transcription
    if not transcription.startswith("Error:") and transcription.strip():
        logging.debug("Writing transcription to cache")
        try:
            cache_path = CACHE_DIR / f"{digest}.txt"
            cache_path.write_text(transcription, encoding="utf-8")
        except Exception as e:
            logging.warning(f"Failed to write cache: {e}")

    file_meta = format_file_metadata(audio, digest)

    if transcription.startswith("Error:") or not transcription.strip():
        logging.warning("Transcription failed or empty, returning without summary")
        return transcription, "", file_meta

    # Display transcription stats
    word_count = len(transcription.split())
    char_count = len(transcription)
    logging.info("=" * 30)
    logging.info(f"* Transcription Statistics:")
    logging.info(f"   Characters: {char_count:,}")
    logging.info(f"   Words: {word_count:,}")
    logging.info(f"   Estimated tokens: ~{int(word_count * 1.3):,}")
    logging.info("=" * 30)
    
    # Clean transcription to reduce token count if needed
    if CUDA_AVAILABLE and len(transcription) > 15000:  # ~20k+ tokens
        logging.info("* Long transcription detected, applying cleanup...")
        logging.info(f"   Original: {char_count:,} characters (~{word_count:,} words)")
        transcription = clean_transcription(transcription)
        word_count_after = len(transcription.split())
        logging.info(f"   After cleanup: {len(transcription):,} characters (~{word_count_after:,} words)")
        logging.info(f"   Reduction: {word_count - word_count_after:,} words ({((word_count - word_count_after)/word_count)*100:.1f}%)")
    
    # Prepare prompt for summarization
    logging.info("Preparing prompt for summarization")
    if not prompt_template.strip():
        prompt_template = PROMPT_MAPPING[PromptChoice.SUMMARY.value]
    elif "{transcription}" not in prompt_template:
        prompt_template += " {transcription}"

    # Check if we need chunked summarization (for 8GB VRAM constraint)
    estimated_tokens = int(word_count * 1.3)
    # CRITICAL: Use chunking if > 2k tokens to prevent KV cache OOM on 8GB GPU!
    # KV cache memory grows linearly with input tokens (~1GB per 1000 tokens)
    use_chunked_summarization = estimated_tokens > 2000  # Use chunking to keep KV cache manageable
    
    if use_chunked_summarization:
        logging.info("="*30)
        logging.info(f"*  Transcription is LONG ({estimated_tokens:,} tokens > 25k limit)")
        logging.info(f"   Will use CHUNKED SUMMARIZATION (MapReduce pattern)")
        logging.info("="*30)
    
    prompt = prompt_template.replace("{transcription}", transcription)
    
    # CUDA OPTIMIZATION: Load LLM after Whisper is unloaded
    if CUDA_AVAILABLE and backend == BackendChoice.HUGGING_FACE:
        # === CUDA SEQUENTIAL PIPELINE: Part 2 ===
        logging.info("=" * 30)
        logging.info("* CUDA Sequential Pipeline: Step 2 - Summarization")
        logging.info("=" * 30)
        
        try:
            # Step 4: Load LLM model (Whisper already unloaded)
            logging.info("* STEP 2.1: Loading LLM model")
            llm_service = get_summarization_service(
                backend, llm_model_choice, base_url, api_key, lazy_load=False
            )
            
            # Step 5: Conduct summarization/flashcard action
            logging.info("* STEP 2.2: Generating output")
            
            # Check if this is flashcard generation
            if prompt_type == PromptChoice.FLASHCARDS.value:
                # Use dedicated flashcard generation
                logging.info("   Mode: FLASHCARD GENERATION")
                summary = generate_flashcards_from_chunks(llm_service, transcription, max_tokens_per_chunk=5000)
            elif use_chunked_summarization:
                # Use iterative refinement chunking for long transcriptions
                logging.info("   Mode: CHUNKED SUMMARIZATION")
                summary = summarize_in_chunks(llm_service, transcription, prompt_template, max_tokens_per_chunk=5000)
            else:
                # Direct summarization for shorter transcriptions
                logging.info("   Mode: DIRECT SUMMARIZATION")
                summary = llm_service.summarize(prompt)
            
            logging.info(f"+ Output generated: {len(summary)} characters")
            
            # Step 6: Unload LLM
            if hasattr(llm_service, 'unload_model'):
                logging.info("* STEP 2.3: Unloading LLM model")
                llm_service.unload_model()
            
            # Final cleanup and memory report
            logging.info("* Final VRAM cleanup...")
            clear_cuda_cache()
            time.sleep(0.5)
            
            # Get detailed memory info
            mem_info = get_cuda_memory_info()
            if "error" not in mem_info:
                allocated_gb = mem_info['allocated_gb']
                free_gb = mem_info['free_gb']
                total_gb = mem_info['total_gb']
                used_percent = (allocated_gb / total_gb) * 100
                
                logging.info("=" * 30)
                logging.info("* FINAL GPU MEMORY STATUS:")
                logging.info(f"   Allocated: {allocated_gb:.2f} GB ({used_percent:.1f}%)")
                logging.info(f"   Free:      {free_gb:.2f} GB / {total_gb:.2f} GB")
                
                if used_percent < 10:
                    logging.info("   + EXCELLENT - VRAM completely freed!")
                elif used_percent < 25:
                    logging.info("   + GOOD - VRAM mostly freed")
                else:
                    logging.info("   *  WARNING - Some VRAM still in use")
                logging.info("=" * 30)
            
        except Exception as e:
            logging.error(f"! Summarization failed: {e}")
            summary = f"Error generating summary: {e}"
    else:
        # Non-CUDA or non-HF backend: use simple approach
        llm_service = get_summarization_service(backend, llm_model_choice, base_url, api_key, lazy_load=False)
        
        if prompt_type == PromptChoice.FLASHCARDS.value:
            # Use dedicated flashcard generation
            summary = generate_flashcards_from_chunks(llm_service, transcription, max_tokens_per_chunk=5000)
        elif use_chunked_summarization:
            # Need to create service for chunking
            summary = summarize_in_chunks(llm_service, transcription, prompt_template, max_tokens_per_chunk=5000)
        else:
            summary = safe_generate_summary(prompt, llm_model_choice, backend, base_url, api_key)
        
        # Still do cleanup for CUDA even with other backends
        if CUDA_AVAILABLE:
            clear_cuda_cache()

    # POST-PROCESSING: Transform to structured bullet points if that prompt type is selected
    if prompt_type == PromptChoice.SUMMARY_BULLETS.value:
        logging.info("="*30)
        logging.info("* POST-PROCESSING: Transforming to structured bullet points")
        logging.info("="*30)
        summary = transform_to_bullet_points(summary, llm_model_choice, backend, base_url, api_key)
        logging.info("+ Bullet point transformation complete")
    
    logging.info("="*30)
    logging.info("* COMPLETE: Transcription and summarization workflow finished")
    logging.info("="*30)
    return summary, transcription, file_meta

# -------------------------
# Gradio UI
# -------------------------
CUSTOM_CSS = """
/* ---------- File Metadata Box ---------- */



/* ---------- Text Areas ---------- */
#summary-output textarea,
#transcription-output textarea {
    font-size: 1rem;
    line-height: 1.6;
    background-color: #fefefe;
    color: #111827;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    padding: 10px;
    resize: vertical;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

#summary-output textarea:focus,
#transcription-output textarea:focus {
    border-color: #2563eb;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2);
    outline: none;
}

/* ---------- Buttons ---------- */
.gr-button-primary {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    border: none;
    color: #ffffff;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    border-radius: 8px;
    padding: 10px 18px;
    transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.2s ease;
}

.gr-button-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35);
    background: linear-gradient(135deg, #1e40af, #1d4ed8);
}

/* Red highlighted buttons */
.red-button {
    background: linear-gradient(135deg, #dc2626, #b91c1c) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3) !important;
    border-radius: 8px !important;
    padding: 10px 18px !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.2s ease !important;
}

.red-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(220, 38, 38, 0.4) !important;
    background: linear-gradient(135deg, #b91c1c, #991b1b) !important;
}

/* ---------- Body & Labels ---------- */
body {
    background-color: #f3f4f6;
    color: #1f2937;
    font-family: "Inter", sans-serif;
}

label {
    color: #111827 !important;
    font-weight: 500;
}

/* ---------- Accordions ---------- */
.gr-accordion {
    border-radius: 8px;
    border: 1px solid #d1d5db;
    background: #ffffff;
    padding: 10px 14px;
}

.gr-accordion .gr-accordion-title {
    font-weight: 600;
    color: #1d4ed8;
}

/* ---------- Misc ---------- */
.gr-row, .gr-column {
    gap: 12px;
}

"""

def get_hardware_status() -> str:
    """Generate hardware status message for UI."""
    status_parts = []
    
    # Primary device indicator
    status_parts.append(f"**Active Device: {DEVICE_TYPE}**")
    
    # MLX status
    if MLX_AVAILABLE:
        status_parts.append("+ MLX (Apple Silicon)")
    else:
        status_parts.append("! MLX")
    
    # CUDA status
    if CUDA_AVAILABLE:
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        status_parts.append(f"+ CUDA ({device_name}, {memory_gb:.1f}GB)")
    else:
        status_parts.append("! CUDA")
    
    # CPU status
    if CPU_ONLY:
        status_parts.append("! CPU Mode (No GPU acceleration)")
    
    # Additional info
    if HF_AVAILABLE:
        status_parts.append("+ Transformers")
    
    return " | ".join(status_parts)

with gr.Blocks(theme=Soft(), css=CUSTOM_CSS) as demo:
    gr.Markdown("# Audio Transcription & Summarization")
    gr.Markdown("Upload an audio file, choose a Whisper model, and generate a tailored summary using the local Granite model.")
    gr.Markdown(f"**Hardware Status:** {get_hardware_status()}")
    
    # Show CUDA-specific optimizations
    if CUDA_AVAILABLE:
        mem_info = get_cuda_memory_info()
        if "error" not in mem_info:
            gr.Markdown(
                f"* **GPU Memory:** {mem_info['free_gb']:.1f}GB free / {mem_info['total_gb']:.1f}GB total\n\n"
                f"**CUDA Optimizations Enabled:**\n"
                f"- 4-bit quantized models (fits in 8GB VRAM)\n"
                f"- Long audio support (>30s) with automatic chunking\n"
                f"- FP16 precision for faster inference"
            )

    with gr.Row():
        with gr.Column(scale=3):
            audio_input = gr.Audio(type="filepath", label="Upload Audio File")
            file_details = gr.Markdown("", elem_id="file-meta")
            with gr.Accordion("Full Transcription", open=False):
                transcription_output = gr.Textbox(label="Transcription", lines=10, show_copy_button=True, elem_id="transcription-output")

        with gr.Column(scale=2):
            # Set default backend based on hardware
            if CUDA_AVAILABLE:
                default_backend = BackendChoice.HUGGING_FACE
                default_models = get_default_hf_models()
                default_model = default_models[0] if default_models else None
                show_api_key = False
                show_base_url = False
                show_fetch_btn = False
            else:
                default_backend = BackendChoice.LOCAL_OPENAI
                default_models = []
                default_model = None
                show_api_key = False
                show_base_url = True
                show_fetch_btn = False
            
            backend_dropdown = gr.Dropdown(
                choices=[b.value for b in BackendChoice],
                label="LLM Backend" + (" (CUDA Optimized)" if CUDA_AVAILABLE else ""),
                value=default_backend
            )
            api_key_input = gr.Textbox(label="API Key", visible=show_api_key, type="password")
            base_url_input = gr.Textbox(label="Base URL", visible=show_base_url, placeholder="http://localhost:1234/v1")
            fetch_models_btn = gr.Button("Fetch Available Models", visible=show_fetch_btn)
            llm_model_dropdown = gr.Dropdown(
                choices=default_models,
                label="Language Model" + (" (4-bit Quantized)" if CUDA_AVAILABLE else ""),
                value=default_model
            )
            available_whisper_models = get_available_whisper_models()
            recommended_whisper_model = get_recommended_whisper_model()
            speech_to_text_dropdown = gr.Dropdown(
                choices=available_whisper_models,
                label=f"Speech-To-Text Model (Recommended: {recommended_whisper_model.split('/')[-1]})",
                value=recommended_whisper_model
            )
            prompt_dropdown = gr.Dropdown(
                choices=[p.value for p in PromptChoice] + ["Custom"],
                label="Prompt Type",
                value=PromptChoice.SUMMARY
            )
            with gr.Accordion("Prompt Template", open=False):
                prompt_textbox = gr.Textbox(
                    label="Template",
                    value=PROMPT_MAPPING[PromptChoice.SUMMARY],
                    interactive=False,
                    lines=6,
                    show_copy_button=True
                )
            with gr.Row():
                submit_btn = gr.Button("Transcribe & Summarise", variant="primary", elem_classes="red-button")
                clear_btn = gr.ClearButton(
                    [audio_input, transcription_output, prompt_textbox, llm_model_dropdown, speech_to_text_dropdown, prompt_dropdown, backend_dropdown, api_key_input, base_url_input, fetch_models_btn, file_details],
                    value="Clear",
                    elem_classes="red-button"
                )

    summary_output = gr.Markdown(label="Summary", elem_id="summary-output")

    prompt_dropdown.change(update_prompt, inputs=prompt_dropdown, outputs=prompt_textbox)
    backend_dropdown.change(update_backend_settings, inputs=backend_dropdown, outputs=[api_key_input, base_url_input, fetch_models_btn, llm_model_dropdown])
    fetch_models_btn.click(fetch_and_update_models, inputs=[base_url_input, api_key_input], outputs=llm_model_dropdown)
    submit_btn.click(
        transcribe_with_custom,
        inputs=[audio_input, llm_model_dropdown, speech_to_text_dropdown, prompt_textbox, backend_dropdown, base_url_input, api_key_input, prompt_dropdown],
        outputs=[summary_output, transcription_output, file_details],
        queue=True
    )

logging.info("Launching Gradio interface")
logging.info("(Note: You may see some h11 protocol warnings - these can be safely ignored.)")
logging.info("Access the app at: http://localhost:7860")
logging.info("=" * 30)

try:
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        quiet=False
    )
except KeyboardInterrupt:
    logging.info("\n" + "=" * 30)
    logging.info("Shutting down gracefully...")
    logging.info("=" * 30)
except Exception as e:
    logging.error(f"Error during launch: {e}")
    logging.error("This may be due to port conflicts or network issues.")
    logging.info("Try: python app.py --server-port 7861")
