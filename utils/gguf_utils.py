import os
import json
import time
from typing import List, Dict, Any, Optional
from llama_cpp import Llama


class GGUFModelManager:
    """Manager for GGUF models using llama-cpp-python"""
    
    def __init__(self):
        self.models = {}
        
    def load_model(self, model_path: str, model_config: Dict[str, Any]) -> Llama:
        """Load a GGUF model with given configuration"""
        if model_path in self.models:
            return self.models[model_path]
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GGUF model file not found: {model_path}")
        
        # Default configuration
        default_config = {
            "n_ctx": 4096,
            "n_threads": None,
            "n_gpu_layers": 0,
            "verbose": False,
            "seed": -1,
            "f16_kv": True,
            "use_mlock": False,
            "use_mmap": True,
        }
        
        # Merge with provided config
        config = {**default_config, **model_config}
        
        print(f"Loading GGUF model: {model_path}")
        print(f"Config: {config}")
        
        model = Llama(model_path=model_path, **config)
        self.models[model_path] = model
        return model
    
    def unload_model(self, model_path: str):
        """Unload a model from memory"""
        if model_path in self.models:
            del self.models[model_path]
    
    def unload_all_models(self):
        """Unload all models from memory"""
        self.models.clear()


# Global model manager instance
model_manager = GGUFModelManager()


def format_messages_for_gguf(messages: List[Dict], model_type: str = "gguf") -> str:
    """Format messages for GGUF model inference"""
    if not messages:
        return ""
    
    # Handle different message formats
    if isinstance(messages, str):
        return messages
    
    # For chat-formatted messages
    formatted_messages = []
    for message in messages:
        if isinstance(message, dict) and "role" in message and "content" in message:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted_messages.append(f"System: {content}")
            elif role == "user":
                formatted_messages.append(f"User: {content}")
            elif role == "assistant":
                formatted_messages.append(f"Assistant: {content}")
        elif isinstance(message, str):
            formatted_messages.append(message)
    
    return "\n\n".join(formatted_messages) + "\n\nAssistant:"


def generate_gguf_response(
    model_path: str,
    model_config: Dict[str, Any],
    messages: List[Dict],
    max_tokens: int = 300,
    temperature: float = 0.0,
    stop_sequences: Optional[List[str]] = None,
    max_retries: int = 3,
) -> str:
    """Generate response using GGUF model"""
    
    # Load model
    model = model_manager.load_model(model_path, model_config)
    
    # Format messages
    prompt = format_messages_for_gguf(messages)
    
    # Default stop sequences
    if stop_sequences is None:
        stop_sequences = ["User:", "Human:", "\n\nUser:", "\n\nHuman:"]
    
    retries = 0
    while retries < max_retries:
        try:
            # Generate response
            response = model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_sequences,
                echo=False,
                stream=False,
            )
            
            # Extract text from response
            if isinstance(response, dict) and "choices" in response:
                text = response["choices"][0]["text"]
            else:
                text = str(response)
            
            # Clean up response
            text = text.strip()
            
            # Remove any remaining stop sequences
            for stop_seq in stop_sequences:
                if text.endswith(stop_seq):
                    text = text[:-len(stop_seq)].strip()
            
            return text
            
        except Exception as e:
            print(f"GGUF generation error (attempt {retries + 1}): {e}")
            retries += 1
            if retries >= max_retries:
                raise Exception(f"GGUF generation failed after {max_retries} attempts: {e}")
            time.sleep(1)


def count_gguf_tokens(text: str, model_path: str, model_config: Dict[str, Any]) -> int:
    """Count tokens for GGUF model (approximate using the model's tokenizer)"""
    try:
        model = model_manager.load_model(model_path, model_config)
        tokens = model.tokenize(text.encode('utf-8'))
        return len(tokens)
    except Exception as e:
        print(f"Token counting error: {e}")
        # Fallback to simple word-based estimation
        return len(text.split()) * 1.3  # Rough approximation


def get_gguf_model_info(model_name: str, gguf_models_config: Dict[str, Any]) -> tuple:
    """Get GGUF model path and configuration"""
    if model_name not in gguf_models_config:
        raise ValueError(f"GGUF model '{model_name}' not found in configuration")
    
    model_info = gguf_models_config[model_name]
    model_path = model_info.get("path")
    model_config = model_info.get("config", {})
    
    if not model_path:
        raise ValueError(f"No path specified for GGUF model '{model_name}'")
    
    return model_path, model_config


def format_gguf_conversation(messages: List[Dict], turn_number: int = 3) -> str:
    """Format multi-turn conversation for GGUF models"""
    conversation_parts = []
    turn_count = 0
    
    for message in messages:
        if isinstance(message, dict) and "role" in message and "content" in message:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                conversation_parts.append(f"User: {content}")
                turn_count += 1
            elif role == "assistant":
                conversation_parts.append(f"Assistant: {content}")
                
            if turn_count >= turn_number and role == "assistant":
                break
    
    return "\n\n".join(conversation_parts)