"""
LLM wrappers for OpenAI, OpenRouter, and local Transformers models.
"""

import logging
import os
from typing import Optional, Dict, Any, List
import json

import requests
import httpx
from langchain.llms.base import LLM
from langchain.schema import HumanMessage

from .utils import Config

logger = logging.getLogger(__name__)

class OpenAILLM:
    """OpenAI LLM wrapper using langchain-openai."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo", 
                 max_tokens: int = 512, temperature: float = 0.2):
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = None
        
        # Initialize OpenAI client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the OpenAI client."""
        try:
            from langchain_openai import ChatOpenAI
            
            self.client = ChatOpenAI(
                openai_api_key=self.api_key,
                model_name=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=30
            )
            
            logger.info(f"Initialized OpenAI client with model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text response from prompt."""
        try:
            # Override parameters if provided
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            temperature = kwargs.get('temperature', self.temperature)
            
            # Update client parameters if different
            if max_tokens != self.max_tokens or temperature != self.temperature:
                self.client.max_tokens = max_tokens
                self.client.temperature = temperature
            
            # Generate response
            message = HumanMessage(content=prompt)
            response = self.client([message])
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating response with OpenAI: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if the OpenAI API is available."""
        try:
            # Test with a simple prompt
            test_response = self.generate("Hello", max_tokens=5)
            return len(test_response) > 0
        except Exception:
            return False

class OpenRouterLLM:
    """OpenRouter LLM wrapper with API key authentication."""
    
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1", 
                 model_name: str = "meta-llama/llama-3.2-3b-instruct:free",
                 max_tokens: int = 512, temperature: float = 0.2):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Headers for requests
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/pdf-research-assistant",
            "X-Title": "PDF Research Assistant"
        }
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text response from prompt using OpenRouter."""
        try:
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            temperature = kwargs.get('temperature', self.temperature)
            
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 1.0,
                "frequency_penalty": 0,
                "presence_penalty": 0
            }
            
            # Make request to OpenRouter
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            # Extract generated text
            if 'choices' in response_data and response_data['choices']:
                return response_data['choices'][0]['message']['content'].strip()
            else:
                raise ValueError("No response generated")
                
        except Exception as e:
            logger.error(f"Error generating response with OpenRouter: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if OpenRouter is available."""
        try:
            # Test with a simple prompt
            test_response = self.generate("Hello", max_tokens=5)
            return len(test_response) > 0
        except Exception as e:
            logger.warning(f"OpenRouter not available: {e}")
            return False

class LocalTransformersLLM:
    """Local Transformers model wrapper as fallback."""
    
    def __init__(self, model_name: str = "distilgpt2", 
                 max_tokens: int = 512, temperature: float = 0.2):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tokenizer = None
        self.model = None
        
        # Initialize model with error handling
        try:
            self._initialize_model()
        except Exception as e:
            logger.error(f"Failed to initialize local model: {e}")
            # Don't raise - let the model be None and is_available() return False
    
    def _initialize_model(self):
        """Initialize the local Transformers model."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            logger.info(f"Loading local model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Successfully loaded local Transformers model")
            
        except Exception as e:
            logger.error(f"Error loading local model: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text response using local model."""
        try:
            import torch
            
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            temperature = kwargs.get('temperature', self.temperature)
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the original prompt from the response
            response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response with local model: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if the local model is available."""
        try:
            return self.model is not None and self.tokenizer is not None
        except Exception:
            return False

class LLMManager:
    """Manages multiple LLM providers with fallback logic."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.current_llm = None
        self.current_provider = None
        
        # Initialize the best available LLM
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the best available LLM based on configuration."""
        providers_tried = []
        
        # 1. Try OpenAI if API key is available
        if self.config.openai_api_key:
            try:
                self.current_llm = OpenAILLM(
                    api_key=self.config.openai_api_key,
                    model_name=self.config.model_name,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
                
                if self.current_llm.is_available():
                    self.current_provider = "OpenAI"
                    logger.info("Successfully initialized OpenAI LLM")
                    return
                else:
                    providers_tried.append("OpenAI (not available)")
                    
            except Exception as e:
                providers_tried.append(f"OpenAI (error: {e})")
                logger.warning(f"Failed to initialize OpenAI: {e}")
        
        # 2. Try OpenRouter if API key is available
        if self.config.openrouter_api_key:
            try:
                self.current_llm = OpenRouterLLM(
                    api_key=self.config.openrouter_api_key,
                    base_url=self.config.openrouter_base_url,
                    model_name=self.config.openrouter_model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
                
                if self.current_llm.is_available():
                    self.current_provider = "OpenRouter"
                    logger.info("Successfully initialized OpenRouter LLM")
                    return
                else:
                    providers_tried.append("OpenRouter (not available)")
                    
            except Exception as e:
                providers_tried.append(f"OpenRouter (error: {e})")
                logger.warning(f"Failed to initialize OpenRouter: {e}")
        
        # 3. Try local Transformers as fallback
        try:
            self.current_llm = LocalTransformersLLM(
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            if self.current_llm.is_available():
                self.current_provider = "Local Transformers"
                logger.info("Successfully initialized Local Transformers LLM")
                return
            else:
                providers_tried.append("Local Transformers (not available)")
                
        except Exception as e:
            providers_tried.append(f"Local Transformers (error: {e})")
            logger.warning(f"Failed to initialize Local Transformers: {e}")
        
        # If no LLM is available, set up a dummy provider with helpful error message
        logger.warning(f"No LLM providers available. Tried: {', '.join(providers_tried)}")
        self.current_llm = None
        self.current_provider = "No Provider Available"
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the current LLM."""
        if self.current_llm is None:
            raise RuntimeError("No LLM available for text generation. Please configure an API key for OpenAI or OpenRouter.")
        
        try:
            return self.current_llm.generate(prompt, **kwargs)
        except Exception as e:
            logger.error(f"Error generating text with {self.current_provider}: {e}")
            raise
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current LLM provider."""
        return {
            'provider': self.current_provider,
            'model_name': getattr(self.current_llm, 'model_name', 'unknown'),
            'max_tokens': getattr(self.current_llm, 'max_tokens', self.config.max_tokens),
            'temperature': getattr(self.current_llm, 'temperature', self.config.temperature),
            'available': self.current_llm is not None
        }
    
    def switch_provider(self, provider: str, **kwargs) -> bool:
        """Switch to a different LLM provider."""
        try:
            if provider.lower() == "openai":
                if not self.config.openai_api_key:
                    raise ValueError("OpenAI API key not available")
                
                self.current_llm = OpenAILLM(
                    api_key=self.config.openai_api_key,
                    model_name=kwargs.get('model_name', self.config.model_name),
                    max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                    temperature=kwargs.get('temperature', self.config.temperature)
                )
                self.current_provider = "OpenAI"
                
            elif provider.lower() == "openrouter":
                if not self.config.openrouter_api_key:
                    raise ValueError("OpenRouter API key not available")
                
                self.current_llm = OpenRouterLLM(
                    api_key=self.config.openrouter_api_key,
                    base_url=self.config.openrouter_base_url,
                    model_name=kwargs.get('model_name', self.config.openrouter_model),
                    max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                    temperature=kwargs.get('temperature', self.config.temperature)
                )
                self.current_provider = "OpenRouter"
                
            elif provider.lower() == "local":
                self.current_llm = LocalTransformersLLM(
                    model_name=kwargs.get('model_name', "distilgpt2"),
                    max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                    temperature=kwargs.get('temperature', self.config.temperature)
                )
                self.current_provider = "Local Transformers"
                
            else:
                raise ValueError(f"Unknown provider: {provider}")
            
            # Test the new provider
            if not self.current_llm.is_available():
                raise RuntimeError(f"Provider {provider} is not available")
            
            logger.info(f"Successfully switched to {provider}")
            return True
            
        except Exception as e:
            logger.error(f"Error switching to {provider}: {e}")
            return False