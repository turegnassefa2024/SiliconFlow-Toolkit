#!/usr/bin/env python3
"""
SiliconFlow Ultimate Configuration Builder with Performance Optimizations
Dynamically fetches ALL models and builds optimized configurations
"""
import json
import requests
import os
import sys
from pathlib import Path
from datetime import datetime
import getpass

class SiliconFlowConfigBuilder:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.siliconflow.com/v1"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        
        # Model type mappings from documentation
        self.model_categories = {
            "chat": {
                "endpoint": "/chat/completions",
                "capabilities": ["text-generation", "tool-calling", "reasoning"],
                "params": self._get_chat_params()
            },
            "embedding": {
                "endpoint": "/embeddings",
                "capabilities": ["text-embedding", "semantic-search"],
                "params": self._get_embedding_params()
            },
            "reranker": {
                "endpoint": "/rerank",
                "capabilities": ["document-reranking", "relevance-scoring"],
                "params": self._get_rerank_params()
            },
            "text-to-image": {
                "endpoint": "/images/generations",  # Inferred from API reference
                "capabilities": ["image-generation"],
                "params": self._get_image_params()
            },
            "image-to-image": {
                "endpoint": "/images/edits",  # Inferred
                "capabilities": ["image-editing", "image-transformation"],
                "params": {}
            }
        }
    
    def _get_chat_params(self):
        """Extract all chat completion parameters from docs"""
        return {
            "stream": {"type": "boolean", "default": True},  # Optimized: Streaming enabled by default
            "max_tokens": {"type": "integer", "default": 2048},  # Optimized: Lower default
            "enable_thinking": {"type": "boolean", "default": False},  # Optimized: Disabled by default
            "thinking_budget": {"type": "integer", "range": [128, 32768], "default": 1024},  # Optimized: Lower default
            "min_p": {"type": "float", "range": [0, 1], "default": 0.05},
            "stop": {"type": "array", "default": []},
            "temperature": {"type": "float", "default": 0.3},  # Optimized: Lower for determinism
            "top_p": {"type": "float", "default": 0.9},
            "top_k": {"type": "integer", "default": 40},  # Optimized: Lower for speed
            "frequency_penalty": {"type": "float", "default": 0.1},  # Optimized: Lower penalty
            "n": {"type": "integer", "default": 1},
            "response_format": {"type": "object", "default": {"type": "text"}},
            "tools": {"type": "array", "max_items": 128},
            "tool_choice": {"type": "object", "default": {"type": "auto", "disable_parallel_tool_use": False}}  # Optimized: Allow parallel
        }
    
    def _get_embedding_params(self):
        """Extract embedding parameters"""
        return {
            "encoding_format": {"type": "string", "options": ["float", "base64"], "default": "float"},
            "dimensions": {"type": "integer", "default": 768},  # Optimized: Lower dimensions for speed
            "batch_size": {"type": "integer", "default": 32}  # Optimized: Added batch size
        }
    
    def _get_rerank_params(self):
        """Extract rerank parameters"""
        return {
            "top_n": {"type": "integer", "default": 5},  # Optimized: Lower default
            "return_documents": {"type": "boolean", "default": True},
            "max_chunks_per_doc": {"type": "integer", "default": 5},  # Optimized: Lower default
            "overlap_tokens": {"type": "integer", "max": 80, "default": 50}
        }
    
    def _get_image_params(self):
        """Image generation parameters (inferred structure)"""
        return {
            "size": {"type": "string", "options": ["512x512", "1024x1024"], "default": "512x512"},  # Optimized: Smaller default
            "quality": {"type": "string", "options": ["standard", "hd"], "default": "standard"},  # Optimized: Standard quality
            "style": {"type": "string", "options": ["vivid", "natural"], "default": "natural"}
        }
    
    def fetch_all_models(self):
        """Fetch ALL models from SiliconFlow API with all types"""
        all_models = []
        
        print("🔄 Fetching models from SiliconFlow API...")
        
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=15  # Optimized: Lower timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                all_models = data.get("data", [])
                print(f"✅ Found {len(all_models)} total models")
                
                # If API returns empty or limited, use known models from docs
                if len(all_models) < 10:
                    print("⚠️  API returned limited models, using documented models")
                    all_models = self._get_documented_models()
            else:
                print(f"⚠️  API returned {response.status_code}, using documented models")
                all_models = self._get_documented_models()
                
        except Exception as e:
            print(f"⚠️  Error fetching models: {e}, using documented models")
            all_models = self._get_documented_models()
        
        # Categorize models by inferred type
        categorized = self._categorize_models(all_models)
        return categorized
    
    def _get_documented_models(self):
        """Fallback to models explicitly documented"""
        documented_models = []
        
        # Chat models from chat completions docs
        chat_models = [
            "Qwen/Qwen2.5-7B-Instruct",  # Optimized: Fast model first
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            "deepseek-ai/DeepSeek-V3.2",
            "Qwen/Qwen2.5-72B-Instruct",
            "moonshotai/Kimi-K2-Instruct",
            "Qwen/Qwen2.5-VL-72B-Instruct",
            "baidu/ERNIE-4.5-300B-A47B",
            "zai-org/GLM-4.6",
            "tencent/Hunyuan-A13B-Instruct",
            "Qwen/Qwen3-235B-A22B-Instruct-2507"
        ]
        
        # Embedding models
        embedding_models = [
            "Qwen/Qwen3-Embedding-4B",  # Optimized: Medium size first
            "Qwen/Qwen3-Embedding-8B",
            "Qwen/Qwen3-Embedding-0.6B",
            "BAAI/bge-large-zh-v1.5"
        ]
        
        # Reranker models
        reranker_models = [
            "Qwen/Qwen3-Reranker-4B",  # Optimized: Medium size first
            "Qwen/Qwen3-Reranker-8B",
            "Qwen/Qwen3-Reranker-0.6B",
            "BAAI/bge-reranker-v2-m3"
        ]
        
        # Add all with placeholder objects
        for model_id in chat_models:
            documented_models.append({
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "",
                "inferred_type": "chat"
            })
        
        for model_id in embedding_models:
            documented_models.append({
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "",
                "inferred_type": "embedding"
            })
        
        for model_id in reranker_models:
            documented_models.append({
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "",
                "inferred_type": "reranker"
            })
        
        return documented_models
    
    def _categorize_models(self, models):
        """Categorize models by their likely function based on name patterns"""
        categorized = {
            "chat": [], "embedding": [], "reranker": [],
            "vision": [], "audio": [], "image": [], "video": []
        }
        
        for model in models:
            model_id = model.get("id", "").lower()
            
            # Enhanced pattern matching
            if any(keyword in model_id for keyword in ["embedding", "bge-", "embed"]):
                model["category"] = "embedding"
                categorized["embedding"].append(model)
            elif any(keyword in model_id for keyword in ["rerank", "reranker"]):
                model["category"] = "reranker"
                categorized["reranker"].append(model)
            elif any(keyword in model_id for keyword in ["vl-", "vision", "visual", "-vl", "glm-4.5v", "glm-4.6v"]):
                model["category"] = "vision"
                categorized["vision"].append(model)
            elif any(keyword in model_id for keyword in ["stable-diffusion", "dalle", "midjourney", "imagen"]):
                model["category"] = "image"
                categorized["image"].append(model)
            elif any(keyword in model_id for keyword in ["whisper", "speech", "audio", "tts"]):
                model["category"] = "audio"
                categorized["audio"].append(model)
            elif any(keyword in model_id for keyword in ["video", "sora", "gen-2"]):
                model["category"] = "video"
                categorized["video"].append(model)
            else:
                # Default to chat for most models
                model["category"] = "chat"
                categorized["chat"].append(model)
        
        # Sort by performance: smaller/faster models first
        for category in categorized:
            categorized[category] = self._sort_models_by_performance(categorized[category])
        
        return categorized
    
    def _sort_models_by_performance(self, models):
        """Sort models by estimated performance (smaller/faster first)"""
        def performance_score(model_id):
            model_id_lower = model_id.lower()
            score = 0
            
            # Smaller models get higher priority (negative weight)
            if "0.6b" in model_id_lower or "0.6-b" in model_id_lower:
                score -= 100
            elif "1b" in model_id_lower or "1-b" in model_id_lower:
                score -= 90
            elif "4b" in model_id_lower or "4-b" in model_id_lower:
                score -= 80
            elif "7b" in model_id_lower or "7-b" in model_id_lower:
                score -= 70
            elif "8b" in model_id_lower or "8-b" in model_id_lower:
                score -= 60
            elif "14b" in model_id_lower or "14-b" in model_id_lower:
                score -= 50
            elif "32b" in model_id_lower or "32-b" in model_id_lower:
                score -= 40
            elif "72b" in model_id_lower or "72-b" in model_id_lower:
                score -= 30
            elif "235b" in model_id_lower or "235-b" in model_id_lower:
                score -= 20
            elif "480b" in model_id_lower or "480-b" in model_id_lower:
                score -= 10
            
            # Prefer models with known good performance
            if "qwen2.5-7b" in model_id_lower:
                score += 5
            if "qwen2.5-coder" in model_id_lower:
                score += 3
            if "deepseek-v3.2" in model_id_lower:
                score += 2
            
            return score
        
        return sorted(models, key=lambda m: performance_score(m.get("id", "")))
    
    def build_crush_config(self, categorized_models):
        """Build comprehensive Crush configuration with performance optimizations"""
        
        # Build provider configurations for each category
        providers = {}
        
        # Chat Provider (Primary) - Optimized for speed
        if categorized_models["chat"]:
            providers["siliconflow-chat"] = {
                "name": "SiliconFlow Chat Models",
                "type": "openai-compat",
                "api_key": self.api_key,
                "base_url": f"{self.base_url}/chat/completions",
                "capabilities": ["chat", "reasoning", "tool-calling", "function-calling"],
                "parameters": self._get_chat_params(),
                "models": [
                    {
                        "id": model["id"],
                        "name": self._format_model_name(model["id"]),
                        "context_window": self._infer_context_window(model["id"]),
                        "default_max_tokens": 2048,  # Optimized: Lower default
                        "can_reason": self._supports_reasoning(model["id"]),
                        "supports_attachments": "vl" in model["id"].lower() or "vision" in model["id"].lower(),
                        "supports_tools": True,
                        "thinking_enabled": self._supports_thinking(model["id"]),
                        "max_thinking_budget": 2048,  # Optimized: Lower max
                        "capabilities": self._infer_capabilities(model["id"]),
                        "estimated_cost_per_1k": self._estimate_cost(model["id"]),
                        "performance_priority": idx + 1  # Lower number = higher priority
                    }
                    for idx, model in enumerate(categorized_models["chat"][:15])  # Optimized: Limit to 15
                ],
                "timeout": 30000,  # Optimized: 30s timeout
                "retry_attempts": 2,  # Optimized: Fewer retries
                "rate_limit": {"requests_per_minute": 30}  # Conservative rate limit
            }
        
        # Embedding Provider - Optimized with caching
        if categorized_models["embedding"]:
            providers["siliconflow-embedding"] = {
                "name": "SiliconFlow Embeddings",
                "type": "openai-compat",
                "api_key": self.api_key,
                "base_url": f"{self.base_url}/embeddings",
                "capabilities": ["text-embedding", "semantic-search", "vector-search"],
                "parameters": self._get_embedding_params(),
                "models": [
                    {
                        "id": model["id"],
                        "name": self._format_model_name(model["id"]),
                        "max_input_tokens": self._infer_embedding_tokens(model["id"]),
                        "dimensions": 768,  # Optimized: Lower default dimensions
                        "encoding_formats": ["float"],
                        "capabilities": ["embedding", "similarity-search"],
                        "batch_size": 32,  # Optimized: Batch processing
                        "cache_enabled": True,  # Optimized: Caching
                        "cache_ttl": 3600  # 1 hour cache
                    }
                    for model in categorized_models["embedding"][:3]  # Optimized: Limit to 3
                ],
                "timeout": 45000,  # Longer timeout for embeddings
                "cache_enabled": True,
                "prefer_local_cache": True
            }
        
        # Reranker Provider - Optimized with limits
        if categorized_models["reranker"]:
            providers["siliconflow-reranker"] = {
                "name": "SiliconFlow Rerankers",
                "type": "openai-compat",
                "api_key": self.api_key,
                "base_url": f"{self.base_url}/rerank",
                "capabilities": ["document-reranking", "relevance-scoring"],
                "parameters": self._get_rerank_params(),
                "models": [
                    {
                        "id": model["id"],
                        "name": self._format_model_name(model["id"]),
                        "max_documents": 50,  # Optimized: Lower limit
                        "return_documents": True,
                        "chunking_support": "bge-reranker" in model["id"],
                        "capabilities": ["reranking", "relevance-scoring"],
                        "default_top_n": 5,  # Optimized: Lower default
                        "score_threshold": 0.3  # Optimized: Lower threshold
                    }
                    for model in categorized_models["reranker"][:2]  # Optimized: Limit to 2
                ],
                "timeout": 60000,
                "document_limit": 50  # Optimized: Lower document limit
            }
        
        # Vision Provider - Optimized for batch
        if categorized_models["vision"]:
            providers["siliconflow-vision"] = {
                "name": "SiliconFlow Vision Language Models",
                "type": "openai-compat",
                "api_key": self.api_key,
                "base_url": f"{self.base_url}/chat/completions",
                "capabilities": ["vision", "image-understanding", "multimodal"],
                "parameters": {**self._get_chat_params(), **self._get_vision_params()},
                "models": [
                    {
                        "id": model["id"],
                        "name": self._format_model_name(model["id"]),
                        "context_window": 32768,  # Optimized: Lower default
                        "supports_images": True,
                        "max_image_size": "1024x1024",  # Optimized: Smaller default
                        "image_formats": ["png", "jpeg", "jpg"],
                        "capabilities": ["vision", "ocr", "image-description"],
                        "max_images_per_request": 3,  # Optimized: Lower limit
                        "image_detail": "low"  # Optimized: Lower detail by default
                    }
                    for model in categorized_models["vision"][:2]  # Optimized: Limit to 2
                ],
                "timeout": 45000,
                "image_compression": True
            }
        
        # Build complete Crush config with all optimizations
        config = {
            "$schema": "https://charm.land/crush.json",
            "providers": providers,
            "defaultProvider": "siliconflow-chat" if "siliconflow-chat" in providers else list(providers.keys())[0],
            "defaultModel": categorized_models["chat"][0]["id"] if categorized_models["chat"] else "",
            
            # Advanced sampling with ALL performance optimizations
            "sampling": {
                "temperature": 0.3,  # Optimized: Lower for determinism
                "top_p": 0.9,
                "top_k": 40,  # Optimized: Lower for speed
                "max_tokens": 2048,  # Optimized: Lower default
                "enable_thinking": False,  # Optimized: Disabled by default
                "thinking_budget": 1024,  # Optimized: Lower default
                "min_p": 0.05,
                "frequency_penalty": 0.1,  # Optimized: Lower penalty
                "presence_penalty": 0.0,
                "stop_sequences": [],
                "n": 1,
                "stream": True,  # Optimized: Streaming enabled
                "response_format": {"type": "text"},
                "best_of": 1  # Optimized: Disable best_of for speed
            },
            
            # Comprehensive features with performance settings
            "features": {
                "lsp_integration": {
                    "enabled": True,
                    "auto_completion": True,
                    "code_actions": True,
                    "diagnostics": True,
                    "hover": True,
                    "signature_help": True,
                    "definition": True,
                    "references": True,
                    "debounce_time": 300,  # Optimized: Longer debounce
                    "max_completion_items": 20  # Optimized: Limit items
                },
                "mcp_tools": {
                    "enabled": True,
                    "servers": [
                        {
                            "name": "filesystem",
                            "command": "npx",
                            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/media/milosvasic/DATA4TB/Projects"],
                            "timeout": 15000
                        },
                        {
                            "name": "sqlite",
                            "command": "npx",
                            "args": ["-y", "@modelcontextprotocol/server-sqlite"],
                            "timeout": 10000
                        }
                    ],
                    "max_concurrent_tools": 3,  # Optimized: Limit concurrent tools
                    "tool_timeout": 10000  # Optimized: 10s timeout
                },
                "session_memory": {
                    "enabled": True,
                    "max_messages": 20,  # Optimized: Lower limit
                    "compression": True
                },
                "session_caching": {
                    "enabled": True,
                    "ttl": 1800,  # Optimized: 30 minutes
                    "max_size": 100
                },
                "token_usage": {
                    "enabled": True,
                    "warn_threshold": 1000,
                    "block_threshold": 4000
                },
                "reasoning_traces": {
                    "enabled": False,  # Optimized: Disabled by default
                    "max_trace_length": 500
                },
                "tool_calling": {
                    "enabled": True,
                    "max_parallel_tools": 3,  # Optimized: Lower limit
                    "validation_timeout": 5000
                },
                "function_calling": {
                    "enabled": True,
                    "max_functions": 20,  # Optimized: Lower limit
                    "strict_schema": False,
                    "parallel_tool_use": True,
                    "timeout": 8000
                },
                "embeddings": {
                    "enabled": True,
                    "cache_enabled": True,
                    "batch_size": 16,  # Optimized: Smaller batches
                    "precompute": False
                },
                "reranking": {
                    "enabled": True,
                    "document_limit": 50,
                    "score_threshold": 0.3
                },
                "vision": {
                    "enabled": len(categorized_models["vision"]) > 0,
                    "max_image_size": "1024x1024",
                    "compression": True
                },
                "performance": {
                    "debounce_input": 500,
                    "throttle_output": 100,
                    "lazy_loading": True,
                    "incremental_updates": True
                }
            },
            
            # Optimized model routing based on task
            "model_routing": {
                "coding": "Qwen/Qwen2.5-Coder-32B-Instruct",
                "debugging": "deepseek-ai/DeepSeek-V3.2",
                "reasoning": "moonshotai/Kimi-K2-Instruct",
                "vision": "Qwen/Qwen2.5-VL-72B-Instruct" if categorized_models["vision"] else None,
                "chat": "Qwen/Qwen2.5-7B-Instruct",  # Optimized: Fast model for chat
                "quick": "Qwen/Qwen2.5-7B-Instruct",  # Optimized: Quick responses
                "embedding": categorized_models["embedding"][0]["id"] if categorized_models["embedding"] else None,
                "reranking": categorized_models["reranker"][0]["id"] if categorized_models["reranker"] else None,
                "default": "Qwen/Qwen2.5-7B-Instruct",  # Optimized: Fast default
                "fallback_chain": [  # Optimized: Performance-aware fallback
                    "Qwen/Qwen2.5-7B-Instruct",
                    "Qwen/Qwen2.5-Coder-32B-Instruct",
                    "Qwen/Qwen2.5-72B-Instruct"
                ]
            },
            
            # Performance-optimized limits
            "limits": {
                "max_input_tokens": 32000,  # Optimized: Lower limit
                "max_output_tokens": 2048,  # Optimized: Lower limit
                "max_context_window": 32768,  # Optimized: Lower limit
                "max_images_per_request": 3,  # Optimized: Lower limit
                "max_tools_per_request": 5,  # Optimized: Lower limit
                "max_embedding_tokens": 8192,  # Optimized: Lower limit
                "max_rerank_documents": 50,  # Optimized: Lower limit
                "max_concurrent_requests": 3,  # Optimized: Limit concurrency
                "request_timeout": 30000,  # Optimized: 30s timeout
                "rate_limit": {
                    "requests_per_minute": 30,
                    "tokens_per_minute": 40000
                }
            },
            
            # Optimized UI configuration
            "ui": {
                "theme": "dark",
                "show_model_info": True,
                "show_token_counts": True,
                "show_reasoning_traces": False,  # Optimized: Disabled by default
                "show_tool_calls": True,
                "log_level": "warn",  # Optimized: Lower log level
                "animation_enabled": False,  # Optimized: Disable animations
                "virtual_scrolling": True,  # Optimized: Enable virtual scroll
                "debounce_delay": 300  # Optimized: Input debounce
            },
            
            # Performance-optimized agent configurations
            "agents": {
                "coder": {
                    "system_prompt": "You are an expert coding assistant. Use reasoning only for complex problems. Prioritize clean, testable code with error handling.",
                    "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
                    "temperature": 0.2,  # Optimized: Lower temperature
                    "enable_thinking": False,  # Optimized: Disabled by default
                    "thinking_budget": 2048,  # Optimized: Lower budget
                    "max_tokens": 2048,
                    "timeout": 30000
                },
                "debugger": {
                    "system_prompt": "Debug systematically. Analyze errors, suggest fixes. Keep responses concise.",
                    "model": "Qwen/Qwen2.5-7B-Instruct",  # Optimized: Faster model
                    "temperature": 0.1,
                    "enable_thinking": True,
                    "thinking_budget": 1024,
                    "max_tokens": 1024
                },
                "researcher": {
                    "system_prompt": "Research and analyze information. Provide concise answers with key points.",
                    "model": "moonshotai/Kimi-K2-Instruct",
                    "temperature": 0.3,
                    "max_tokens": 1536,
                    "enable_thinking": False
                },
                "vision_analyst": {
                    "system_prompt": "Analyze images concisely. Focus on key elements and actionable insights.",
                    "model": "Qwen/Qwen2.5-VL-7B-Instruct" if any("7b" in m["id"].lower() for m in categorized_models.get("vision", [])) else 
                            (categorized_models["vision"][0]["id"] if categorized_models["vision"] else "Qwen/Qwen2.5-7B-Instruct"),
                    "temperature": 0.2,
                    "max_images": 2,  # Optimized: Limit images
                    "image_detail": "low"  # Optimized: Lower detail
                }
            },
            
            # Optimized tool configurations
            "tools": {
                "enabled": True,
                "auto_tool_choice": True,
                "max_parallel_tools": 3,  # Optimized: Lower limit
                "timeout": 10000,  # Optimized: 10s timeout
                "validation": {
                    "validate_arguments": True,
                    "allow_partial": True,  # Optimized: Allow partial
                    "strict_mode": False
                },
                "cache_results": True,  # Optimized: Cache tool results
                "cache_ttl": 300  # Optimized: 5 minute cache
            },
            
            # Performance-optimized embedding configurations
            "embeddings": {
                "enabled": True,
                "default_model": categorized_models["embedding"][0]["id"] if categorized_models["embedding"] else None,
                "encoding_format": "float",
                "dimensions": 768,  # Optimized: Lower dimensions
                "normalize": True,
                "batch_size": 16,  # Optimized: Smaller batches
                "cache_enabled": True,  # Optimized: Enable cache
                "cache_ttl": 3600,
                "precompute_frequent": True  # Optimized: Precompute
            },
            
            # Optimized reranking configurations
            "reranking": {
                "enabled": True,
                "default_model": categorized_models["reranker"][0]["id"] if categorized_models["reranker"] else None,
                "top_n": 5,  # Optimized: Lower default
                "return_documents": True,
                "score_threshold": 0.3,  # Optimized: Lower threshold
                "document_limit": 50,
                "chunk_size": 500  # Optimized: Smaller chunks
            },
            
            # Performance-optimized options
            "options": {
                "debug": False,  # Optimized: Disable debug
                "debug_lsp": False,
                "auto_save_session": True,
                "auto_update_models": False,  # Optimized: Disable auto-update
                "cache_embeddings": True,  # Optimized: Enable cache
                "prefer_native_tools": True,
                "fallback_to_local": True,
                "timeout": 30000,  # Optimized: 30s timeout
                "retry_attempts": 2,  # Optimized: Fewer retries
                "retry_delay": 1000,
                "compression": True,  # Optimized: Enable compression
                "lazy_loading": True,  # Optimized: Lazy load
                "incremental_updates": True  # Optimized: Incremental UI updates
            },
            
            # Performance monitoring
            "performance_monitoring": {
                "enabled": True,
                "sampling_rate": 0.1,  # Optimized: 10% sampling
                "metrics": ["response_time", "token_usage", "cache_hit_rate"],
                "alert_thresholds": {
                    "response_time_ms": 10000,
                    "error_rate": 0.05,
                    "cache_miss_rate": 0.3
                }
            }
        }
        
        return config
    
    def build_opencode_config(self, categorized_models):
        """Build comprehensive OpenCode configuration with performance optimizations"""
        
        # Build models object for OpenCode
        models = {}
        for category, model_list in categorized_models.items():
            for idx, model in enumerate(model_list[:8]):  # Optimized: Limit to 8 per category
                model_id = model["id"]
                models[model_id] = {
                    "name": self._format_model_name(model_id),
                    "category": category,
                    "capabilities": self._infer_capabilities(model_id),
                    "contextWindow": self._infer_context_window(model_id),
                    "supportsFunctionCalling": category in ["chat", "vision"],
                    "supportsVision": category in ["vision", "image"],
                    "supportsAudio": category == "audio",
                    "supportsVideo": category == "video",
                    "maxTokens": 2048 if category == "chat" else None,  # Optimized: Lower default
                    "recommendedFor": self._recommended_for(category, model_id),
                    "performancePriority": idx + 1,
                    "estimatedCost": self._estimate_cost(model_id)
                }
        
        # Find optimal default models
        default_chat_model = next((m for m in categorized_models.get("chat", []) 
                                 if "7b" in m["id"].lower()), 
                                categorized_models["chat"][0]["id"] if categorized_models["chat"] else "")
        
        config = {
            "$schema": "https://opencode.ai/config.json",
            "theme": "dark",
            "model": default_chat_model,  # Optimized: Fast model by default
            "small_model": "Qwen/Qwen2.5-7B-Instruct",
            "coding_model": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "reasoning_model": "moonshotai/Kimi-K2-Instruct",
            "vision_model": categorized_models["vision"][0]["id"] if categorized_models["vision"] else None,
            "embedding_model": categorized_models["embedding"][0]["id"] if categorized_models["embedding"] else None,
            "reranker_model": categorized_models["reranker"][0]["id"] if categorized_models["reranker"] else None,
            
            # Comprehensive provider configuration with performance settings
            "provider": {
                "siliconflow": {
                    "name": "SiliconFlow Full Stack",
                    "npm": "@ai-sdk/openai-compatible",
                    "models": models,
                    "options": {
                        "apiKey": self.api_key,
                        "baseURL": self.base_url,
                        "defaultHeaders": {
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.api_key}"
                        },
                        "timeout": 30000,  # Optimized: 30s timeout
                        "maxRetries": 2,  # Optimized: Fewer retries
                        "retryDelay": 1000,
                        "enableStreaming": True,  # Optimized: Enable streaming
                        "compression": True  # Optimized: Enable compression
                    },
                    "endpoints": {
                        "chat": "/chat/completions",
                        "embeddings": "/embeddings",
                        "rerank": "/rerank",
                        "images": "/images/generations",
                        "audio": "/audio/transcriptions"
                    },
                    "performance": {
                        "connectionPool": 3,
                        "keepAlive": True,
                        "timeToLive": 30000
                    }
                }
            },
            
            # Performance-optimized instructions
            "instructions": [
                "You are a performance-optimized AI assistant. Keep responses concise and focused.",
                "Use smaller models for simple tasks, larger models only when necessary.",
                "Enable thinking mode only for complex reasoning problems.",
                "Use streaming for responses longer than 200 tokens.",
                "Cache embeddings and reuse when possible.",
                "For coding: write efficient, clean code with minimal comments.",
                "Use tools only when they provide clear value."
            ],
            
            # Performance-optimized tool integrations
            "tools": {
                "fileSystem": {
                    "enabled": True,
                    "cacheFiles": True,  # Optimized: Cache files
                    "maxFileSize": 1048576  # 1MB limit
                },
                "terminal": {
                    "enabled": True,
                    "timeout": 10000,
                    "maxOutput": 4096  # Limit output
                },
                "git": True,
                "browser": False,  # Optimized: Disable heavy browser
                "mcp": {
                    "enabled": True,
                    "servers": [
                        {
                            "name": "filesystem",
                            "command": "npx",
                            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/media/milosvasic/DATA4TB/Projects"],
                            "timeout": 15000,
                            "cache": True
                        },
                        {
                            "name": "sqlite",
                            "command": "npx",
                            "args": ["-y", "@modelcontextprotocol/server-sqlite"],
                            "timeout": 10000
                        }
                    ],
                    "maxConcurrent": 2,  # Optimized: Limit concurrency
                    "requestTimeout": 10000
                },
                "embeddings": {
                    "enabled": True,
                    "provider": "siliconflow",
                    "model": categorized_models["embedding"][0]["id"] if categorized_models["embedding"] else None,
                    "autoEmbed": False,  # Optimized: Disable auto-embed
                    "cache": True,
                    "batchSize": 16
                },
                "reranking": {
                    "enabled": True,
                    "provider": "siliconflow",
                    "model": categorized_models["reranker"][0]["id"] if categorized_models["reranker"] else None,
                    "topN": 5,
                    "threshold": 0.3
                }
            },
            
            # Optimized API parameters
            "parameters": {
                "chat": self._get_chat_params(),
                "embedding": self._get_embedding_params(),
                "rerank": self._get_rerank_params()
            },
            
            # Performance-optimized completion configuration
            "completion": {
                "enabled": True,
                "provider": "siliconflow",
                "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
                "temperature": 0.1,  # Optimized: Very low for code
                "maxTokens": 256,  # Optimized: Shorter completions
                "enable_thinking": False,  # Optimized: Disabled
                "thinking_budget": 512,
                "stream": True,  # Optimized: Enable streaming
                "debounce": 300  # Optimized: Input debounce
            },
            
            # Optimized chat configuration
            "chat": {
                "enabled": True,
                "provider": "siliconflow",
                "defaultModel": "Qwen/Qwen2.5-7B-Instruct",  # Optimized: Fast default
                "temperature": 0.3,  # Optimized: Lower temperature
                "maxTokens": 1536,  # Optimized: Lower limit
                "stream": True,  # Optimized: Enable streaming
                "enable_thinking": False,  # Optimized: Disabled by default
                "thinking_budget": 1024,  # Optimized: Lower budget
                "tools": True,
                "tool_choice": "auto",
                "timeout": 30000
            },
            
            # Performance-optimized session management
            "session": {
                "memoryEnabled": True,
                "maxMessages": 20,  # Optimized: Lower limit
                "autoSummarize": True,
                "summaryLength": 500,  # Optimized: Shorter summaries
                "embedMessages": False,  # Optimized: Disable embedding
                "similarityThreshold": 0.8,
                "compression": True  # Optimized: Enable compression
            },
            
            # Optimized UI configuration
            "ui": {
                "showTokenCounts": True,
                "showReasoningTraces": False,  # Optimized: Disabled
                "showToolCalls": True,
                "showEmbeddingVectors": False,  # Optimized: Disabled
                "showRerankScores": False,  # Optimized: Disabled
                "autoFormatCode": True,
                "theme": "dark",
                "fontSize": 13,  # Optimized: Slightly smaller
                "lineHeight": 1.5,
                "animationDuration": 100,  # Optimized: Faster animations
                "virtualScroll": True  # Optimized: Virtual scrolling
            },
            
            # Keybindings
            "keybindings": {
                "toggleChat": "ctrl+shift+c",
                "quickCompletion": "ctrl+space",
                "explainCode": "ctrl+shift+e",
                "refactorCode": "ctrl+shift+r",
                "debugCode": "ctrl+shift+d",
                "visionAnalysis": "ctrl+shift+v"
            },
            
            # Performance-optimized features
            "features": {
                "autoModelSelection": True,
                "contextAwareRouting": True,
                "intentRecognition": True,
                "costOptimization": {
                    "enabled": True,
                    "preferSmallModels": True,
                    "cacheStrategies": True,
                    "budgetLimit": 1000
                },
                "fallbackStrategies": {
                    "enabled": True,
                    "chain": ["7B", "32B", "72B"],
                    "timeout": 5000
                },
                "performanceMonitoring": {
                    "enabled": True,
                    "metrics": ["latency", "tokens", "cache"],
                    "sampling": 0.1
                },
                "usageAnalytics": {
                    "enabled": True,
                    "anonymize": True,
                    "batchSize": 10
                }
            },
            
            # Performance-optimized limits
            "limits": {
                "maxContextTokens": 32768,  # Optimized: Lower limit
                "maxCompletionTokens": 2048,  # Optimized: Lower limit
                "maxEmbeddingTokens": 8192,  # Optimized: Lower limit
                "maxRerankDocuments": 50,  # Optimized: Lower limit
                "maxImagesPerRequest": 3,  # Optimized: Lower limit
                "maxToolsPerRequest": 5,  # Optimized: Lower limit
                "rateLimit": {
                    "requestsPerMinute": 30,  # Optimized: Lower limit
                    "tokensPerMinute": 40000
                },
                "timeouts": {
                    "request": 30000,
                    "tool": 10000,
                    "connection": 5000
                }
            },
            
            # Performance-optimized cache configuration
            "cache": {
                "enabled": True,
                "ttl": 1800,  # Optimized: 30 minutes
                "maxSize": 500,  # Optimized: Lower cache size
                "embeddingCache": {
                    "enabled": True,
                    "ttl": 3600,
                    "maxSize": 1000
                },
                "completionCache": {
                    "enabled": True,
                    "ttl": 300,
                    "maxSize": 200
                },
                "modelResponseCache": {
                    "enabled": True,
                    "ttl": 600,
                    "maxSize": 100
                },
                "compression": True  # Optimized: Cache compression
            },
            
            # Performance settings
            "performance": {
                "debounceInput": 300,
                "throttleOutput": 50,
                "lazyLoadComponents": True,
                "incrementalRendering": True,
                "workerThreads": 2,
                "memoryManagement": {
                    "gcInterval": 30000,
                    "maxHeap": 256
                }
            }
        }
        
        return config
    
    # Helper methods
    def _format_model_name(self, model_id):
        """Format model ID for display"""
        parts = model_id.split("/")
        if len(parts) > 1:
            return f"{parts[0]} {parts[1]}"
        return model_id
    
    def _infer_context_window(self, model_id):
        """Infer context window from model name"""
        model_id_lower = model_id.lower()
        if "128k" in model_id_lower or "200000" in model_id:
            return 131072  # Optimized: Cap at 128k
        elif "32k" in model_id_lower:
            return 32768
        elif "16k" in model_id_lower:
            return 16384
        elif "8k" in model_id_lower:
            return 8192
        elif "qwen3" in model_id_lower:
            return 32768  # Optimized: Lower default for Qwen3
        elif "qwen2.5" in model_id_lower:
            return 16384  # Optimized: Lower default for Qwen2.5
        else:
            return 4096
    
    def _supports_reasoning(self, model_id):
        """Check if model supports reasoning"""
        return any(keyword in model_id.lower() for keyword in 
                  ["r1", "thinking", "k2", "reason", "deepseek-r1"])
    
    def _supports_thinking(self, model_id):
        """Check if model supports thinking mode"""
        return any(keyword in model_id.lower() for keyword in 
                  ["qwen3", "hunyuan", "glm-4.6v", "glm-4.5v", "deepseek-v3"])
    
    def _infer_capabilities(self, model_id):
        """Infer model capabilities"""
        caps = []
        model_id_lower = model_id.lower()
        
        if "coder" in model_id_lower or "code" in model_id_lower:
            caps.append("coding")
        if "vl" in model_id_lower or "vision" in model_id_lower:
            caps.append("vision")
        if "embedding" in model_id_lower:
            caps.append("embedding")
        if "reranker" in model_id_lower:
            caps.append("reranking")
        if any(keyword in model_id_lower for keyword in ["r1", "thinking", "reason"]):
            caps.append("reasoning")
        if "chat" in model_id_lower or "instruct" in model_id_lower:
            caps.append("chat")
        
        return caps if caps else ["general"]
    
    def _infer_embedding_tokens(self, model_id):
        """Infer max embedding tokens"""
        if "qwen3-embedding-8b" in model_id.lower():
            return 16384  # Optimized: Lower limit
        elif "bge-large" in model_id.lower():
            return 512
        elif "bge-m3" in model_id.lower():
            return 4096  # Optimized: Lower limit
        else:
            return 2048
    
    def _estimate_cost(self, model_id):
        """Estimate cost per 1K tokens"""
        model_id_lower = model_id.lower()
        
        if "0.6b" in model_id_lower:
            return 0.01
        elif "4b" in model_id_lower:
            return 0.02
        elif "7b" in model_id_lower or "8b" in model_id_lower:
            return 0.03
        elif "14b" in model_id_lower:
            return 0.05
        elif "32b" in model_id_lower:
            return 0.08
        elif "72b" in model_id_lower:
            return 0.15
        elif "235b" in model_id_lower:
            return 0.30
        else:
            return 0.05  # Average
    
    def _recommended_for(self, category, model_id):
        """Get recommended use cases"""
        recommendations = {
            "chat": ["quick-chat", "simple-qa", "summarization"],
            "embedding": ["semantic-search", "clustering"],
            "reranker": ["document-ranking", "relevance"],
            "vision": ["image-analysis", "ocr"],
            "image": ["image-generation"],
            "audio": ["speech-recognition"],
            "video": ["video-generation"]
        }
        
        # Add specific recommendations
        specific = []
        if "coder" in model_id.lower():
            specific.append("coding")
        if "instruct" in model_id.lower():
            specific.append("instructions")
        if "reason" in model_id.lower() or "thinking" in model_id.lower():
            specific.append("complex-reasoning")
        if "7b" in model_id.lower():
            specific.append("fast-response")
        
        return recommendations.get(category, []) + specific
    
    def _get_vision_params(self):
        """Get vision-specific parameters"""
        return {
            "max_images": {"type": "integer", "default": 2},  # Optimized: Lower default
            "image_detail": {"type": "string", "options": ["low", "high", "auto"], "default": "low"},  # Optimized: Low detail
            "image_format": {"type": "string", "options": ["url", "base64"], "default": "url"}
        }

def main():
    print("🚀 SiliconFlow Performance-Optimized Configuration Builder")
    print("=" * 70)
    print("Optimizations applied:")
    print("• Lower default tokens (2048 instead of 4096)")
    print("• Smaller default models (7B instead of 72B)")
    print("• Caching enabled for embeddings and completions")
    print("• Conservative thinking budgets (1024 instead of 4096)")
    print("• Lower temperatures for determinism")
    print("• Streaming enabled by default")
    print("• Batch processing for embeddings")
    print("• Timeout optimizations (30s instead of 60s)")
    print("• Cost-aware model selection with fallback chains")
    print("=" * 70)
    
    # Get API key
    api_key = getpass.getpass("Enter your SiliconFlow API Key: ")
    
    # Initialize builder
    builder = SiliconFlowConfigBuilder(api_key)
    
    # Fetch all models
    categorized_models = builder.fetch_all_models()
    
    # Print summary
    print("\n📊 Model Summary (Performance Sorted):")
    for category, models in categorized_models.items():
        if models:
            print(f"  {category.capitalize()}: {len(models)} models")
            for model in models[:2]:  # Show first 2
                print(f"    - {model['id']} (Priority: {models.index(model) + 1})")
    
    # Build configurations
    print("\n⚙️ Building optimized configurations...")
    
    crush_config = builder.build_crush_config(categorized_models)
    opencode_config = builder.build_opencode_config(categorized_models)
    
    # Create backup directory
    backup_dir = Path.home() / ".config" / "siliconflow_backup"
    backup_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save configurations
    crush_path = Path.home() / ".config" / "crush" / "crush.json"
    opencode_path = Path.home() / ".config" / "opencode" / "config.json"
    
    # Create directories
    crush_path.parent.mkdir(parents=True, exist_ok=True)
    opencode_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Backup existing
    if crush_path.exists():
        backup_crush = backup_dir / f"crush_backup_{timestamp}.json"
        with open(crush_path, "r") as src, open(backup_crush, "w") as dst:
            dst.write(src.read())
        print(f"✅ Crush config backed up to: {backup_crush}")
    
    if opencode_path.exists():
        backup_opencode = backup_dir / f"opencode_backup_{timestamp}.json"
        with open(opencode_path, "r") as src, open(backup_opencode, "w") as dst:
            dst.write(src.read())
        print(f"✅ OpenCode config backed up to: {backup_opencode}")
    
    # Write new configs
    with open(crush_path, "w") as f:
        json.dump(crush_config, f, indent=2, ensure_ascii=False)
    
    with open(opencode_path, "w") as f:
        json.dump(opencode_config, f, indent=2, ensure_ascii=False)
    
    # Set secure permissions
    os.chmod(crush_path, 0o600)
    os.chmod(opencode_path, 0o600)
    
    print(f"\n✅ Performance-optimized configurations saved:")
    print(f"   Crush: {crush_path}")
    print(f"   OpenCode: {opencode_path}")
    
    # Create update script
    update_script = Path.home() / ".config" / "update_siliconflow_models.py"
    with open(update_script, "w") as f:
        f.write('''#!/usr/bin/env python3
"""
Auto-update SiliconFlow models with performance optimizations
Run this periodically to keep models updated
"""
import sys
import json
from pathlib import Path
import getpass

def extract_api_key_from_config(config_path):
    """Extract API key from existing config"""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            
        # Try Crush format
        if "providers" in config:
            for provider in config.get("providers", {}).values():
                if isinstance(provider, dict) and "api_key" in provider:
                    return provider["api_key"]
        
        # Try OpenCode format
        if "provider" in config:
            provider = config.get("provider", {})
            if "siliconflow" in provider:
                options = provider["siliconflow"].get("options", {})
                if "apiKey" in options:
                    return options["apiKey"]
        
        return None
    except:
        return None

def update_models():
    print("🔄 Checking for SiliconFlow model updates...")
    
    # Try to get API key from existing configs
    api_key = None
    for config_path in [
        Path.home() / ".config" / "crush" / "crush.json",
        Path.home() / ".config" / "opencode" / "config.json"
    ]:
        if config_path.exists():
            api_key = extract_api_key_from_config(config_path)
            if api_key:
                print(f"✅ Found API key in {config_path.name}")
                break
    
    if not api_key:
        api_key = getpass.getpass("Enter your SiliconFlow API Key: ")
    
    # Import the builder (make sure it's in the same directory)
    try:
        from siliconflow_builder import SiliconFlowConfigBuilder
    except ImportError:
        print("❌ Error: siliconflow_builder.py must be in the same directory")
        return
    
    builder = SiliconFlowConfigBuilder(api_key)
    
    try:
        categorized_models = builder.fetch_all_models()
    except Exception as e:
        print(f"❌ Failed to fetch models: {e}")
        return
    
    # Update configurations
    try:
        crush_config = builder.build_crush_config(categorized_models)
        opencode_config = builder.build_opencode_config(categorized_models)
        
        # Save
        with open(Path.home() / ".config" / "crush" / "crush.json", "w") as f:
            json.dump(crush_config, f, indent=2)
        
        with open(Path.home() / ".config" / "opencode" / "config.json", "w") as f:
            json.dump(opencode_config, f, indent=2)
        
        print(f"✅ Updated {len(categorized_models.get('chat', []))} chat models")
        print(f"✅ Updated {len(categorized_models.get('embedding', []))} embedding models")
        print(f"✅ Updated {len(categorized_models.get('reranker', []))} reranker models")
        
        # Print performance tips
        print("\\n💡 Performance Tips:")
        print("  • Default model is now Qwen2.5-7B-Instruct (fastest)")
        print("  • Thinking mode disabled by default (enable for complex tasks)")
        print("  • Embedding cache enabled (1 hour TTL)")
        print("  • Response tokens limited to 2048 by default")
        
    except Exception as e:
        print(f"❌ Failed to update configurations: {e}")

if __name__ == "__main__":
    update_models()
''')
    
    os.chmod(update_script, 0o755)
    
    # Create performance monitoring script
    perf_script = Path.home() / ".config" / "check_siliconflow_perf.py"
    with open(perf_script, "w") as f:
        f.write('''#!/usr/bin/env python3
"""
SiliconFlow Performance Monitor
Checks configuration and suggests optimizations
"""
import json
from pathlib import Path

def check_crush_config():
    config_path = Path.home() / ".config" / "crush" / "crush.json"
    if not config_path.exists():
        return "❌ Crush config not found"
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    optimizations = []
    
    # Check sampling settings
    sampling = config.get("sampling", {})
    if sampling.get("max_tokens", 0) > 4096:
        optimizations.append("Consider lowering max_tokens from {sampling['max_tokens']} to 2048")
    if sampling.get("temperature", 1.0) > 0.5:
        optimizations.append(f"Consider lowering temperature from {sampling.get('temperature')} to 0.3")
    if sampling.get("thinking_budget", 0) > 4096:
        optimizations.append(f"Consider lowering thinking_budget from {sampling.get('thinking_budget')} to 1024")
    
    # Check model routing
    routing = config.get("model_routing", {})
    if "7b" not in routing.get("default", "").lower():
        optimizations.append("Consider setting default model to a 7B model for speed")
    
    # Check cache settings
    if not config.get("embeddings", {}).get("cache_enabled", False):
        optimizations.append("Enable embedding cache for better performance")
    
    return optimizations if optimizations else ["✅ Crush config is performance-optimized"]

def check_opencode_config():
    config_path = Path.home() / ".config" / "opencode" / "config.json"
    if not config_path.exists():
        return "❌ OpenCode config not found"
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    optimizations = []
    
    # Check chat settings
    chat = config.get("chat", {})
    if chat.get("maxTokens", 0) > 4096:
        optimizations.append(f"Consider lowering maxTokens from {chat.get('maxTokens')} to 1536")
    if chat.get("temperature", 1.0) > 0.5:
        optimizations.append(f"Consider lowering temperature from {chat.get('temperature')} to 0.3")
    
    # Check session settings
    session = config.get("session", {})
    if session.get("maxMessages", 0) > 50:
        optimizations.append(f"Consider lowering maxMessages from {session.get('maxMessages')} to 20")
    
    # Check cache
    cache = config.get("cache", {})
    if not cache.get("enabled", False):
        optimizations.append("Enable caching for better performance")
    
    return optimizations if optimizations else ["✅ OpenCode config is performance-optimized"]

def main():
    print("🔍 SiliconFlow Performance Check")
    print("=" * 50)
    
    print("\\nChecking Crush configuration...")
    crush_optimizations = check_crush_config()
    for opt in crush_optimizations:
        print(f"  {opt}")
    
    print("\\nChecking OpenCode configuration...")
    opencode_optimizations = check_opencode_config()
    for opt in opencode_optimizations:
        print(f"  {opt}")
    
    print("\\n📊 Performance Recommendations:")
    print("  1. Use Qwen2.5-7B-Instruct for simple tasks")
    print("  2. Enable streaming for responses > 200 tokens")
    print("  3. Cache embeddings with 1-hour TTL")
    print("  4. Limit tool calls to 3 concurrent")
    print("  5. Set timeout to 30s for quick fallbacks")

if __name__ == "__main__":
    main()
''')
    
    os.chmod(perf_script, 0o755)
    
    print(f"\n📝 Performance scripts created:")
    print(f"   Update script: {update_script}")
    print(f"   Performance monitor: {perf_script}")
    
    print("\n🎯 Performance Optimizations Applied:")
    print("   1. Default model: Qwen2.5-7B-Instruct (fastest)")
    print("   2. Max tokens: 2048 (reduced from 4096)")
    print("   3. Temperature: 0.3 (more deterministic)")
    print("   4. Thinking budget: 1024 (reduced from 4096)")
    print("   5. Embedding cache: Enabled (1 hour TTL)")
    print("   6. Streaming: Enabled by default")
    print("   7. Timeout: 30s (faster fallbacks)")
    print("   8. Batch size: 16 (optimal for embeddings)")
    print("   9. Concurrent tools: 3 (reduced from 5)")
    print("   10. UI debounce: 300ms (smoother typing)")
    
    print("\n💡 Usage Tips:")
    print("   1. Run 'python ~/.config/update_siliconflow_models.py' monthly")
    print("   2. Run 'python ~/.config/check_siliconflow_perf.py' weekly")
    print("   3. Use 7B models for simple tasks, 32B for coding, 72B for complex analysis")
    print("   4. Enable thinking mode only when needed (adds latency)")
    print("   5. Cache frequently used embeddings")
    
    print("\n🚀 Restart Crush and OpenCode to apply optimizations!")

if __name__ == "__main__":
    main()