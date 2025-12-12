#!/usr/bin/env python3
"""
Dynamic SiliconFlow Model Discovery System
Automatically discovers and categorizes all available models from SiliconFlow API
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from siliconflow_client import SiliconFlowAPIClient, ModelCapabilities

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelCategory:
    """Model category with associated models"""
    name: str
    description: str
    models: List[ModelCapabilities]
    default_model: Optional[str] = None


@dataclass
class ModelRegistry:
    """Complete model registry with all categories"""
    chat_models: ModelCategory
    vision_models: ModelCategory
    audio_models: ModelCategory
    video_models: ModelCategory
    embedding_models: ModelCategory
    rerank_models: ModelCategory
    last_updated: str
    api_version: str = "v1"


class SiliconFlowModelDiscovery:
    """Dynamic model discovery and categorization system"""

    def __init__(self, api_key: str, cache_dir: Optional[Path] = None):
        self.client = SiliconFlowAPIClient(api_key, cache_dir)
        self.cache_dir = cache_dir or Path.home() / ".cache" / "siliconflow"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def discover_all_models(self) -> ModelRegistry:
        """Discover and categorize all available models"""
        logger.info("Starting comprehensive model discovery...")

        try:
            # Fetch all models from API
            all_models = self.client.get_models()
            logger.info(f"Discovered {len(all_models)} total models")

            # Categorize models
            categories = self._categorize_models(all_models)

            # Create registry
            registry = ModelRegistry(
                chat_models=categories['chat'],
                vision_models=categories['vision'],
                audio_models=categories['audio'],
                video_models=categories['video'],
                embedding_models=categories['embedding'],
                rerank_models=categories['rerank'],
                last_updated=self._get_current_timestamp(),
                api_version="v1"
            )

            # Cache the registry
            self._cache_registry(registry)

            logger.info("Model discovery completed successfully")
            return registry

        except Exception as e:
            logger.error(f"Model discovery failed: {e}")
            # Try to load from cache as fallback
            cached = self._load_cached_registry()
            if cached:
                logger.info("Using cached model registry as fallback")
                return cached
            raise

    def _categorize_models(self, models: List[ModelCapabilities]) -> Dict[str, ModelCategory]:
        """Categorize models into different types"""
        categories = {
            'chat': ModelCategory(
                name="Chat Models",
                description="Models for text-based conversational AI",
                models=[]
            ),
            'vision': ModelCategory(
                name="Vision Models",
                description="Models that can process images and visual content",
                models=[]
            ),
            'audio': ModelCategory(
                name="Audio Models",
                description="Models for speech synthesis and audio processing",
                models=[]
            ),
            'video': ModelCategory(
                name="Video Models",
                description="Models for video generation and processing",
                models=[]
            ),
            'embedding': ModelCategory(
                name="Embedding Models",
                description="Models for creating text embeddings",
                models=[]
            ),
            'rerank': ModelCategory(
                name="Rerank Models",
                description="Models for document reranking and relevance scoring",
                models=[]
            )
        }

        for model in models:
            if model.supports_chat:
                categories['chat'].models.append(model)
            if model.supports_vision:
                categories['vision'].models.append(model)
            if model.supports_audio:
                categories['audio'].models.append(model)
            if model.supports_video:
                categories['video'].models.append(model)
            if model.supports_embeddings:
                categories['embedding'].models.append(model)
            if model.supports_rerank:
                categories['rerank'].models.append(model)

        # Set default models for each category
        self._set_default_models(categories)

        return categories

    def _set_default_models(self, categories: Dict[str, ModelCategory]):
        """Set default models for each category based on priority"""
        # Chat models priority
        chat_priority = [
            'Qwen/Qwen2.5-14B-Instruct',
            'Qwen/Qwen2.5-7B-Instruct',
            'deepseek-ai/DeepSeek-V3',
            'Qwen/Qwen3-8B'
        ]
        categories['chat'].default_model = self._find_first_available(chat_priority, categories['chat'].models)

        # Vision models priority
        vision_priority = [
            'Qwen/Qwen2.5-VL-7B-Instruct',
            'Qwen/Qwen2.5-VL-32B-Instruct',
            'deepseek-ai/deepseek-vl2'
        ]
        categories['vision'].default_model = self._find_first_available(vision_priority, categories['vision'].models)

        # Audio models priority
        audio_priority = [
            'FunAudioLLM/CosyVoice2-0.5B',
            'IndexTeam/IndexTTS-2'
        ]
        categories['audio'].default_model = self._find_first_available(audio_priority, categories['audio'].models)

        # Video models priority
        video_priority = [
            'Wan-AI/Wan2.2-T2V-A14B',
            'Wan-AI/Wan2.2-I2V-A14B'
        ]
        categories['video'].default_model = self._find_first_available(video_priority, categories['video'].models)

        # Embedding models - usually just one
        if categories['embedding'].models:
            categories['embedding'].default_model = categories['embedding'].models[0].id

        # Rerank models - usually just one
        if categories['rerank'].models:
            categories['rerank'].default_model = categories['rerank'].models[0].id

    def _find_first_available(self, priority_list: List[str], available_models: List[ModelCapabilities]) -> Optional[str]:
        """Find first available model from priority list"""
        available_ids = {model.id for model in available_models}
        for model_id in priority_list:
            if model_id in available_ids:
                return model_id
        # If none from priority list, return first available
        return available_models[0].id if available_models else None

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.now().isoformat()

    def _cache_registry(self, registry: ModelRegistry):
        """Cache the model registry"""
        cache_file = self.cache_dir / "model_registry.json"
        try:
            with open(cache_file, 'w') as f:
                # Convert dataclasses to dicts for JSON serialization
                registry_dict = asdict(registry)
                json.dump(registry_dict, f, indent=2)
            logger.info(f"Cached model registry to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache registry: {e}")

    def _load_cached_registry(self) -> Optional[ModelRegistry]:
        """Load cached model registry"""
        cache_file = self.cache_dir / "model_registry.json"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)

            # Convert back to ModelRegistry
            categories = {}
            for cat_name in ['chat_models', 'vision_models', 'audio_models', 'video_models', 'embedding_models', 'rerank_models']:
                cat_data = data[cat_name]
                models = [ModelCapabilities(**m) for m in cat_data['models']]
                categories[cat_name] = ModelCategory(
                    name=cat_data['name'],
                    description=cat_data['description'],
                    models=models,
                    default_model=cat_data.get('default_model')
                )

            return ModelRegistry(
                chat_models=categories['chat_models'],
                vision_models=categories['vision_models'],
                audio_models=categories['audio_models'],
                video_models=categories['video_models'],
                embedding_models=categories['embedding_models'],
                rerank_models=categories['rerank_models'],
                last_updated=data['last_updated'],
                api_version=data.get('api_version', 'v1')
            )

        except Exception as e:
            logger.warning(f"Failed to load cached registry: {e}")
            return None

    def get_model_info(self, model_id: str) -> Optional[ModelCapabilities]:
        """Get detailed information about a specific model"""
        registry = self.discover_all_models()

        # Search through all categories
        all_models = (
            registry.chat_models.models +
            registry.vision_models.models +
            registry.audio_models.models +
            registry.video_models.models +
            registry.embedding_models.models +
            registry.rerank_models.models
        )

        for model in all_models:
            if model.id == model_id:
                return model

        return None

    def validate_model_compatibility(self, model_id: str, required_capabilities: List[str]) -> bool:
        """Validate if a model has the required capabilities"""
        model_info = self.get_model_info(model_id)
        if not model_info:
            return False

        capability_map = {
            'chat': model_info.supports_chat,
            'vision': model_info.supports_vision,
            'audio': model_info.supports_audio,
            'video': model_info.supports_video,
            'embedding': model_info.supports_embeddings,
            'rerank': model_info.supports_rerank,
            'function_calling': model_info.supports_function_calling,
            'reasoning': model_info.supports_reasoning
        }

        return all(capability_map.get(cap, False) for cap in required_capabilities)

    def get_models_by_capability(self, capability: str) -> List[ModelCapabilities]:
        """Get all models that support a specific capability"""
        registry = self.discover_all_models()
        all_models = (
            registry.chat_models.models +
            registry.vision_models.models +
            registry.audio_models.models +
            registry.video_models.models +
            registry.embedding_models.models +
            registry.rerank_models.models
        )

        capability_map = {
            'chat': lambda m: m.supports_chat,
            'vision': lambda m: m.supports_vision,
            'audio': lambda m: m.supports_audio,
            'video': lambda m: m.supports_video,
            'embedding': lambda m: m.supports_embeddings,
            'rerank': lambda m: m.supports_rerank,
            'function_calling': lambda m: m.supports_function_calling,
            'reasoning': lambda m: m.supports_reasoning
        }

        filter_func = capability_map.get(capability)
        if not filter_func:
            return []

        return [model for model in all_models if filter_func(model)]

    def get_category_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all model categories"""
        registry = self.discover_all_models()

        summary = {}
        for cat_name, category in [
            ('chat', registry.chat_models),
            ('vision', registry.vision_models),
            ('audio', registry.audio_models),
            ('video', registry.video_models),
            ('embedding', registry.embedding_models),
            ('rerank', registry.rerank_models)
        ]:
            summary[cat_name] = {
                'name': category.name,
                'description': category.description,
                'model_count': len(category.models),
                'default_model': category.default_model,
                'models': [model.id for model in category.models]
            }

        return summary
