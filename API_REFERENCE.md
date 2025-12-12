# SiliconFlow API Reference

This document provides comprehensive documentation for the SiliconFlow Toolkit API client and all supported endpoints.

## Table of Contents

- [SiliconFlowAPIClient](#siliconflowapiclient)
- [Model Discovery](#model-discovery)
- [Configuration Builder](#configuration-builder)
- [API Endpoints](#api-endpoints)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Examples](#examples)

## SiliconFlowAPIClient

The main API client class that provides access to all SiliconFlow endpoints.

### Initialization

```python
from siliconflow_client import SiliconFlowAPIClient

client = SiliconFlowAPIClient(
    api_key="your-api-key",
    cache_dir=None  # Optional: custom cache directory
)
```

### Parameters

- `api_key` (str): Your SiliconFlow API key
- `cache_dir` (Path, optional): Directory for caching API responses

## Model Discovery

The `SiliconFlowModelDiscovery` class automatically discovers and categorizes all available models.

### Usage

```python
from model_discovery import SiliconFlowModelDiscovery

discovery = SiliconFlowModelDiscovery("your-api-key")
registry = discovery.discover_all_models()
```

### Methods

#### `discover_all_models() -> ModelRegistry`

Discovers and categorizes all available models from the SiliconFlow API.

**Returns**: `ModelRegistry` object containing categorized models

#### `get_model_info(model_id: str) -> Optional[ModelCapabilities]`

Get detailed information about a specific model.

**Parameters**:
- `model_id` (str): The model identifier

**Returns**: `ModelCapabilities` object or `None` if not found

#### `validate_model_compatibility(model_id: str, required_capabilities: List[str]) -> bool`

Check if a model supports the required capabilities.

**Parameters**:
- `model_id` (str): The model identifier
- `required_capabilities` (List[str]): List of required capabilities

**Returns**: `bool` indicating compatibility

#### `get_models_by_capability(capability: str) -> List[ModelCapabilities]`

Get all models that support a specific capability.

**Parameters**:
- `capability` (str): The capability to filter by

**Returns**: List of `ModelCapabilities` objects

#### `get_category_summary() -> Dict[str, Dict[str, Any]]`

Get a summary of all model categories.

**Returns**: Dictionary with category summaries

## Configuration Builder

The `EnhancedSiliconFlowConfigBuilder` creates compatible configurations for OpenCode and Crush.

### Usage

```python
from install import EnhancedSiliconFlowConfigBuilder

builder = EnhancedSiliconFlowConfigBuilder("your-api-key")
opencode_config = builder.build_opencode_config()
crush_config = builder.build_crush_config()
```

### Methods

#### `build_opencode_config() -> Dict[str, Any]`

Generate OpenCode-compatible configuration.

**Returns**: Dictionary containing OpenCode configuration

#### `build_crush_config() -> Dict[str, Any]`

Generate Crush-compatible configuration.

**Returns**: Dictionary containing Crush configuration

#### `merge_configs_smartly(existing: Dict, new: Dict, config_type: str) -> Dict`

Intelligently merge existing configuration with new configuration.

**Parameters**:
- `existing` (Dict): Existing configuration
- `new` (Dict): New configuration to merge
- `config_type` (str): Either "opencode" or "crush"

**Returns**: Merged configuration dictionary

## API Endpoints

### Chat Completions

Create chat completions with support for streaming, tools, and reasoning.

```python
response = client.chat_completion(ChatRequest(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        ChatMessage(role="user", content="Hello!")
    ],
    stream=False,
    max_tokens=1000,
    temperature=0.7
))
```

**Supported Parameters**:
- `model`: Model identifier
- `messages`: List of chat messages
- `stream`: Enable streaming responses
- `max_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature (0.0-2.0)
- `top_p`: Nucleus sampling parameter
- `top_k`: Top-k sampling parameter
- `frequency_penalty`: Frequency penalty
- `presence_penalty`: Presence penalty
- `stop`: Stop sequences
- `enable_thinking`: Enable reasoning mode
- `thinking_budget`: Maximum reasoning tokens
- `min_p`: Minimum probability threshold
- `response_format`: Response format specification
- `tools`: List of tools for function calling

### Embeddings

Generate embeddings for text inputs.

```python
response = client.create_embeddings(EmbeddingRequest(
    model="text-embedding-ada-002",
    input=["Hello world", "How are you?"],
    encoding_format="float"
))
```

**Supported Parameters**:
- `model`: Embedding model identifier
- `input`: Text or list of texts to embed
- `encoding_format`: Encoding format ("float" or "base64")
- `dimensions`: Output dimensions (optional)
- `user`: User identifier (optional)

### Reranking

Rerank documents based on relevance to a query.

```python
response = client.create_rerank(RerankRequest(
    model="rerank-model",
    query="What is machine learning?",
    documents=["ML is a subset of AI", "Python is a programming language"],
    top_n=5
))
```

**Supported Parameters**:
- `model`: Reranking model identifier
- `query`: Search query
- `documents`: List of documents to rerank
- `top_n`: Number of top results to return
- `return_documents`: Include documents in response

### Image Generation

Generate images from text prompts.

```python
response = client.create_image(ImageGenerationRequest(
    model="black-forest-labs/FLUX.1-dev",
    prompt="A beautiful sunset over mountains",
    image_size="1024x1024",
    batch_size=1
))
```

**Supported Parameters**:
- `model`: Image generation model
- `prompt`: Text description of the image
- `negative_prompt`: What to avoid in the image
- `image_size`: Output image size
- `batch_size`: Number of images to generate
- `seed`: Random seed for reproducibility
- `num_inference_steps`: Number of denoising steps
- `guidance_scale`: Classifier-free guidance scale

### Audio Synthesis

Generate speech from text.

```python
response = client.create_speech(AudioGenerationRequest(
    model="FunAudioLLM/CosyVoice2-0.5B",
    input="Hello, world!",
    voice="alloy",
    response_format="mp3",
    speed=1.0
))
```

**Supported Parameters**:
- `model`: Speech synthesis model
- `input`: Text to synthesize
- `voice`: Voice identifier
- `response_format`: Audio format ("mp3", "wav", etc.)
- `speed`: Speech speed multiplier

### Audio Transcription

Transcribe audio files to text.

```python
response = client.create_transcription(
    audio_file=Path("recording.wav"),
    model="FunAudioLLM/SenseVoiceSmall"
)
```

**Supported Parameters**:
- `audio_file`: Path to audio file
- `model`: Transcription model

### Voice Management

Manage reference voices for speech synthesis.

```python
# Upload a voice
response = client.upload_voice(
    voice_file=Path("my_voice.wav"),
    voice_name="custom_voice"
)

# List voices
voices = client.list_voices()

# Delete a voice
client.delete_voice("voice_id")
```

### Video Generation

Generate videos from text prompts.

```python
response = client.create_video(VideoGenerationRequest(
    model="Wan-AI/Wan2.2-T2V-A14B",
    prompt="A cat playing in a garden",
    duration=5,
    aspect_ratio="16:9"
))
```

**Supported Parameters**:
- `model`: Video generation model
- `prompt`: Text description of the video
- `duration`: Video duration in seconds
- `aspect_ratio`: Video aspect ratio

### Video Status

Check the status of video generation jobs.

```python
status = client.get_video_status("video_job_id")
```

### User Information

Get user account information.

```python
info = client.get_user_info()
```

## Data Models

### ModelCapabilities

Represents the capabilities and metadata of a SiliconFlow model.

**Attributes**:
- `id` (str): Model identifier
- `name` (str): Human-readable model name
- `context_window` (int): Maximum context window size
- `supports_chat` (bool): Supports chat completions
- `supports_vision` (bool): Supports vision tasks
- `supports_audio` (bool): Supports audio tasks
- `supports_video` (bool): Supports video tasks
- `supports_embeddings` (bool): Supports embeddings
- `supports_rerank` (bool): Supports reranking
- `supports_function_calling` (bool): Supports function calling
- `supports_reasoning` (bool): Supports reasoning/thinking
- `max_tokens` (Optional[int]): Maximum output tokens
- `input_pricing` (Optional[float]): Input pricing per million tokens
- `output_pricing` (Optional[float]): Output pricing per million tokens
- `input_cached_pricing` (Optional[float]): Cached input pricing
- `output_cached_pricing` (Optional[float]): Cached output pricing

### ModelRegistry

Container for all discovered and categorized models.

**Attributes**:
- `chat_models` (ModelCategory): Chat/completion models
- `vision_models` (ModelCategory): Vision-enabled models
- `audio_models` (ModelCategory): Audio processing models
- `video_models` (ModelCategory): Video generation models
- `embedding_models` (ModelCategory): Embedding models
- `rerank_models` (ModelCategory): Reranking models
- `last_updated` (str): ISO timestamp of last update
- `api_version` (str): API version

### ModelCategory

Represents a category of models with the same capabilities.

**Attributes**:
- `name` (str): Category name
- `description` (str): Category description
- `models` (List[ModelCapabilities]): Models in this category
- `default_model` (Optional[str]): Default model ID for this category

## Error Handling

The API client includes comprehensive error handling:

### Exception Types

- `SiliconFlowAPIError`: Base exception for API errors
- `SiliconFlowRateLimitError`: Rate limit exceeded
- `SiliconFlowAuthenticationError`: Authentication failed

### Error Handling Pattern

```python
from siliconflow_client import SiliconFlowAPIError

try:
    response = client.chat_completion(request)
except SiliconFlowAPIError as e:
    print(f"API Error: {e}")
except SiliconFlowRateLimitError:
    print("Rate limit exceeded, please retry later")
except SiliconFlowAuthenticationError:
    print("Authentication failed, check your API key")
```

## Examples

### Complete Chat Example

```python
from siliconflow_client import SiliconFlowAPIClient, ChatRequest, ChatMessage

# Initialize client
client = SiliconFlowAPIClient("your-api-key")

# Create a chat request
request = ChatRequest(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Explain quantum computing in simple terms.")
    ],
    max_tokens=1000,
    temperature=0.7,
    enable_thinking=True,
    thinking_budget=500
)

# Make the request
response = client.chat_completion(request)

# Process the response
if "choices" in response and response["choices"]:
    content = response["choices"][0]["message"]["content"]
    print(f"Response: {content}")

    # Check for reasoning content
    reasoning = response["choices"][0]["message"].get("reasoning_content")
    if reasoning:
        print(f"Reasoning: {reasoning}")
```

### Model Discovery Example

```python
from model_discovery import SiliconFlowModelDiscovery

# Initialize discovery
discovery = SiliconFlowModelDiscovery("your-api-key")

# Discover all models
registry = discovery.discover_all_models()

# Get category summary
summary = discovery.get_category_summary()
for category, info in summary.items():
    print(f"{category}: {info['model_count']} models")

# Find models with specific capabilities
vision_models = discovery.get_models_by_capability("vision")
reasoning_models = discovery.get_models_by_capability("reasoning")

# Validate model compatibility
is_compatible = discovery.validate_model_compatibility(
    "Qwen/Qwen2.5-7B-Instruct",
    ["chat", "function_calling"]
)
```

### Configuration Generation Example

```python
from install import EnhancedSiliconFlowConfigBuilder

# Initialize builder
builder = EnhancedSiliconFlowConfigBuilder("your-api-key")

# Generate configurations
opencode_config = builder.build_opencode_config()
crush_config = builder.build_crush_config()

# Save configurations
import json
from pathlib import Path

# Save OpenCode config
opencode_path = Path.home() / ".config" / "opencode" / "config.json"
with open(opencode_path, 'w') as f:
    json.dump(opencode_config, f, indent=2)

# Save Crush config
crush_path = Path.home() / ".config" / "crush" / "crush.json"
with open(crush_path, 'w') as f:
    json.dump(crush_config, f, indent=2)
```

### Advanced Features Example

```python
# Using function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

request = ChatRequest(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[ChatMessage(role="user", content="What's the weather in Tokyo?")],
    tools=tools,
    tool_choice="auto"
)

response = client.chat_completion(request)

# Check for tool calls
if "choices" in response and response["choices"]:
    message = response["choices"][0]["message"]
    if "tool_calls" in message:
        for tool_call in message["tool_calls"]:
            print(f"Tool call: {tool_call['function']['name']}")
            print(f"Arguments: {tool_call['function']['arguments']}")
```

This API reference covers all major functionality of the SiliconFlow Toolkit. For more detailed examples and advanced usage patterns, refer to the test files and source code documentation.</content>
<parameter name="filePath">/media/milosvasic/DATA4TB/Projects/SuperAgent/Toolkit/SiliconFlow/API_REFERENCE.md