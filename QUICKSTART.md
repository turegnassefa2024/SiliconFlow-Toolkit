# SiliconFlow Toolkit - Quick Start Guide

This is a concise quick start guide for the SiliconFlow Toolkit. For detailed documentation, see the full documentation files.

## 🚀 Quick Installation

### 1. Get Your API Key
```bash
export SILICONFLOW_API_KEY="your-siliconflow-api-key"
```

### 2. Install the Toolkit
```bash
# Install for both OpenCode and Crush (default)
./install

# Install only for OpenCode
./install --opencode

# Install only for Crush
./install --crush

# Use custom config directory
./install --config-home /tmp/my-configs
```

### 3. Set Environment Variables
```bash
# These are set automatically, but you can set them manually
export OPENAI_API_KEY="$SILICONFLOW_API_KEY"
export OPENAI_BASE_URL="https://api.siliconflow.com/v1"
```

## 🧪 Quick Testing

### Basic Tests
```bash
# Run unit tests
python3 test_siliconflow.py

# Run compatibility tests
python3 test_compatibility.py
```

### Full Integration Tests (Requires API Key)
```bash
# Test both apps with real builds
SILICONFLOW_API_KEY=your-key python3 integration_test.py

# Test only OpenCode
SILICONFLOW_API_KEY=your-key python3 integration_test.py --opencode-only

# Keep test files for debugging
SILICONFLOW_API_KEY=your-key python3 integration_test.py --keep-temp
```

## 📁 What Gets Installed

### OpenCode Config (`~/.config/opencode/config.json`)
- 82+ SiliconFlow models
- Full API integration
- Chat, vision, audio, video support

### Crush Config (`~/.config/crush/crush.json`)
- 4 provider categories
- 97+ total models
- Automatic model selection

## 🔧 Custom Configuration

### Custom Paths
```bash
# Global config directory
./install --config-home /custom/path

# App-specific directories
./install --opencode-config-dir /custom/opencode
./install --crush-config-dir /custom/crush
```

### Force Reinstall
```bash
./install --force  # Overwrites existing configs
```

## 📊 Available Models

| Category | Count | Examples |
|----------|-------|----------|
| Chat | 77+ | Qwen2.5-7B, DeepSeek-V3, Kimi-K2 |
| Vision | 12+ | Qwen2.5-VL-7B, GLM-4V |
| Audio | 3+ | CosyVoice2, IndexTTS |
| Video | 2+ | Wan2.2-T2V, Wan2.2-I2V |

## 🛠️ Development

### Run All Tests
```bash
# Basic testing
python3 test_siliconflow.py && python3 test_compatibility.py

# Full testing (requires API key)
SILICONFLOW_API_KEY=your-key python3 integration_test.py
```

### Check Help
```bash
./install --help              # Installation options
python3 integration_test.py --help  # Integration test options
```

## 📚 Full Documentation

- `README.md` - Complete user guide
- `API_REFERENCE.md` - Technical API documentation
- `INTEGRATION_TESTS.md` - Integration testing guide
- `AGENTS.md` - Development guidelines

## 🎉 Done!

Your SiliconFlow models are now integrated with OpenCode and/or Crush! You can start using them immediately.

### Example Usage in OpenCode
- Select a SiliconFlow model from the model dropdown
- All 97+ models are available with full capabilities

### Example Usage in Crush
- SiliconFlow providers are automatically detected
- Default model is set for optimal performance

**🚀 Ready to use SiliconFlow models in your coding workflow!**