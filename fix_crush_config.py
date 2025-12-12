#!/usr/bin/env python3
"""
Quick fix for Crush "Not Found" error
"""
import json
from pathlib import Path

def fix_crush_config():
    config_path = Path.home() / ".config" / "crush" / "crush.json"
    
    if not config_path.exists():
        print(f"❌ Crush config not found at: {config_path}")
        return
    
    print(f"🔧 Fixing Crush configuration at: {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Track changes
    changes_made = False
    
    # Fix all providers with incorrect base_url
    for provider_name, provider in config.get("providers", {}).items():
        if "base_url" in provider:
            base_url = provider["base_url"]
            
            # Check if base_url contains endpoint paths (which is wrong)
            if any(endpoint in base_url for endpoint in ["/chat/completions", "/embeddings", "/rerank"]):
                # Extract just the base URL
                if "/chat/completions" in base_url:
                    new_base = base_url.replace("/chat/completions", "")
                elif "/embeddings" in base_url:
                    new_base = base_url.replace("/embeddings", "")
                elif "/rerank" in base_url:
                    new_base = base_url.replace("/rerank", "")
                elif "/v1/" in base_url and base_url.count("/") > 4:
                    # Generic fix: keep only up to /v1
                    parts = base_url.split("/")
                    new_base = "/".join(parts[:4])  # https:, , api.siliconflow.com, v1
                else:
                    new_base = "https://api.siliconflow.com/v1"
                
                print(f"  🔄 Fixing {provider_name}: {base_url} → {new_base}")
                provider["base_url"] = new_base
                changes_made = True
            elif not base_url.startswith("https://api.siliconflow.com"):
                # Also fix if it's completely wrong
                print(f"  🔄 Correcting {provider_name} base_url to standard endpoint")
                provider["base_url"] = "https://api.siliconflow.com/v1"
                changes_made = True
    
    if changes_made:
        # Backup original
        backup_path = config_path.with_suffix(".json.backup")
        import shutil
        shutil.copy2(config_path, backup_path)
        print(f"  📦 Backup created: {backup_path}")
        
        # Save fixed config
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuration fixed! Please restart Crush.")
        
        # Also create a test configuration
        create_test_config()
    else:
        print("✅ No changes needed. Configuration looks correct.")
        
        # Still verify with test
        test_api_connection(config)

def create_test_config():
    """Create a minimal test configuration"""
    test_config = {
        "$schema": "https://charm.land/crush.json",
        "providers": {
            "siliconflow-test": {
                "name": "SiliconFlow Test",
                "type": "openai-compat",
                "api_key": "YOUR_API_KEY_HERE",  # You need to replace this
                "base_url": "https://api.siliconflow.com/v1",
                "models": [
                    {
                        "id": "Qwen/Qwen2.5-7B-Instruct",
                        "name": "Qwen2.5 7B (Test)",
                        "context_window": 32768,
                        "default_max_tokens": 512
                    }
                ]
            }
        },
        "defaultProvider": "siliconflow-test",
        "sampling": {
            "temperature": 0.3,
            "max_tokens": 512,
            "stream": True
        }
    }
    
    test_path = Path.home() / ".config" / "crush" / "test_config.json"
    with open(test_path, "w") as f:
        json.dump(test_config, f, indent=2)
    
    print(f"📝 Test config created: {test_path}")
    print("💡 Replace 'YOUR_API_KEY_HERE' with your actual API key to test")

def test_api_connection(config):
    """Test the API connection with curl command"""
    print("\n🔍 Testing API configuration...")
    
    # Try to find an API key
    api_key = None
    for provider in config.get("providers", {}).values():
        if "api_key" in provider:
            api_key = provider["api_key"]
            break
    
    if api_key:
        # Show curl test command
        print(f"📡 API Key found (preview: {api_key[:8]}...{api_key[-4:]})")
        print("\n🧪 Test with this curl command:")
        print(f"curl -X POST \\")
        print(f'  "https://api.siliconflow.com/v1/chat/completions" \\')
        print(f'  -H "Authorization: Bearer {api_key}" \\')
        print(f'  -H "Content-Type: application/json" \\')
        print(f'  -d \'{{"model": "Qwen/Qwen2.5-7B-Instruct", "messages": [{{"role": "user", "content": "Hello"}}], "max_tokens": 10}}\'')
    else:
        print("❓ No API key found in configuration")

def main():
    print("🚀 Crush Configuration Fix Tool")
    print("=" * 50)
    fix_crush_config()

if __name__ == "__main__":
    main()
